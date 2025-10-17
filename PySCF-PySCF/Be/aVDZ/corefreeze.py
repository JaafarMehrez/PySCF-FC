import pyscf
import numpy as np
from functools import reduce
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask
from pyscf.tools.fcidump import from_integrals
from pyscf import lib
from pyscf.scf import atom_hf

np.set_printoptions(threshold=np.inf)


def freezeCore(oneBody, twoBody, core, valence=None):
    """
    Apply frozen-core approximation to integrals.

    Args:
        oneBody: one-electron integrals in MO basis (nmo x nmo)
        twoBody: two-electron integrals in MO basis (nmo x nmo x nmo x nmo)
        core: boolean mask (length nmo) True for core (frozen) orbitals
        valence: boolean mask True for valence (active) orbitals; if None use ~core

    Returns:
        h_active, twoBody_active, constant (scalar energy from frozen core)
    """
    if valence is None:
        valence = ~core
    else:
        assert not (core & valence).any(), "Orbitals cannot be both core and valence."

    core_idx = np.where(core)[0]
    valence_idx = np.where(valence)[0]

    constant = 2.0 * np.einsum('ii->', oneBody[np.ix_(core_idx, core_idx)])
    core_core = twoBody[np.ix_(core_idx, core_idx, core_idx, core_idx)]
    constant += 2.0 * np.einsum('iijj->', core_core)
    constant -= 1.0 * np.einsum('ijji->', core_core)

    h_active = oneBody[np.ix_(valence_idx, valence_idx)].copy()
    coul_block = twoBody[np.ix_(valence_idx, valence_idx, core_idx, core_idx)]
    h_active += 2.0 * np.einsum('pqkk->pq', coul_block)
    exch_block = twoBody[np.ix_(valence_idx, core_idx, core_idx, valence_idx)]
    h_active -= np.einsum('pkkq->pq', exch_block)

    twoBody_active = twoBody[np.ix_(valence_idx, valence_idx, valence_idx, valence_idx)].copy()

    return h_active, twoBody_active, constant


def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, np.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(f" {nmo:4d} {nelec:2d}\n")
    if orbsym is not None and len(orbsym) > 0:
        orbsym = [x + 1 for x in orbsym]
        fout.write(f"{' '.join([str(x) for x in orbsym])}\n")
    else:
        fout.write(f"{' 1' * nmo}\n")
    fout.write(' 150000\n')

from pyscf.tools import fcidump
fcidump.write_head = write_head


def main():
    from pyscf import gto, scf, ao2mo
    name = 'out'
    mol = pyscf.M(
        atom='BE',
        unit='angstrom',
        basis={'BE': parse_gaussian.load('BE-STO-3G-EMSL.gbs', 'BE')},
        charge=0,
        spin=0,
        verbose=9,
        symmetry=True,
        output=name + '.txt',
        symmetry_subgroup='D2h',
        max_memory=4000,
    )
    original_AtomSphAverageRHF = atom_hf.AtomSphAverageRHF
    class CustomAtomSphAverageRHF(original_AtomSphAverageRHF):
        def __init__(self, mol):
            super().__init__(mol)
            self.max_cycle = 9999
            self.direct_scf = False
    atom_hf.AtomSphAverageRHF = CustomAtomSphAverageRHF
    mymf = mol.RHF().set(
        conv_tol=1e-14,
        max_cycle=9999,
        ddm_tol=1e-15,
        direct_scf=False,
        chkfile=name + '.chk',
        init_guess='atom',
        irrep_nelec={'Ag': 4}
    )
    ekrhf = mymf.kernel()
    atom_hf.AtomSphAverageRHF = original_AtomSphAverageRHF
    nocc = mymf.mol.nelectron // 2
    nao, nmo_full = mymf.mo_coeff.shape
    nvir = nmo_full - nocc

    nocc_active = 1
    nvir_active = 3

    frozen = list(range(0, nocc - nocc_active)) + list(range(nocc + nvir_active, nmo_full))
    if len(frozen) == 0:
        frozen = None

    from pyscf.cc import ccsd
    mycc = ccsd.RCCSD(mymf, frozen=frozen)
    
    mo_coeff = mymf.mo_coeff
    assert mo_coeff.dtype == np.double
    orbsym_full = getattr(mo_coeff, 'orbsym', None)
    nuc = mymf.energy_nuc()
    h1e = reduce(np.dot, (mo_coeff.T, mymf.get_hcore(), mo_coeff))
    nmo_full = mo_coeff.shape[1]
    mo = np.asarray(mo_coeff, order='F')
    eri = pyscf.ao2mo.restore(1,pyscf.ao2mo.incore.general(mymf._eri, (mo_coeff, mo_coeff, mo_coeff, mo_coeff),compact=False), nmo_full)

    active = get_frozen_mask(mycc)
    frozen_core = np.zeros_like(active, dtype=np.bool_)
    nocc_full = mol.nelectron // 2
    frozen_core[:nocc_full] = ~active[:nocc_full]

    if orbsym_full is None:
        orbsym_active = None
    else:
        orbsym_arr = np.asarray(orbsym_full, dtype=int)
        valence_idx = np.where(active)[0]
        orbsym_active = [int(x) for x in orbsym_arr[valence_idx]]

    h1e_active, h2e_active, constant = freezeCore(h1e, eri, frozen_core, active)
    nmo_active = h1e_active.shape[0]

    if orbsym_active is not None:
        if len(orbsym_active) != nmo_active:
            raise RuntimeError(f"Length mismatch: orbsym_active has length {len(orbsym_active)} "
                               f"but number of active orbitals is {nmo_active}.")

    filename = 'fort.55'
    #nelec = (mycc.nocc, mycc.nocc)
    nelec_full = mol.nelectron
    spin = mol.spin       
    nalpha_full = (nelec_full + spin) // 2
    nbeta_full  = (nelec_full - spin) // 2
    nocc_full = mol.nelectron // 2
    nfrozen_core = int(np.count_nonzero(frozen_core[:nocc_full]))
    nalpha_active = nalpha_full - nfrozen_core
    nbeta_active  = nbeta_full  - nfrozen_core
    nelec_active = (nalpha_active, nbeta_active)
    from_integrals(filename, h1e_active, h2e_active, nmo_active, nelec_active, nuc + constant, 0, orbsym_active,
                   tol=1e-18, float_format='% 0.20E')

if __name__ == '__main__':
    main()

