#!/usr/bin/env python

import pyscf
import numpy as np
from functools import reduce
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask
from pyscf.tools.fcidump import from_integrals
from pyscf import lib
from pyscf.scf import atom_hf

# Ensure full numpy array printing for debugging if needed
np.set_printoptions(threshold=np.inf)

def freezeCore(oneBody, twoBody, core, valence=None):
    """
    Apply the frozen core approximation to the Hamiltonian.

    Args:
        oneBody (ndarray): Full one-body Hamiltonian in MO basis (nmo, nmo).
        twoBody (ndarray): Full two-body ERI tensor in MO basis (nmo, nmo, nmo, nmo),
                          in chemists' notation (pq|rs).
        core (ndarray): Boolean array, True for frozen core orbitals.
        valence (ndarray, optional): Boolean array, True for active orbitals.
                                     If None, taken as ~core.

    Returns:
        tuple: (oneBody_active, twoBody_active, constant)
               - oneBody_active: One-body Hamiltonian in active space.
               - twoBody_active: Two-body Hamiltonian in active space.
               - constant: Energy shift due to core orbitals.
    """
    # Determine valence mask
    if valence is None:
        valence = ~core
    else:
        # Ensure no overlap between core and valence
        assert not (core & valence).any(), "Orbitals cannot be both core and valence."

    # Get indices for core and valence orbitals
    core_idx = np.where(core)[0]
    print(core_idx)
    valence_idx = np.where(valence)[0]
    print(core_idx)

    # Compute constant energy shift from core orbitals
    # E_core = 2 * sum_i h_{ii} + sum_{i,j} [2 (ii|jj) - (ij|ji)]
    constant = 2. * np.einsum('ii->', oneBody[np.ix_(core_idx, core_idx)])
    constant += 2. * np.einsum('iijj->', twoBody[np.ix_(core_idx, core_idx, core_idx, core_idx)])
    constant -= np.einsum('ijji->', twoBody[np.ix_(core_idx, core_idx, core_idx, core_idx)])

    # Compute effective one-body Hamiltonian in active space
    # h_{pq}^eff = h_{pq} + sum_k [2 (pk|qk) - (pk|kq)]
    oneBody_active = oneBody[np.ix_(valence_idx, valence_idx)]
    oneBody_active += 2 * np.einsum('pkqk->pq', twoBody[np.ix_(valence_idx, core_idx, valence_idx, core_idx)])
    oneBody_active -= np.einsum('pkkq->pq', twoBody[np.ix_(valence_idx, core_idx, core_idx, valence_idx)])

    # Extract two-body integrals for active space
    twoBody_active = twoBody[np.ix_(valence_idx, valence_idx, valence_idx, valence_idx)]

    return oneBody_active, twoBody_active, constant

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    """
    Custom header writer for FCIDUMP file.

    Args:
        fout (file): File object to write to.
        nmo (int): Number of molecular orbitals.
        nelec (int or tuple): Number of electrons (total or (alpha, beta)).
        ms (int): Spin multiplicity - 1 (default 0 for singlet).
        orbsym (list): Orbital symmetry labels.
    """
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

# Override PySCF's default FCIDUMP header writer
from pyscf.tools import fcidump
fcidump.write_head = write_head

def main():
    """
    Main function to set up the molecule, perform RHF, apply frozen core approximation,
    and generate the FCIDUMP file.
    """
    from pyscf import gto, scf, ao2mo

    # Output filename prefix
    name = 'out'

    # Define the molecule (Neon atom as an example)
    mol = pyscf.M(
        atom='NE',
        unit='angstrom',
        basis={'NE': parse_gaussian.load('NE-aVDZ-EMSL.gbs', 'NE')},
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

    # Set up RHF calculation
    mymf = mol.RHF().set(
        conv_tol=1e-10,
        max_cycle=999,
        ddm_tol=1e-14,
        direct_scf=False,
        chkfile=name + '.chk',
        init_guess='atom',
        irrep_nelec={'Ag': 4, 'B3u': 2, 'B2u': 2, 'B1u': 2}
    )
    ekrhf = mymf.kernel()
    atom_hf.AtomSphAverageRHF = original_AtomSphAverageRHF

    # Define active space for frozen core
    nocc = mymf.mol.nelectron // 2  # Number of occupied orbitals
    nao, nmo = mymf.mo_coeff.shape  # Number of AOs and MOs
    nvir = nmo - nocc  # Number of virtual orbitals
    nocc_active = 4  # Number of active occupied orbitals
    nvir_active = 18  # Number of active virtual orbitals
    frozen = list(range(0, nocc - nocc_active)) + list(range(nocc + nvir_active, nmo))
    if len(frozen) == 0:
        frozen = None

    # Set up CCSD with frozen orbitals to get active space mask
    from pyscf.cc import ccsd
    mycc = ccsd.RCCSD(mymf, frozen=frozen)

    # Get MO coefficients and symmetry
    mo_coeff = mymf.mo_coeff
    assert mo_coeff.dtype == np.double
    orbsym = getattr(mo_coeff, 'orbsym', None)
    nuc = mymf.energy_nuc()

    # Compute one-body Hamiltonian in MO basis
    h1e = reduce(np.dot, (mo_coeff.T, mymf.get_hcore(), mo_coeff))

    # Compute two-body Hamiltonian (ERI tensor) in MO basis
    nmo = mo_coeff.shape[1]
    mo = np.asarray(mo_coeff, order='F')
    eri = pyscf.ao2mo.restore(1, pyscf.ao2mo.incore.general(mymf._eri, (mo_coeff, mo_coeff, mo_coeff, mo_coeff), compact=False), nmo)

    # Define active and frozen core orbitals
    active = get_frozen_mask(mycc)
    frozen_core = np.zeros_like(active, dtype=np.bool_)
    nocc = mol.nelectron // 2
    frozen_core[:nocc] = ~active[:nocc]

    # Apply frozen core approximation
    h1e, h2e, constant = freezeCore(h1e, eri, frozen_core, active)

    # Update nmo to number of active orbitals
    nmo = h1e.shape[0]

    # Write FCIDUMP file
    filename = 'fort.55'
    nelec = (mycc.nocc, mycc.nocc)  # Number of alpha and beta electrons in active space
    from_integrals(filename, h1e, h2e, nmo, nelec, nuc + constant, 0, orbsym, tol=1e-18, float_format='% 0.20E')

if __name__ == '__main__':
    main()
