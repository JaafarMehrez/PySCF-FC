import pyscf
import numpy as np
import warnings
from functools import reduce
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask

from pyscf import lib, ao2mo
from pyscf.scf import atom_hf
from pyscf import __config__

np.set_printoptions(threshold=np.inf)

def freezeCore(oneBody_a, oneBody_b, twoBody_a, twoBody_b, twoBody_ab, core, valence=None):
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
    
    
    constant  = np.einsum('ii->', oneBody_a[np.ix_(core_idx, core_idx)])
    constant += np.einsum('ii->', oneBody_b[np.ix_(core_idx, core_idx)])
    
    core_aa = twoBody_a[np.ix_(core_idx, core_idx, core_idx, core_idx)]
    core_bb = twoBody_b[np.ix_(core_idx, core_idx, core_idx, core_idx)]
    core_ab = twoBody_ab[np.ix_(core_idx, core_idx, core_idx, core_idx)]
    
    constant += 0.5 * ( np.einsum('iijj->', core_aa) - np.einsum('ijji->', core_aa)
               + np.einsum('iijj->', core_bb) - np.einsum('ijji->', core_bb)
               + 2.0 * np.einsum('iijj->', core_ab) )
    
    print(constant)

    h_active = oneBody[np.ix_(valence_idx, valence_idx)].copy()
    coul_block = twoBody[np.ix_(valence_idx, valence_idx, core_idx, core_idx)]
    h_active += 2.0 * np.einsum('pqkk->pq', coul_block)
    exch_block = twoBody[np.ix_(valence_idx, core_idx, core_idx, valence_idx)]
    h_active -= np.einsum('pkkq->pq', exch_block)

    twoBody_active = twoBody[np.ix_(valence_idx, valence_idx, valence_idx, valence_idx)].copy()

    return h_active_a, h_active_b, twoBody_active_a, twoBody_active_b, twoBody_active_ab, constant

def main():
    from pyscf import gto, scf, ao2mo
    name = 'out'
    mol = pyscf.M(
        atom='N',
        unit='angstrom',
        basis={'N': parse_gaussian.load('N-STO-3G-EMSL.gbs', 'N')},
        charge=0,
        spin=3,
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
    mymf = mol.UHF().set(
        conv_tol=1e-14,
        max_cycle=9999,
        ddm_tol=1e-15,
        direct_scf=False,
        chkfile=name + '.chk',
        init_guess='atom',
        irrep_nelec={'Ag': 4, 'B3u': 1, 'B2u': 1, 'B1u': 1}
    )
    ekrhf = mymf.kernel()
    atom_hf.AtomSphAverageRHF = original_AtomSphAverageRHF
    
    from pyscf import cc
    mycc = cc.UCCSD(mymf, frozen=1)

    def compute_mo_irreps(mol, mo_coeff):
        symm_orbs = mol.symm_orb
        irrep_labels = mol.irrep_name
        mo_irreps = []
        for mo in mo_coeff.T:
            projections = [np.linalg.norm(symm_orbs[i].T @ mo) for i in range(len(symm_orbs))]
            irrep_idx = np.argmax(projections)
            mo_irreps.append(irrep_labels[irrep_idx])
        return mo_irreps

    def align_beta_orbitals_to_alpha(mol, mo_coeff):
        alpha_orbs, beta_orbs = mo_coeff[0], mo_coeff[1]
        alpha_irreps = compute_mo_irreps(mol, mo_coeff[0])
        beta_irreps = compute_mo_irreps(mol, mo_coeff[1])
        beta_orbs_sorted = []
        used_indices = set()
        for target_irrep in alpha_irreps:
            for idx, beta_irrep in enumerate(beta_irreps):
                if beta_irrep == target_irrep and idx not in used_indices:
                    beta_orbs_sorted.append(beta_orbs[:, idx])
                    used_indices.add(idx)
                    break
            else:
                raise ValueError(f"No matching beta orbital found for alpha irrep: {target_irrep}")
        beta_orbs_sorted = np.column_stack(beta_orbs_sorted)
        return alpha_orbs, beta_orbs_sorted

    mo_coeff = mymf.mo_coeff
    mol = mymf.mol
    assert mo_coeff[0].dtype == np.double and mo_coeff[1].dtype == np.double
    orbsym_full = getattr(mo_coeff, 'orbsym', None)
    nuc = mymf.energy_nuc()

    mo_coeff_a, mo_coeff_b = align_beta_orbitals_to_alpha(mol, mo_coeff)
    
    h1e_a  = reduce(np.dot, (mo_coeff_a.T, mymf.get_hcore(), mo_coeff_a))
    h1e_b  = reduce(np.dot, (mo_coeff_b.T, mymf.get_hcore(), mo_coeff_b))
    
    #mymf._eri = mol.intor('int2e_sph', aosym='s8')
    eri_a  = ao2mo.restore(1,ao2mo.incore.general(mymf._eri,(mo_coeff[0], mo_coeff[0], mo_coeff[0], mo_coeff[0]),compact=False),h1e_a.shape[0])
    eri_b  = ao2mo.restore(1,ao2mo.incore.general(mymf._eri,(mo_coeff[1], mo_coeff[1], mo_coeff[1], mo_coeff[1]),compact=False),h1e_a.shape[0])
    eri_ab = ao2mo.restore(1,ao2mo.incore.general(mymf._eri,(mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]),compact=False),h1e_a.shape[0])
    
    nmo_full = mymf.mo_coeff[0].shape
    nmo = nmo_full

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

    #h1e_a, h1e_b, eri_a, eri_b, eri_ab, constant = freezeCore(h1e_a, h1e_b, eri_a, eri_b, eri_ab,frozen_core=(frozen_core_a, frozen_core_b),active=(act_a, act_b),)
    h1e_a, h1e_b, eri_a, eri_b, eri_ab, constant = freezeCore(h1e_a, h1e_b, eri_a, eri_b, eri_ab,frozen_core, active)
    
    nmo_active = h1e_a.shape[0]

    if orbsym_full is None:
        orbsym_active = None
    else:
        orbsym_arr = np.asarray(orbsym_full, dtype=int)
        valence_idx = np.where(active)[0]
        orbsym_active = [int(x) for x in orbsym_arr[valence_idx]]
        
    if orbsym_active is not None:
        if len(orbsym_active) != nmo_active:
            raise RuntimeError(f"Length mismatch: orbsym_active has length {len(orbsym_active)} "
                               f"but number of active orbitals is {nmo_active}.")

    filename = 'fort.55'
    nelec_full = mol.nelectron
    spin = mol.spin       
    nalpha_full = (nelec_full + spin) // 2
    nbeta_full  = (nelec_full - spin) // 2
    nocc_full = mol.nelectron // 2
    nfrozen_core = int(np.count_nonzero(frozen_core[:nocc_full]))
    nalpha_active = nalpha_full - nfrozen_core
    nbeta_active  = nbeta_full  - nfrozen_core
    nelec_active = (nalpha_active, nbeta_active)
    
    DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
    TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)
    
    def write_hcore_uhf(fout, h1e_a, h1e_b, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
        h1e_a = h1e_a.reshape(nmo,nmo)
        h1e_b = h1e_b.reshape(nmo,nmo)
        indx = [i+1 for i in range(nmo)]
        output_format = float_format + ' %5d %5d     0     0\n'
        for i in range(nmo):
            for j in range(nmo):
                fout.write(output_format % (h1e_a[i,j], indx[i], indx[j]))
        fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
        for i in range(nmo):
            for j in range(nmo):
                fout.write(output_format % (h1e_b[i,j], indx[i], indx[j]))
        fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
    
    def write_eri_uhf(fout, eri_a, eri_b, eri_ab, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
        npair = nmo*(nmo+1)//2
        output_format = float_format + ' %5d %5d %5d %5d\n'
        indx = [i+1 for i in range(nmo)]
        if all(x.ndim == 2 for x in (eri_a, eri_b, eri_ab)): # 4-fold symmetry
            assert all(x.ndim == 2 and x.size == npair**2 for x in (eri_a, eri_b, eri_ab))
            kl = 0
            for l in range(nmo):
                for k in range(0, l+1):
                    ij = 0
                    for i in range(0, nmo):
                        for j in range(0, i+1):
                            if i >= k:
                                if abs(eri_a[ij,kl]) > tol:
                                    fout.write(output_format % (eri_a[ij,kl], indx[i], indx[j], indx[k], indx[l]))
                            ij += 1
                    kl += 1
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            kl = 0
            for l in range(nmo):
                for k in range(0, l+1):
                    ij = 0
                    for i in range(0, nmo):
                        for j in range(0, i+1):
                            if i >= k:
                                if abs(eri_b[ij,kl]) > tol:
                                    fout.write(output_format % (eri_b[ij,kl], indx[i], indx[j], indx[k], indx[l]))
                            ij += 1
                    kl += 1
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            ij = 0
            for j in range(nmo):
                for i in range(0, j+1):
                    kl = 0
                    for k in range(nmo):
                        for l in range(0, k+1):
                            if abs(eri_ab[ij,kl]) > tol:
                                fout.write(output_format % (eri_ab[ij,kl], indx[i], indx[j], indx[k], indx[l]))
                            kl += 1
                    ij +=1
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
    
    def write_head(fout, nmo, nelec, ms=0, orbsym=None):
        is_uhf = isinstance(nelec, (list, tuple)) and len(nelec) == 2 and nelec[0] != nelec[1]
        if not isinstance(nelec, (int, np.number)):
            ms = abs(nelec[0] - nelec[1])
            nelec = nelec[0] + nelec[1]
        fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
        if orbsym is not None and len(orbsym) > 0:
            fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
        else:
            fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
        fout.write('  ISYM=1,\n')
        if is_uhf:
            fout.write('  IUHF=1,\n')
        fout.write(' &END\n')
    
    
    def write_head55(fout, nmo, nelec, ms=0, orbsym=None):
        if not isinstance(nelec, (int, np.number)):
            ms = abs(nelec[0] - nelec[1])
            nelec = nelec[0] + nelec[1]
        fout.write(f"{nmo:1d} {nelec:1d}\n")
        if orbsym is not None and len(orbsym) > 0:
            orbsym = [x + 1 for x in orbsym]
            fout.write(f"{' '.join([str(x) for x in orbsym])}\n")
        else:
            fout.write(f"{' 1' * nmo}\n")
        fout.write(' 150000\n')
    
    
    def from_integrals_uhf(filename, h1e_a, h1e_b, eri_a, eri_b, eri_ab, nmo, nelec, nuc=0, ms=0, orbsym=None,
                       tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
        '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
        with open(filename, 'w') as fout:
            if filename == 'fort.55':
                write_head55(fout, nmo, nelec, ms, orbsym)
            else:
                write_head(fout, nmo, nelec, ms, orbsym)
            write_eri_uhf(fout, eri_a, eri_b, eri_ab, nmo, tol=tol, float_format=float_format)
            write_hcore_uhf(fout, h1e_a, h1e_b, nmo, tol=tol, float_format=float_format)
            output_format = float_format + '     0     0     0     0\n'
            fout.write(output_format % nuc)
    
    
    
    
    from_integrals_uhf(filename, h1e_a, h1e_b, eri_a, eri_b, eri_ab, nmo_active, nelec_active, nuc + constant, 0, orbsym_active,tol=1e-18, float_format='% 0.20E')


if __name__ == '__main__':
    main()

