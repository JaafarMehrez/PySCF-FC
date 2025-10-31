import pyscf
import numpy as np
import warnings
import pyscf.symm.param as param
import pyscf.lib.logger as logger
import pandas as pd
from functools import reduce
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask
from pyscf import lib, ao2mo
from pyscf.scf import atom_hf
from pyscf import symm
from pyscf import __config__

name = 'out'
mol = pyscf.M(
    atom = '''
        O
    ''',
    unit = 'angstrom',
    basis = {
            'O' : parse_gaussian.load('O-aCVDZ-EMSL.gbs', 'O')
    },
    charge = 0,
    spin = 2,
    symmetry = True,
    verbose = 9,
    symmetry_subgroup = 'D2h',
    output = name +'.txt',
    max_memory = 4000,
)
original_AtomSphAverageRHF = atom_hf.AtomSphAverageRHF
class CustomAtomSphAverageRHF(original_AtomSphAverageRHF):
    def __init__(self, mol):
        super().__init__(mol)
        self.max_cycle = 9999
        self.direct_scf = False
atom_hf.AtomSphAverageRHF = CustomAtomSphAverageRHF
mf = mol.UHF().set(
    conv_tol=1e-14,
    max_cycle=9999,
    ddm_tol=1e-15,
    direct_scf=False,
    chkfile=name + '.chk',
    init_guess='atom',
    irrep_nelec={'Ag': 4, 'B3u': 2, 'B2u': 1, 'B1u': 1}
)
mf.kernel()
atom_hf.AtomSphAverageRHF = original_AtomSphAverageRHF

def as_spin_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    xa = np.asarray(x)
    if xa.ndim == 1:
        return (xa,)
    if xa.ndim == 2 and xa.shape[0] in (2,):
        return tuple(xa[i] for i in range(xa.shape[0]))
    return (xa,)

def build_mo_table(mf, check_orbsym=False):
    
    mol = mf.mol
    s = mol.intor_symmetric('int1e_ovlp')
    mo_coeffs = as_spin_tuple(mf.mo_coeff)
    mo_energies = as_spin_tuple(mf.mo_energy)
    mo_occs = as_spin_tuple(mf.mo_occ)
    
    spins = ('alpha','beta') if len(mo_coeffs) == 2 else ('alpha',)
    
    rows = []
    for spin_name, coeff_mat, energies, occs in zip(spins, mo_coeffs, mo_energies, mo_occs):
        coeff_mat = np.asarray(coeff_mat)
        energies = np.asarray(energies)  
        occs = np.asarray(occs)          
        
        orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                     coeff_mat, s=s, check=check_orbsym)
        orbsym = np.asarray(orbsym)
        
        nmo = energies.shape[0]
        for mo_idx in range(nmo):
            rows.append({
                'spin': spin_name,
                'spin_index': 0 if spin_name=='alpha' else 1,
                'mo_index': int(mo_idx),
                'energy': float(energies[mo_idx]),
                'occ': float(occs[mo_idx]),
                'irrep': str(orbsym[mo_idx]),
                'coeff': coeff_mat[:, mo_idx],
            })
            
    df = pd.DataFrame(rows)
    df['is_occupied'] = df['occ'] > 0.5
    df = df.sort_values(['spin','mo_index']).reset_index(drop=True)
    return df
    
def reorder_mf_mos(mf, ordered_pairs):
    
    mo_coeffs = as_spin_tuple(mf.mo_coeff)
    mo_energies = as_spin_tuple(mf.mo_energy)
    mo_occs = as_spin_tuple(mf.mo_occ)
    spins = ('alpha','beta') if len(mo_coeffs)==2 else ('alpha',)
    
    per_spin_order = {s:[] for s in spins}
    for spin, moidx in ordered_pairs:
        per_spin_order[spin].append(int(moidx))
        
    new_coeffs = []
    new_energies = []
    new_occs = []
    for s, coeff_mat, energies, occs in zip(spins, mo_coeffs, mo_energies, mo_occs):
        order = per_spin_order.get(s, [])
        if len(order)==0:
            new_coeffs.append(coeff_mat.copy())
            new_energies.append(np.array(energies).copy())
            new_occs.append(np.array(occs).copy())
            continue
        coeff_mat = np.asarray(coeff_mat)
        new_coeffs.append(coeff_mat[:, order].copy())
        new_energies.append(np.asarray(energies)[order].copy())
        new_occs.append(np.asarray(occs)[order].copy())
        
    if len(new_coeffs) == 1:
        return new_coeffs[0], new_energies[0], new_occs[0]
    return tuple(new_coeffs), tuple(new_energies), tuple(new_occs)

'''
Example usage:

df = build_mo_table(mf, check_orbsym=False)
new_coeffs, new_energies, new_occs = reorder_mf_mos(mf, ordered)
'''

def pair_and_sort_irrep_mos(df, pairing='zip'):
    
    pairs = []
    irr_names = list(dict.fromkeys(df['irrep']))
    
    alpha_df = df[df['spin']=='alpha'].copy().sort_values(['irrep','energy'], ascending=[True, True])
    beta_df  = df[df['spin']=='beta'].copy().sort_values(['irrep','energy'], ascending=[True, True])
    
    for ir in irr_names:
        a_block = alpha_df[alpha_df['irrep']==ir]
        b_block = beta_df[beta_df['irrep']==ir]
        
        a_list = list(a_block[['mo_index','energy']].itertuples(index=False, name=None))
        b_list = list(b_block[['mo_index','energy']].itertuples(index=False, name=None))
        
        if pairing == 'zip':
            n = max(len(a_list), len(b_list))
            for i in range(n):
                ai = a_list[i] if i < len(a_list) else (None, None)
                bi = b_list[i] if i < len(b_list) else (None, None)
                a_idx, a_e = ai
                b_idx, b_e = bi
                energies = [e for e in (a_e, b_e) if e is not None]
                score = float(np.mean(energies)) if energies else float('inf')
                pairs.append({
                    'irrep': ir,
                    'alpha_idx': None if a_idx is None else int(a_idx),
                    'beta_idx' : None if b_idx is None else int(b_idx),
                    'alpha_energy': None if a_e is None else float(a_e),
                    'beta_energy' : None if b_e is None else float(b_e),
                    'score': score,
                })
        elif pairing == 'nearest':
            a_remaining = a_list.copy()
            b_remaining = b_list.copy()
            while a_remaining and b_remaining:
                a_energies = [e for (_, e) in a_remaining]
                b_energies = [e for (_, e) in b_remaining]
                min_a = a_remaining[np.argmin(a_energies)]
                min_b = b_remaining[np.argmin(b_energies)]
                if min_a[1] <= min_b[1]:
                    a_idx, a_e = min_a
                    b_diffs = [abs(e - a_e) for (_, e) in b_remaining]
                    j = int(np.argmin(b_diffs))
                    b_idx, b_e = b_remaining.pop(j)
                    a_remaining.remove(min_a)
                else:
                    b_idx, b_e = min_b
                    a_diffs = [abs(e - b_e) for (_, e) in a_remaining]
                    j = int(np.argmin(a_diffs))
                    a_idx, a_e = a_remaining.pop(j)
                    b_remaining.remove(min_b)
                score = float(np.mean([a_e, b_e]))
                pairs.append({
                    'irrep': ir,
                    'alpha_idx': int(a_idx),
                    'beta_idx' : int(b_idx),
                    'alpha_energy': float(a_e),
                    'beta_energy' : float(b_e),
                    'score': score,
                })
            for a_idx, a_e in a_remaining:
                pairs.append({'irrep': ir, 'alpha_idx': int(a_idx), 'beta_idx': None,
                              'alpha_energy': float(a_e), 'beta_energy': None, 'score': float(a_e)})
            for b_idx, b_e in b_remaining:
                pairs.append({'irrep': ir, 'alpha_idx': None, 'beta_idx': int(b_idx),
                              'alpha_energy': None, 'beta_energy': float(b_e), 'score': float(b_e)})
        else:
            raise ValueError("pairing must be 'zip' or 'nearest'")
        
    for p in pairs:
        
        energies = [e for e in (p['alpha_energy'], p['beta_energy']) if e is not None]
        if not energies:
            
            p['is_virtual_pair'] = True
            p['sort_key'] = (1, float('inf'))
            continue
        
        is_virtual_pair = all(e > 0.0 for e in energies)
        mean_e = float(np.mean(energies))
        min_e = float(min(energies))
        if is_virtual_pair:
            
            p['is_virtual_pair'] = True
            p['sort_key'] = (1, min_e)
        else:
            
            p['is_virtual_pair'] = False
            p['sort_key'] = (0, mean_e)
            
    pairs = sorted(pairs, key=lambda x: x['sort_key'])
    
    ordered_pairs_flat = []
    for p in pairs:
        if p['alpha_idx'] is not None:
            ordered_pairs_flat.append(('alpha', p['alpha_idx']))
        if p['beta_idx'] is not None:
            ordered_pairs_flat.append(('beta', p['beta_idx']))
            
    return pairs, ordered_pairs_flat

def print_pairs_table(pairs):
    
    print(f"{'Row':>3s} {'Irrep':>6s} {'Alpha':>12s} {'Beta':>12s} {'Score':>12s}")
    print("-"*52)
    for i, p in enumerate(pairs, start=1):
        a = f"{p['alpha_energy']: .6f}" if p['alpha_energy'] is not None else "  -----"
        b = f"{p['beta_energy']: .6f}"  if p['beta_energy']  is not None else "  -----"
        sc = f"{p['score']: .6f}" if p['score'] is not None else "  -----"
        print(f"{i:3d} {p['irrep']:>6s} {a:>12s} {b:>12s} {sc:>12s}")
        
df = build_mo_table(mf, check_orbsym=False)
pairs, order = pair_and_sort_irrep_mos(df, pairing='nearest')
new_coeffs, new_energies, new_occs = reorder_mf_mos(mf, order)
print_pairs_table(pairs)

filename  = 'fort.55'
nelec     = mol.nelectron

orbs   = new_coeffs
orbs_a = orbs[0]
orbs_b = orbs[1]

if mol.symmetry:
        groupname = mol.groupname
        if groupname in ('SO3', 'Dooh'):
            logger.info(mol, 'Lower symmetry from %s to D2h', groupname)
            raise RuntimeError('Lower symmetry from %s to D2h' % groupname)
        elif groupname == 'Coov':
            logger.info(mol, 'Lower symmetry from Coov to C2v')
            raise RuntimeError('''Lower symmetry from Coov to C2v''')
        
orbsym = pyscf.symm.label_orb_symm(mol,mol.irrep_name,mol.symm_orb,orbs_a)
orbsym = np.array(orbsym)
orbsym = [param.IRREP_ID_TABLE[groupname][i] for i in orbsym]

nmo    = orbs_a.shape[1]
eri_a  = pyscf.ao2mo.restore(4,pyscf.ao2mo.incore.general(mf._eri, (orbs_a,orbs_a,orbs_a,orbs_a), compact=False),nmo)
eri_b  = pyscf.ao2mo.restore(4,pyscf.ao2mo.incore.general(mf._eri, (orbs_b,orbs_b,orbs_b,orbs_b), compact=False),nmo)
eri_ab = pyscf.ao2mo.restore(4,pyscf.ao2mo.incore.general(mf._eri, (orbs_a,orbs_a,orbs_b,orbs_b), compact=False),nmo)

h_core = mf.get_hcore(mol)
h1e_a  = reduce(np.dot, (orbs_a.T, h_core, orbs_a))
h1e_b  = reduce(np.dot, (orbs_b.T, h_core, orbs_b))

nuc = mol.energy_nuc()
float_format = '% 0.20E'

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)

def write_hcore_uhf(fout, h1e_a, h1e_b, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
        h1e_a = h1e_a.reshape(nmo,nmo)
        h1e_b = h1e_b.reshape(nmo,nmo)
        indx = [i+1 for i in range(nmo)]
        output_format = float_format + ' %5d %5d     0     0\n'
        for i in range(nmo):
            for j in range(i, nmo):
                if abs(h1e_a[i,j]) > TOL:
                    fout.write(output_format % (h1e_a[i,j], indx[i], indx[j]))
        fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
        for i in range(nmo):
            for j in range(i, nmo):
                if abs(h1e_b[i,j]) > TOL:
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
        with open(filename, 'w') as fout:
            if filename == 'fort.55':
                write_head55(fout, nmo, nelec, ms, orbsym)
            else:
                write_head(fout, nmo, nelec, ms, orbsym)
            write_eri_uhf(fout, eri_a, eri_b, eri_ab, nmo, tol=tol, float_format=float_format)
            write_hcore_uhf(fout, h1e_a, h1e_b, nmo, tol=tol, float_format=float_format)
            output_format = float_format + '     0     0     0     0\n'
            fout.write(output_format % nuc)
            
from_integrals_uhf(filename, h1e_a, h1e_b, eri_a, eri_b, eri_ab, nmo, nelec, nuc, 0, 
                   orbsym, tol=1e-18, float_format='% 0.20E')
