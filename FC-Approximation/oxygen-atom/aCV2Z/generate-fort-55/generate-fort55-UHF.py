import pyscf
import numpy as np
import warnings
import numpy as np
import pandas as pd
import pyscf.symm.param as param
import pyscf.lib.logger as logger
from functools import reduce
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask
from pyscf import lib, ao2mo
from pyscf.scf import atom_hf
from pyscf import symm
from pyscf import __config__

np.set_printoptions(threshold=np.inf)

def _normalize_spin_masks(mask, nmo):
    if isinstance(mask, (tuple, list)) and len(mask) == 2:
        ma = np.asarray(mask[0], dtype=bool)
        mb = np.asarray(mask[1], dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.size == 2 * nmo:
            ma = m[:nmo].copy()
            mb = m[nmo:].copy()
        elif m.size == nmo:
            ma = mb = m.copy()
        else:
            raise ValueError(f"Mask length {m.size} not compatible with nmo={nmo}.")
    return ma, mb

def freezeCore(oneBody_a, oneBody_b, twoBody_a, twoBody_b, twoBody_ab, frozen_core, active=None):
    nmo_a = np.asarray(oneBody_a).shape[0]
    nmo_b = np.asarray(oneBody_b).shape[0]
    if nmo_a != nmo_b:
        raise ValueError(f"Different nmo in oneBody_a ({nmo_a}) and oneBody_b ({nmo_b}).")
    nmo = nmo_a
    core_a, core_b = _normalize_spin_masks(frozen_core, nmo)
    if active is None:
        valence_a = ~core_a
        valence_b = ~core_b
    else:
        valence_a, valence_b = _normalize_spin_masks(active, nmo)
    assert core_a.shape == (nmo,) and core_b.shape == (nmo,)
    assert valence_a.shape == (nmo,) and valence_b.shape == (nmo,)
    
    core_idx_a = np.where(core_a)[0]
    core_idx_b = np.where(core_b)[0]
    val_idx_a  = np.where(valence_a)[0]
    val_idx_b  = np.where(valence_b)[0]
    constant   = np.einsum('ii->', oneBody_a[np.ix_(core_idx_a, core_idx_a)])
    constant  += np.einsum('ii->', oneBody_b[np.ix_(core_idx_b, core_idx_b)])
    core_aa    = twoBody_a[np.ix_(core_idx_a, core_idx_a, core_idx_a, core_idx_a)]
    core_bb    = twoBody_b[np.ix_(core_idx_b, core_idx_b, core_idx_b, core_idx_b)]
    core_ab    = twoBody_ab[np.ix_(core_idx_a, core_idx_a, core_idx_b, core_idx_b)]
    constant  += 0.5 * ( np.einsum('iijj->', core_aa) - np.einsum('ijji->', core_aa)
              + np.einsum('iijj->', core_bb) - np.einsum('ijji->', core_bb)
              + 2.0 * np.einsum('iijj->', core_ab) )

    h_active_a  = oneBody_a[np.ix_(val_idx_a, val_idx_a)].copy()
    coul_aa     = twoBody_a[np.ix_(val_idx_a, val_idx_a, core_idx_a, core_idx_a)]
    coul_ab     = twoBody_ab[np.ix_(val_idx_a, val_idx_a, core_idx_b, core_idx_b)]
    h_active_a += np.einsum('pqkk->pq', coul_aa)
    h_active_a += np.einsum('pqkk->pq', coul_ab)
    exch_aa     = twoBody_a[np.ix_(val_idx_a, core_idx_a, core_idx_a, val_idx_a)]
    h_active_a -= np.einsum('pkkq->pq', exch_aa)
    
    h_active_b   = oneBody_b[np.ix_(val_idx_b, val_idx_b)].copy()
    coul_bb      = twoBody_b[np.ix_(val_idx_b, val_idx_b, core_idx_b, core_idx_b)]
    h_active_b  += np.einsum('pqkk->pq', coul_bb)
    twoBody_ab_T = twoBody_ab.transpose(2,3,0,1)
    coul_ba      = twoBody_ab_T[np.ix_(val_idx_b, val_idx_b, core_idx_a, core_idx_a)]
    h_active_b  += np.einsum('pqkk->pq', coul_ba)
    exch_bb      = twoBody_b[np.ix_(val_idx_b, core_idx_b, core_idx_b, val_idx_b)]
    h_active_b  -= np.einsum('pkkq->pq', exch_bb)
    
    twoBody_active_a  = twoBody_a [np.ix_(val_idx_a, val_idx_a, val_idx_a, val_idx_a)].copy()
    twoBody_active_b  = twoBody_b [np.ix_(val_idx_b, val_idx_b, val_idx_b, val_idx_b)].copy()
    twoBody_active_ab = twoBody_ab[np.ix_(val_idx_a, val_idx_a, val_idx_b, val_idx_b)].copy()
    
    return h_active_a, h_active_b, twoBody_active_a, twoBody_active_b, twoBody_active_ab, constant

def main():
    from pyscf import gto, scf, ao2mo
    name = 'out'
    mol = pyscf.M(
        atom='O',
        unit='angstrom',
        basis={'O': parse_gaussian.load('O-aCVDZ-EMSL.gbs', 'O')},
        charge=0,
        spin=2,
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
    
    def order_mos(df, mode='global_energy_asc'):
        
        if mode not in ('global_energy_asc','per_spin_energy_asc','interleave_by_spin','by_irrep_then_energy'):
            raise ValueError("invalid mode")
        
        if mode == 'global_energy_asc':
            ordered = df.sort_values('energy', ascending=True)
            return list(zip(ordered['spin'], ordered['mo_index']))
        
        if mode == 'per_spin_energy_asc':
            ordered = df.sort_values(['spin','energy'], ascending=[True, True])
            return list(zip(ordered['spin'], ordered['mo_index']))
        
        if mode == 'interleave_by_spin':
            alpha_df = df[df['spin']=='alpha'].sort_values('energy', ascending=True)
            beta_df  = df[df['spin']=='beta'].sort_values('energy', ascending=True)
            al = list(zip(alpha_df['spin'], alpha_df['mo_index']))
            bl = list(zip(beta_df['spin'], beta_df['mo_index']))
            out = []
            na = len(al); nb = len(bl)
            n = max(na, nb)
            for i in range(n):
                if i < na: out.append(al[i])
                if i < nb: out.append(bl[i])
            return out
        
        if mode == 'by_irrep_then_energy':
            irr_order = list(dict.fromkeys(df['irrep']))
            out = []
            for ir in irr_order:
                block = df[df['irrep']==ir].sort_values('energy', ascending=True)
                out.extend(list(zip(block['spin'], block['mo_index'])))
            return out
        
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
    
    def print_irrep_ordered_list(df, ordered_pairs, top_n=None):
        
        if top_n is None:
            top_n = len(ordered_pairs)
        for i, (spin, moidx) in enumerate(ordered_pairs[:top_n], start=1):
            row = df[(df['spin']==spin) & (df['mo_index']==moidx)]
            if row.empty:
                print(f"{i:3d}: {spin} mo {moidx}  <missing>")
                continue
            r = row.iloc[0]
            occ_mark = 'occ' if r['is_occupied'] else 'vir'
            print(f"{i:3d}: {spin:5s} {r['irrep']:>4s} idx={r['mo_index']:>3d} {occ_mark:>3s}  E = {r['energy']: .12f}")
            
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
    
    
    from pyscf import cc
    mycc = cc.UCCSD(mf, frozen=1)
    
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
    
    nuc = mf.energy_nuc()
    nmo = orbs_a.shape[0]
    
    h1e_a  = reduce(np.dot, (orbs_a.T, mf.get_hcore(), orbs_a))
    h1e_b  = reduce(np.dot, (orbs_b.T, mf.get_hcore(), orbs_b))
    
    eri_a  = ao2mo.restore(1,ao2mo.incore.general(mf._eri,(orbs_a,orbs_a,orbs_a,orbs_a),compact=False),nmo)
    eri_b  = ao2mo.restore(1,ao2mo.incore.general(mf._eri,(orbs_b,orbs_b,orbs_b,orbs_b),compact=False),nmo)
    eri_ab = ao2mo.restore(1,ao2mo.incore.general(mf._eri,(orbs_a,orbs_a,orbs_b,orbs_b),compact=False),nmo)
    
    active = get_frozen_mask(mycc)
    active_in = active
    
    mo_occ_obj = getattr(getattr(mycc, '_scf', None) or getattr(mycc, 'mf', None) or mycc, 'mo_occ', None)
    if isinstance(mo_occ_obj, np.ndarray) and mo_occ_obj.ndim == 2 and mo_occ_obj.shape[0] == 2:
        mo_occ_a = mo_occ_obj[0]
        mo_occ_b = mo_occ_obj[1]
    elif isinstance(mo_occ_obj, (list, tuple)) and len(mo_occ_obj) == 2:
        mo_occ_a = np.asarray(mo_occ_obj[0])
        mo_occ_b = np.asarray(mo_occ_obj[1])
    else:
        mo_occ_a = mo_occ_b = np.asarray(mo_occ_obj)
        
    nmo = mo_occ_a.size
    
    if isinstance(active_in, (tuple, list)) and len(active_in) == 2:
        act_a = np.asarray(active_in[0], dtype=bool)
        act_b = np.asarray(active_in[1], dtype=bool)
    else:
        act = np.asarray(active_in, dtype=bool)
        if act.size == 2 * nmo:
            act_a = act[:nmo].copy()
            act_b = act[nmo:].copy()
        elif act.size == nmo:
            act_a = act_b = act.copy()
        else:
            raise ValueError(f"Unexpected active length {act.size}; expected {nmo} or {2*nmo}")
        
    shared = np.asarray(act_a, dtype=bool) & np.asarray(act_b, dtype=bool)
    
    act_a[:] = shared
    act_b[:] = shared
    
    active_full = np.empty(2 * nmo, dtype=bool)
    active_full[:nmo] = act_a
    active_full[nmo:] = act_b
    active = active_full
    frozen_core = np.zeros_like(active, dtype=np.bool_)
    nocc_full = mol.nelectron // 2
    frozen_core[:nocc_full] = ~active[:nocc_full]
    
    nocc_a = int(np.count_nonzero(np.asarray(mo_occ_a) > 0.5))
    nocc_b = int(np.count_nonzero(np.asarray(mo_occ_b) > 0.5))
    
    frozen_core_a = np.zeros_like(act_a, dtype=bool)
    frozen_core_b = np.zeros_like(act_b, dtype=bool)
    
    frozen_core_a[:nocc_a] = ~act_a[:nocc_a]
    frozen_core_b[:nocc_b] = ~act_b[:nocc_b]
    
    h1e_a, h1e_b, eri_a, eri_b, eri_ab, constant = freezeCore(h1e_a, h1e_b, eri_a, eri_b, eri_ab, frozen_core=(frozen_core_a, frozen_core_b), active=(act_a, act_b))
    
    nmo_active = h1e_a.shape[0]
    
    if orbsym is None:
        orbsym_active = None
    else:
        orbsym_arr = np.asarray(orbsym, dtype=int)
        valence_idx = np.where(act_a)[0]
        orbsym_active = [int(x) for x in orbsym_arr[valence_idx]]
        
    if orbsym_active is not None:
        if len(orbsym_active) != nmo_active:
            raise RuntimeError(f"Length mismatch: orbsym_active has length {len(orbsym_active)} "
                               f"but number of active orbitals is {nmo_active}.")
            
    filename      = 'fort.55'
    nelec_full    = mol.nelectron
    spin          = mol.spin
    nalpha_full   = (nelec_full + spin) // 2
    nbeta_full    = (nelec_full - spin) // 2
    nocc_full     = mol.nelectron // 2
    nfrozen_core  = int(np.count_nonzero(frozen_core[:nocc_full]))
    nalpha_active = nalpha_full - nfrozen_core
    nbeta_active  = nbeta_full  - nfrozen_core
    nelec_active  = (nalpha_active, nbeta_active)
    
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
        eri_a = np.asarray(eri_a)
        eri_b = np.asarray(eri_b)
        eri_ab = np.asarray(eri_ab)
        npair = nmo * (nmo + 1) // 2
        output_format = float_format + ' %5d %5d %5d %5d\n'
        indx = [i + 1 for i in range(nmo)]
        
        def pair_index(i, j):
            return i * (i + 1) // 2 + j
        
        if eri_a.ndim == 2 and eri_b.ndim == 2 and eri_ab.ndim == 2:
            assert eri_a.shape == (npair, npair) and eri_b.shape == (npair, npair) and eri_ab.shape == (npair, npair)
            kl = 0
            for l in range(nmo):
                for k in range(0, l+1):
                    ij = 0
                    for i in range(0, nmo):
                        for j in range(0, i+1):
                            if i >= k:
                                if abs(eri_a[ij, kl]) > tol:
                                    fout.write(output_format % (eri_a[ij, kl], indx[i], indx[j], indx[k], indx[l]))
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
                                if abs(eri_b[ij, kl]) > tol:
                                    fout.write(output_format % (eri_b[ij, kl], indx[i], indx[j], indx[k], indx[l]))
                            ij += 1
                    kl += 1
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            
            ij = 0
            for j in range(nmo):
                for i in range(0, j+1):
                    kl = 0
                    for k in range(nmo):
                        for l in range(0, k+1):
                            if abs(eri_ab[ij, kl]) > tol:
                                fout.write(output_format % (eri_ab[ij, kl], indx[i], indx[j], indx[k], indx[l]))
                            kl += 1
                    ij += 1
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            return
        
        # CASE B: full 4D arrays
        if eri_a.ndim == 4 and eri_b.ndim == 4 and eri_ab.ndim == 4:
            for i in range(nmo):
                for j in range(0, i + 1):
                    ij_idx = pair_index(i, j)
                    for k in range(nmo):
                        for l in range(0, k + 1):
                            kl_idx = pair_index(k, l)
                            if ij_idx >= kl_idx:
                                val = eri_a[i, j, k, l]
                                if abs(val) > tol:
                                    fout.write(output_format % (val, indx[i], indx[j], indx[k], indx[l]))
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            
            for i in range(nmo):
                for j in range(0, i + 1):
                    ij_idx = pair_index(i, j)
                    for k in range(nmo):
                        for l in range(0, k + 1):
                            kl_idx = pair_index(k, l)
                            if ij_idx >= kl_idx:
                                val = eri_b[i, j, k, l]
                                if abs(val) > tol:
                                    fout.write(output_format % (val, indx[i], indx[j], indx[k], indx[l]))
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            
            for i in range(nmo):
                for j in range(0, i + 1):
                    for k in range(nmo):
                        for l in range(0, k + 1):
                            val = eri_ab[i, j, k, l]
                            if abs(val) > tol:
                                fout.write(output_format % (val, indx[i], indx[j], indx[k], indx[l]))
            fout.write(' 0.00000000000000000000E+00' + '     0     0     0     0\n')
            return
        
        raise RuntimeError(f"Unsupported ERI shapes: eri_a {eri_a.shape}, eri_b {eri_b.shape}, eri_ab {eri_ab.shape}")
    
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
            
    from_integrals_uhf(filename, h1e_a, h1e_b, eri_a, eri_b, eri_ab, nmo_active, nelec_active, nuc + constant, 0, orbsym_active, tol=1e-18, float_format='% 0.20E')
    
if __name__ == '__main__':
    main()