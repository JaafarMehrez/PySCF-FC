#!/usr/bin/env python

import pyscf
import numpy as np
from functools import reduce
from pyscf.gto.basis import parse_gaussian
from pyscf.mp.mp2 import get_frozen_mask
from pyscf.tools.fcidump import from_integrals
from pyscf import lib

def freezeCore(oneBody,twoBody,core,valence=None):


    if valence is None: valence = ~core
    else: assert not (core & valence).any(), f"Cannot be both core and valence.\n Core and valence:\n {core & valence}"

    # Effect of core H1 on other terms.
    constant = 2 * np.einsum('ii->', oneBody[core][:,core])
    # Core 2-body contributions to constant
    core_eri = twoBody[np.ix_(core, core, core, core)]
    constant += 2 * np.einsum('iijj->', core_eri)
    constant -= np.einsum('ijij->', core_eri)

    # 2. One-body modifications
    oneBody = oneBody[np.ix_(valence, valence)]
    
    # 2a. (kc|kc) type contraction
    oneBody += 2 * np.einsum('kkab->ab', 
                           twoBody[np.ix_(core, core, valence, valence)])
    
    # 2b. (ak|bk) type contraction
    oneBody -= np.einsum('akbk->ab', 
                       twoBody[np.ix_(valence, core, valence, core)])

    # 3. Update twoBody
    twoBody = twoBody[np.ix_(valence, valence, valence, valence)]
    
    return oneBody, twoBody, constant
    

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
    atom =
            '''
            NE
            ''',
    unit = 'angstrom',
    basis = {
            'NE' : parse_gaussian.load('NE-aVDZ-EMSL.gbs', 'NE')
    },
    charge = 0,
    spin = 0,
    verbose = 9,
    symmetry = True,
    output = name +'.txt',
    symmetry_subgroup = 'D2h',
    max_memory = 4000)

    mymf =  mol.RHF().set(conv_tol=1e-10,max_cycle=999,ddm_tol=1e-14,direct_scf_tol=1e-15, chkfile=name+'.chk',init_guess=    'atom',irrep_nelec={'Ag': 4, 'B3u':2 , 'B2u':2 ,'B1u':2})
    ekrhf = mymf.kernel()

    # FCIDUMP with frozen orbitals
    nocc = mymf.mol.nelectron // 2
    nao, nmo = mymf.mo_coeff.shape
    nvir = nmo - nocc
    nocc_active = 4
    nvir_active = 18
    frozen = list(range(0, nocc-nocc_active)) + list(range(nocc+nvir_active, nmo))
    if len(frozen) == 0:
        frozen = None

    from pyscf.cc import ccsd
    mycc = ccsd.RCCSD(mymf, frozen=frozen)
    
    mo_coeff = mymf.mo_coeff
    assert mo_coeff.dtype == np.double
    orbsym = getattr(mo_coeff, 'orbsym', None)
    nuc = mymf.energy_nuc()
    
    # Get full 1-body Hamiltonian
    h1e = reduce(np.dot, (mo_coeff.T, mymf.get_hcore(), mo_coeff))
    
    #from pyscf.ao2mo import _ao2mo
    nmo = mo_coeff.shape[1]
    mo = np.asarray(mo_coeff, order='F')
    
    #ijslice = (0, nmo, 0, nmo)
    #pq = np.empty((nmo, nmo))
    #eri = ao2mo.incore.full(mymf._eri, mo, compact=True)
    #reshape((nmo,)*4)
    
    # Get full 2-body Hamiltonain
    eri = pyscf.ao2mo.restore(1,pyscf.ao2mo.incore.general(mymf._eri,(mo_coeff,mo_coeff,mo_coeff,mo_coeff),compact=False),nmo)
    
    #for eri1 in range(nmo):
        #pq = _ao2mo.nr_e2(eri, mo, ijslice, aosym='s2', mosym='s1', out=pq)
    #print(pq)
    
    #eri = pyscf.ao2mo.restore(4,pyscf.ao2mo.incore.general(mymf._eri, (mo_coeff,mo_coeff,mo_coeff,mo_coeff),    compact=False),nmo) 
    #eri = pyscf.ao2mo.restore(4,pyscf.ao2mo.incore.general(mymf._eri, (mo_coeff,mo_coeff,mo_coeff,mo_coeff),    compact=False),nmo)
    #np.set_printoptions(threshold=np.inf, edgeitems=np.inf)

    active = get_frozen_mask(mycc)
    frozen_core = np.zeros_like(active, dtype=np.bool_)
    nocc = mol.nelectron//2
    frozen_core[:nocc] = ~active[:nocc]
    h1e, h2e, constant = freezeCore(h1e, eri, frozen_core, active)
    nmo = mycc.nmo

    # I suppose that orbsym should be modified here
    # It is also possible that we need to update h2e so that the frozen orbitals are deleted 
    filename = 'fort.55'
    assert(nmo == h1e.shape[0])
    nelec = (mycc.nocc, mycc.nocc)
    from_integrals(filename, h1e, h2e, nmo, nelec, nuc+constant, 0, orbsym, tol=1e-18, float_format='% 0.20E')

if __name__ == '__main__':

    main()
