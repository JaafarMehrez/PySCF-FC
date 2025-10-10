#!/usr/bin/env python

from functools import reduce
import numpy as np
from pyscf.mp.mp2 import get_frozen_mask
from pyscf.tools.fcidump import TOL, DEFAULT_FLOAT_FORMAT, from_integrals
from pyscf import lib


def freezeCore(oneBody,twoBody,core,valence=None):
    ''' Freeze Hamiltonian orbitals.

    ```
    orbs = numpy.array(10)
    oneBody,twoBody,constant = freezeCore(oneBody,twoBody,core=(orbs<3),valence=(orbs>3)&(orb<7))
    # orbs 7,8,9 are discarded virtual orbitals.
    Only implemented for RHF oneBody and twoBody (UHF would break spin-symmetry).
    ```

    Args:
        oneBody (array): one-body part of the Hamiltonian.
        twoBody (array): two-body part of the Hamiltonian.
        core (bool array): True only for each orbital to consider occupied, but frozen.
        valence (bool array): True only for each orbital to consider active in the valence.
           If None, it will be ~core.
           Orbitals not core or valence are truncated from the simulation.
    Returns:
        Pointers back to oneBody, twoBody, and the new value of the constant.
        Note: oneBody and twoBody are modified in-place due to their size.
    '''

    if valence is None: valence = ~core
    else: assert not (core & valence).any(), f"Cannot be both core and valence.\n Core and valence:\n {core & valence}"

    # Unpack index.
    #twoBody = twoBody.reshape(twoBody.shape[0],oneBody.shape[0],oneBody.shape[1])

    # Effect of core H1 on other terms.
    constant = 2.*oneBody[core][:,core].trace()

    # Remove frozen space from one-body
    oneBody = oneBody[valence][:,valence]

    # Effect of core H2 on other terms.
    constant += 2*np.einsum('gii,gjj->',
            twoBody[:,core][:,:,core],
            twoBody[:,core][:,:,core].conj()
        )
    constant -= np.einsum('gij,gij->',
            twoBody[:,core][:,:,core],
            twoBody[:,core][:,:,core].conj()
        )
    oneBody += np.einsum('gkk,gji->ij',
            twoBody[:,core][:,:,core],
            twoBody[:,valence][:,:,valence].conj()
        )
    oneBody += -0.5*np.einsum('gik,gjk->ij',
            twoBody[:,valence][:,:,core],
            twoBody[:,valence][:,:,core].conj()
        )
    oneBody += np.einsum('gkk,gij->ij',
            twoBody[:,core][:,:,core].conj(),
            twoBody[:,valence][:,:,valence]
        )
    oneBody += -0.5*np.einsum('gki,gkj->ij',
            twoBody[:,core][:,:,valence].conj(),
            twoBody[:,core][:,:,valence]
        )

    # Remove frozen space from two-body.
    twoBody = twoBody[:,valence][:,:,valence]
    #twoBody = twoBody.reshape(twoBody.shape[0],twoBody.shape[1]*twoBody.shape[2])

    return oneBody,twoBody,constant


def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, np.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(f" {nmo:4d} {nelec:2d}\n")
    if orbsym is not None and len(orbsym) > 0:
        fout.write(f"{' '.join([str(x) for x in orbsym])}\n")
    else:
        fout.write(f"{' 1' * nmo}\n")
    fout.write(' 150000\n')


from pyscf.tools import fcidump
fcidump.write_head = write_head

def main():
    from pyscf.pbc import gto, scf

    cell = gto.Cell()
    cell.unit = 'A'
    # strucure is obtained from CCDC, CSD entry: PENCEN10 
    cell.atom = ''' 
C          4.71141        0.64706        0.10780
C          4.95445        0.67849        2.55640
C          5.44163        0.34038        1.27820
C          1.74491        3.15947        0.14014
C          3.77775        4.16326        1.14884
C          4.48690        4.49222        2.30846
C          3.98974        4.20504        3.57587
C          4.72860        4.57069        4.76167
C          4.24559        4.30345        5.98597
C          2.97396        3.63518        6.13997
C          2.25175        3.28304        5.04195
C          2.71431        3.55727        3.70523
C          2.00347        3.22256        2.58719
C          2.48829        3.49963        1.29360
H          3.89017        1.07641        0.18788
H          4.36393        1.13522        5.10817
H          4.13164        1.10613        2.63031
H          0.92032        2.74087        0.24332
H          5.31239        4.91309        2.22992
H          5.54949        4.99874        4.67543
H          4.73274        4.55006        6.73903
H          2.64928        3.44500        6.99159
H          1.43501        2.85266        5.15745
H          1.17959        2.80201        2.68268
C          5.18275        0.32614       -1.16732
C          5.66057        0.39218        3.69137
C          6.73325       -0.32614        1.16732
C          2.20677        3.43270       -1.14884
C          4.23960        4.43649       -0.14014
C          4.48126        0.60985       -2.34388
C          6.47437       -0.34038       -1.27820
C          5.18905        0.71718        5.01269
C          6.93099       -0.27069        3.59435
C          7.20459       -0.64706       -0.10780
C          7.43474       -0.60985        2.34388
C          1.49761        3.10373       -2.30846
C          3.49622        4.09632       -1.29360
H          5.06420        4.85509       -0.24332
H          3.65743        1.03716       -2.28690
C          4.98501        0.27069       -3.59435
C          6.96155       -0.67849       -2.55640
C          5.91601        0.43083        6.12765
C          7.68053       -0.55492        4.79709
H          8.02583       -1.07641       -0.18788
H          8.25857       -1.03716        2.28690
H          0.67212        2.68286       -2.22992
C          1.99478        3.39092       -3.57587
C          3.98105        4.37340       -2.58719
C          4.23547        0.55492       -4.79709
C          6.25543       -0.39218       -3.69137
H          7.78436       -1.10613       -2.63031
H          5.58497        0.65297        6.96849
C          7.20048       -0.21913        6.00599
H          8.50690       -0.97582        4.73241
C          1.25591        3.02526       -4.76167
C          3.27021        4.03868       -3.70523
H          4.80492        4.79394       -2.68268
H          3.40910        0.97582       -4.73241
C          4.71552        0.21913       -6.00599
C          6.72695       -0.71718       -5.01269
H          7.69799       -0.40635        6.76829
H          0.43503        2.59722       -4.67543
C          1.73892        3.29250       -5.98597
C          3.73276        4.31292       -5.04195
H          4.21801        0.40635       -6.76829
C          6.00000       -0.43083       -6.12765
H          7.55207       -1.13522       -5.10817
H          1.25177        3.04589       -6.73903
C          3.01056        3.96078       -6.13997
H          4.54951        4.74329       -5.15745
H          6.33103       -0.65297       -6.96849
H          3.33524        4.15095       -6.99159
    '''
    cell.a = [[ 5.958     ,  0.        ,  0.        ],
       [ 0.02651499,  7.59595372,  0.        ],
       [ 0.93662692,  2.37133022, 15.39997192]] 
    cell.basis = 'ccecp-cc-pvtz'
    cell.pseudo = 'ccecp'
    cell.verbose = 7
    cell.max_memory = 300000
    cell.build()

    mymf = scf.RHF(cell, exxdiv=None)
    auxbasis = 'cc-pvtz-ri'
    mymf = mymf.density_fit(auxbasis=auxbasis)
    ekrhf = mymf.kernel()

    # FCIDUMP with frozen orbitals
    nocc = mymf.mol.nelectron // 2
    nao, nmo = mymf.mo_coeff.shape
    nvir = nmo - nocc
    nocc_active = 32
    nvir_active = 32
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
    # Get full Lpq (in MO basis)
    from pyscf.ao2mo import _ao2mo
    with_df = mymf.with_df
    nmo = mo_coeff.shape[1]
    naux = with_df.get_naoaux()
    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = np.empty((naux, nmo, nmo))
    Lpq_blk = None
    for eri1 in with_df.loop():
        Lpq_blk = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq_blk)
        p0, p1 = p1, p1 + Lpq_blk.shape[0]
        Lpq[p0:p1] = Lpq_blk.reshape(p1-p0,nmo,nmo)
    Lpq_blk = None

    active = get_frozen_mask(mycc)
    frozen_core = np.zeros_like(active, dtype=np.bool)
    nocc = cell.nelectron//2
    frozen_core[:nocc] = ~active[:nocc]
    h1e, Lpq, constant = freezeCore(h1e, Lpq, frozen_core, active)
    nmo = mycc.nmo

    # compact h2e
    # npq_pair = nmo * (nmo+1) // 2
    # Lpq = lib.pack_tril(Lpq)
    # print(f"new Lpq shape: {Lpq.shape}")
    # h2e = np.zeros((npq_pair, npq_pair))
    # lib.dot(Lpq.T, Lpq, 1, h2e, 1)

    # full h2e
    h2e = np.zeros((nmo, nmo, nmo, nmo))
    h2e = np.einsum('Lpq,Lrs->pqrs', Lpq, Lpq)
    Lpq = None

    # Save h1e, h2e, and nuc
    filename = 'fort.55'
    assert(nmo == h1e.shape[0])
    nelec = (mycc.nocc, mycc.nocc)
    from_integrals(filename, h1e, h2e, nmo, nelec, nuc+constant, 0, orbsym, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT)

if __name__ == '__main__':
    main()

