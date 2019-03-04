from __future__ import print_function
import os
import gc
import sys
import time
import signal
import traceback
from distutils.version import LooseVersion
from unittest import SkipTest

import numpy as np
from ase.utils import devnull

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters, tf_parameters
from gpaw.utilities import compiled_with_sl, compiled_with_libvdwxc
from gpaw.utilities.elpa import LibElpa
from gpaw import setup_paths
from gpaw import mpi
import gpaw


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y).any() or (np.abs(x - y) > tolerance).any():
        msg = '{} {} != {} (error: |{}| > {:.9g})'.format(msg, x, y, x - y,
                                                          tolerance)
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)


def print_reference(data_i, name='ref_i', fmt='%.12le'):
    if mpi.world.rank == 0:
        print('%s = [' % name, end='')
        for i, val in enumerate(data_i):
            if i > 0:
                print('', end='\n')
                print(' ' * (len(name) + 4), end='')
            print(fmt % val, end='')
            print(',', end='')
        print('\b]')


def findpeak(x, y):
    dx = x[1] - x[0]
    i = y.argmax()
    a, b, c = np.polyfit([-1, 0, 1], y[i - 1:i + 2], 2)
    assert a < 0
    x = -0.5 * b / a
    return dx * (i + x), a * x**2 + b * x + c


def gen(symbol, exx=False, name=None, yukawa_gamma=None, **kwargs):
    setup = None
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        if 'orbital_free' in kwargs:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          use_restart_file=False,
                          **tf_parameters.get(symbol, {'rcut': 0.9}))
        else:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          use_restart_file=False,
                          **parameters[symbol])
    setup = mpi.broadcast(setup, 0)
    if setup_paths[0] != '.':
        setup_paths.insert(0, '.')
    return setup


tests = [
    'linalg/gemm_complex.py',
    'ase_features/ase3k_version.py',
    'kpt.py',
    'mpicomm.py',
    'pathological/numpy_core_multiarray_dot.py',
    'eigen/cg2.py',
    'fd_ops/laplace.py',
    'linalg/lapack.py',
    'linalg/eigh.py',
    'parallel/submatrix_redist.py',
    'lfc/second_derivative.py',
    'parallel/parallel_eigh.py',
    'lfc/gp2.py',
    'linalg/blas.py',
    'Gauss.py',
    'symmetry/check.py',
    'fd_ops/nabla.py',
    'linalg/dot.py',
    'linalg/mmm.py',
    'xc/pbe_pw91.py',
    'fd_ops/gradient.py',
    'maths/erf.py',
    'initial_occs.py',
    'lfc/lf.py',
    'maths/fsbt.py',
    'parallel/compare.py',
    'vdw/libvdwxc_functionals.py',
    'radial/integral4.py',
    'linalg/zher.py',
    'fd_ops/gd.py',
    'pw/interpol.py',
    'poisson/screened_poisson.py',
    'electronphonon.py',
    'xc/xc.py',
    'xc/XC2.py',
    'radial/yukawa_radial.py',
    'vdw/potential.py',
    'radial/lebedev.py',
    'occupations.py',
    'lfc/derivatives.py',
    # 'parallel/realspace_blacs.py',
    'pw/reallfc.py',
    'parallel/pblas.py',
    'fd_ops/non_periodic.py',
    'spectrum.py',
    'pw/lfc.py',
    'gauss_func.py',
    'multipoletest.py',
    'cluster.py',
    'poisson/poisson.py',
    'poisson/fastpoisson.py',
    'poisson/poisson_asym.py',
    'parallel/arraydict_redist.py',
    'parallel/scalapack.py',
    'gauss_wave.py',
    'fd_ops/transformations.py',
    'parallel/blacsdist.py',
    'pbc.py',
    'atoms_too_close.py',
    'ext_potential/harmonic.py',
    'atoms_mismatch.py',
    'setup_basis_spec.py',
    'pw/direct.py',
    'libelpa.py',
    'vdw/libvdwxc_spin.py',                 # ~1s
    'timing.py',                            # ~1s
    'parallel/ut_parallel.py',              # ~1s
    'lcao/density.py',                      # ~1s
    'pw/stresstest.py',                     # ~1s
    'pw/fftmixer.py',                       # ~1s
    'lcao/fftmixer.py',                     # ~1s
    'symmetry/usesymm.py',                  # ~1s
    'coulomb.py',                           # ~1s
    'xc/xcatom.py',                         # ~1s
    'force_as_stop.py',                     # ~1s
    'vdwradii.py',                          # ~1s
    'ase_features/ase3k.py',                # ~1s
    'pathological/numpy_zdotc_graphite.py',  # ~1s
    'utilities/eed.py',                     # ~1s
    'lcao/dos.py',                          # ~1s
    'solvation/nan_radius.py',              # ~1s
    'solvation/pbc_pos_repeat.py',          # ~1s
    'lcao/generate_ngto.py',                # ~1s
    'reuse_wfs_celldisp.py',                # ~1s
    'linalg/gemv.py',                       # ~2s
    'fileio/idiotproof_setup.py',           # ~2s
    'radial/ylexpand.py',                   # ~2s
    'eigen/keep_htpsit.py',                 # ~2s
    'xc/gga_atom.py',                       # ~2s
    'generic/hydrogen.py',                  # ~2s
    'aeatom.py',                            # ~2s
    'ase_features/plt.py',                  # ~2s
    'ds_beta.py',                           # ~2s
    'multipoleH2O.py',                      # ~2s
    'spinorbit_Kr.py',                      # ~2s
    'stdout.py',                            # ~2s
    'lcao/largecellforce.py',               # ~2s
    'parallel/scalapack_diag_simple.py',    # ~2s
    'fixdensity.py',                        # ~2s
    'pseudopotential/ah.py',                # ~2s
    'lcao/gllb_si.py',                      # ~2s
    'fileio/wfs_io.py',                     # ~3s
    'lrtddft/2.py',                         # ~3s
    'gllbghost.py',                         # ~3s
    'fileio/file_reference.py',             # ~3s
    'fileio/restart.py',                    # ~3s
    'broydenmixer.py',                      # ~3s
    'pw/fulldiagk.py',                      # ~3s
    'ext_potential/external.py',            # ~3s
    'ext_potential/external_pw.py',         # ~3s
    'lcao/atomic_corrections.py',           # ~3s
    'vdw/libvdwxc_h2.py',                   # ~3s
    'generic/mixer.py',                     # ~3s
    'parallel/lcao_projections.py',         # ~3s
    'lcao/h2o.py',                          # ~3s
    'corehole/h2o.py',                      # ~3s
    'fileio/wfs_auto.py',                   # ~3s
    'pw/fulldiag.py',                       # ~3s
    'symmetry/symmetry_ft.py',              # ~3s
    'response/aluminum_EELS_RPA.py',        # ~3s
    'poisson/poisson_extended.py',          # ~3s
    'solvation/vacuum.py',                  # ~3s
    'vdw/libvdwxc_mbeef.py',                # ~3s
    'response/graphene_refined_response.py',  # ~3s
    'response/pdens_tool.py',               # ~4s
    'pw/par_strategies.py',                  # ~4s
    'pseudopotential/sg15_hydrogen.py',     # ~4s
    'generic/move_across_cell.py',          # ~4s
    'parallel/augment_grid.py',             # ~4s
    'pw/augment_grids.py',
    'utilities/ewald.py',                   # ~4s
    'symmetry/symmetry.py',                 # ~4s
    'xc/revPBE.py',                         # ~4s
    'xc/nonselfconsistentLDA.py',           # ~4s
    'response/aluminum_EELS_ALDA.py',       # ~4s
    'spin/spin_contamination.py',           # ~4s
    'inducedfield_lrtddft.py',              # ~4s
    'generic/H_force.py',                   # ~4s
    'symmetry/usesymm2.py',                 # ~4s
    'mgga/mgga_restart.py',                 # ~4s
    'fixocc.py',                            # ~4s
    'spin/spinFe3plus.py',                  # ~4s
    'fermisplit.py',                        # ~4s
    'generic/Cl_minus.py',                  # ~4s
    'lrtddft/pes.py',                       # ~4s
    'generic/proton.py',                    # ~4s
    'parallel/lcao_elpa_kpts.py',           # ~4s
    'parallel/lcao_elpa.py',                # ~5s
    'corehole/h2o_recursion.py',            # ~5s
    'xc/nonselfconsistent.py',              # ~5s
    'spin/spinpol.py',                      # ~5s
    'eigen/cg.py',                          # ~5s
    'parallel/kptpar.py',                   # ~5s
    'utilities/elf.py',                     # ~5s
    'eigen/blocked_rmm_diis.py',            # ~5s
    'pw/slab.py',                           # ~5s
    'generic/si.py',                        # ~5s
    'lcao/bsse.py',                         # ~5s
    'parallel/lcao_hamiltonian.py',         # ~5s
    'xc/degeneracy.py',                     # ~5s
    'fileio/refine.py',                     # ~5s
    'solvation/pbc.py',                     # ~5s
    'generic/asym_box.py',                  # ~5s
    'overlap.py',                           # ~5s
    'linalg/gemm.py',                       # ~6s
    'generic/al_chain.py',                  # ~6s
    'fileio/parallel.py',                   # ~6s
    'fixmom.py',                            # ~6s
    'exx/unocc.py',                         # ~6s
    'eigen/davidson.py',                    # ~6s
    'vdw/H_Hirshfeld.py',                   # ~6s
    'parallel/redistribute_grid.py',        # ~7s
    'aedensity.py',                         # ~7s
    'pw/h.py',                              # ~7s
    'lrtddft/apmb.py',                      # ~7s
    'pseudopotential/hgh_h2o.py',           # ~7s
    'poisson/poisson_restart.py',           # ~7s
    'poisson/poisson_extravacuum.py',       # ~7s
    'fdtd/ed_wrapper.py',                   # ~7s
    'fdtd/ed_shapes.py',                    # ~7s
    'fdtd/ed.py',                           # ~12s
    'fdtd/ed_inducedfield.py',              # ~16s
    'inducedfield_td.py',                   # ~9s
    'pw/bulk.py',                           # ~7s
    'gllb/ne.py',                           # ~7s
    'lcao/force.py',                        # ~7s
    'xc/pplda.py',                          # ~7s
    'response/test_unit_sphere_area.py',    # ~7s
    'rsf_yukawa/lrtddft_short.py',          # ~7s
    'fileio/restart_density.py',            # ~8s
    'rpa/rpa_energy_Ni.py',                 # ~8s
    'tddft/be_nltd_ip.py',                  # ~8s
    'ibzqpt.py',                            # ~8s
    'noncollinear/o2.py',
    'generic/si_primitive.py',              # ~9s
    'tddft/ehrenfest_nacl.py',              # ~9s
    'lcao/fd2lcao_restart.py',              # ~9s
    'ext_potential/constant_e_field.py',    # ~9s
    'complex.py',                           # ~9s
    'vdw/quick.py',                         # ~9s
    'pathological/lcao_spos_derivative.py',  # ~9s
    'lrtddft2/H2O-lcao.py',                 # ~10s
    'lrtddft2/Al2.py',                      # ~10s
    'lcaotddft/simple.py',                  # ~10s
    'lcaotddft/restart.py',                 # ~10s
    'lcaotddft/fxc_is_xc.py',               # ~10s
    'lcaotddft/fxc_rpa.py',                 # ~10s
    'lcaotddft/fxc_vs_linearize.py',        # ~10s
    'lcaotddft/replay.py',                  # ~10s
    'lcaotddft/ksdecomp.py',                # ~10s
    'tddft/fxc_linearize.py',               # ~10s
    'timelimit.py',                         # ~10s
    'ralda/ralda_energy_N2.py',             # ~10s
    'parallel/lcao_complicated.py',         # ~10s
    'generic/bulk.py',                      # ~10s
    'sic/scfsic_h2.py',                     # ~10s
    'kpt_refine.py',                        # ~10s
    'lcao/bulk.py',                         # ~11s
    'reuse_wfs.py',                         # ~11s
    'generic/2Al.py',                       # ~11s
    'lrtddft/kssingles_Be.py',              # ~11s
    'generic/relax.py',                     # ~11s
    'solvation/adm12.py',                   # ~11s
    'rsf_yukawa/rsf_ivo_sing_mg.py',        # ~11s
    'solvation/lrtddft.py',                 # ~12s
    'dscf/dscf_lcao.py',                    # ~12s
    'generic/8Si.py',                       # ~12s
    'utilities/partitioning.py',            # ~12s
    'xc/lxc_xcatom.py',                     # ~12s
    'solvation/sfgcm06.py',                 # ~12s
    'solvation/sss09.py',                   # ~12s
    'tddft/restart.py',                     # ~13s
    'gllb/atomic.py',                       # ~13s
    'generic/guc_force.py',                 # ~13s
    'ralda/ralda_energy_Ni.py',             # ~13s
    'utilities/simple_stm.py',              # ~13s
    'ofdft/ofdft_pbc.py',                   # ~13s
    'gllb/restart_band_structure.py',       # ~14s
    'exx/exx.py',                           # ~14s
    'response/iron_sf_ALDA_gridrep.py',     # ~15s
    'response/iron_sf_gssALDA_gridrep.py',  # ~15s
    'Hubbard_U.py',                         # ~15s
    'rpa/rpa_energy_Si.py',                 # ~15s
    'dipole.py',                            # ~15s
    'generic/IP_oxygen.py',                 # ~15s
    'rpa/rpa_energy_Na.py',                 # ~15s
    'parallel/fd_parallel.py',              # ~15s
    'solvation/poisson.py',                 # ~15s
    'solvation/water_water.py',             # ~15s
    'xc/pygga.py',                          # ~15s
    'pseudopotential/atompaw.py',           # ~15s
    'parallel/lcao_parallel.py',            # ~16s
    'xc/atomize.py',                        # ~16s
    'lrtddft/excited_state.py',             # ~16s
    'gllb/ne_disc.py',                      # ~16s
    'ofdft/ofdft.py',                       # ~17s
    'response/bse_silicon.py',              # ~18s
    'tpss.py',                              # ~18s
    'tddft/td_na2.py',                      # ~18s
    'exx/coarse.py',                        # ~18s
    'corehole/si.py',                       # ~18s
    'mgga/mgga_sc.py',                      # ~19s
    'Hubbard_U_Zn.py',                      # ~20s
    'lrtddft/1.py',                         # ~20s
    'gllb/spin.py',                         # ~21s
    'parallel/fd_parallel_kpt.py',          # ~21s
    'generic/Cu.py',                        # ~21s
    'vdw/ts09.py',                          # ~21s
    'response/na_plasmon.py',               # ~22s
    'fermilevel.py',                        # ~23s
    'ralda/ralda_energy_H2.py',             # ~23s
    'symmetry/kpoint_mapping.py',           # ~23s
    'response/diamond_absorption.py',       # ~24s
    'ralda/ralda_energy_Si.py',             # ~24s
    'jellium.py',                           # ~24s
    'utilities/ldos.py',                    # ~25s
    'solvation/swap_atoms.py',              # ~25s
    'xc/revPBE_Li.py',                      # ~26s
    'ofdft/ofdft_scale.py',                 # ~26s
    'parallel/lcao_parallel_kpt.py',        # ~29s
    'lrtddft/placzek_profeta_albrecht.py',  # ~29s
    'corehole/h2o_dks.py',                  # ~30s
    'lcaotddft/parallel_options.py',        # ~30s
    'lcaotddft/lcaotddft_vs_lrtddft2.py',   # ~30s
    'lcaotddft/lcaotddft_vs_lrtddft2_rpa.py',  # ~30s
    'mgga/nsc_MGGA.py',                     # ~32s
    'solvation/spinpol.py',                 # ~32s
    'gllb/diamond.py',                      # ~33s
    'vdw/quick_spin.py',                    # ~37s
    'pw/expert_diag.py',                    # ~37s
    'pathological/LDA_unstable.py',         # ~42s
    'response/bse_aluminum.py',             # ~42s
    'response/au02_absorption.py',          # ~44s
    'xc/tb09.py',
    'ext_potential/point_charge.py',
    'ase_features/wannierk.py',             # ~45s
    'ut_tddft.py',                          # ~49s
    'response/pair.py',                     # ~50s
    'rpa/rpa_energy_N2.py',                 # ~52s
    'vdw/ar2.py',                           # ~53s
    'rsf_yukawa/rsf_general.py',            # ~54s
    'solvation/forces_symmetry.py',         # ~56s
    'parallel/diamond_gllb.py',             # ~59s
    'xc/qna_force.py',
    'xc/qna_stress.py',
    'xc/qna_band.py',
    'xc/qna_spinpol.py',
    'beef.py',
    'response/chi0.py',                     # ~71s
    'lrtddft/3.py',                         # ~75s
    'response/chi0_intraband_test.py',      # ~76s
    'pathological/nonlocalset.py',          # ~82s
    'response/gw0_hBN.py',                  # ~82s
    'xc/lb94.py',                           # ~84s
    'response/iron_sf_ALDA.py',             # ~86s
    'exx/exx_scf.py',                       # ~91s
    'pw/si_stress.py',                      # ~100s
    'response/gw_hBN_extrapolate.py',       # ~109s
    'exx/AA_enthalpy.py',                   # ~119s
    'response/na_plasmons.py',
    'response/na_plasmons_tetrahedron.py',  # ~120s
    'solvation/forces.py',                  # ~140s
    'response/gw_MoS2_cut.py',
    'response/gwsi.py',                     # ~147s
    'response/graphene.py',                 # ~160s
    'response/symmetry.py',                 # ~300s
    'pw/moleculecg.py',                     # duration unknown
    'potential.py',                         # duration unknown
    'lcao/pair_and_coulomb.py',             # duration unknown
    'ase_features/asewannier.py',           # duration unknown
    'pw/davidson_pw.py',                    # duration unknown
    'ase_features/neb.py',                  # duration unknown
    'utilities/wannier_ethylene.py',        # duration unknown
    'muffintinpot.py',                      # duration unknown
    'sic/nscfsic.py',                       # duration unknown
    'coreeig.py',                           # duration unknown
    'response/bse_MoS2_cut.py',             # duration unknown
    'parallel/scalapack_mpirecv_crash.py',  # duration unknown
    'cdft.py']                              # 40s

# 'symmetry/fractional_translations.py',
# 'response/graphene_EELS.py', disabled while work is in progress

# 'symmetry/fractional_translations_med.py',
# 'symmetry/fractional_translations_big.py',

# 'linalg/eigh_perf.py', # Requires LAPACK 3.2.1 or later
# XXX https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
# 'parallel/scalapack_pdlasrt_hang.py',
# 'dscf/dscf_forces.py',
# 'ext_potential/stark_shift.py',


exclude = []

if mpi.size > 1:
    exclude += ['ase_features/asewannier.py',
                'coreeig.py',
                'ext_potential/stark_shift.py',
                'spinorbit_Kr.py',
                'fd_ops/laplace.py',
                'potential.py',
                'lcao/generate_ngto.py',
                'lcao/pair_and_coulomb.py',
                'muffintinpot.py',
                'pw/moleculecg.py',
                'pw/davidson_pw.py',
                'sic/nscfsic.py',
                # scipy.weave fails often in parallel due to
                # ~/.python*_compiled
                # https://github.com/scipy/scipy/issues/1895
                'scipy_test.py',
                'utilities/wannier_ethylene.py']

if mpi.size > 2:
    exclude += ['ase_features/neb.py',
                'poisson/poisson_restart.py',
                'response/pair.py',
                'electronphonon.py']

if mpi.size > 4:
    exclude += ['poisson/poisson_extravacuum.py']

if mpi.size < 4:
    exclude += ['parallel/fd_parallel.py',
                'parallel/lcao_parallel.py',
                'parallel/pblas.py',
                'parallel/scalapack.py',
                'parallel/scalapack_diag_simple.py',
                'parallel/realspace_blacs.py',
                'exx/AA_enthalpy.py',
                'exx/exx_scf.py',
                'response/bse_aluminum.py',
                'response/iron_sf_ALDA.py',
                'response/bse_MoS2_cut.py',
                'fileio/parallel.py',
                'parallel/diamond_gllb.py',
                'parallel/lcao_parallel_kpt.py',
                'parallel/fd_parallel_kpt.py',
                'response/na_plasmons.py',
                'response/na_plasmons_tetrahedron.py']


if mpi.size != 4:
    exclude += ['parallel/scalapack_mpirecv_crash.py',
                'parallel/scalapack_pdlasrt_hang.py',
                'response/bse_silicon.py']

if mpi.size == 1 or not compiled_with_sl():
    exclude += ['parallel/submatrix_redist.py',
                'lcaotddft/parallel_options.py']

if mpi.size != 1 and not compiled_with_sl():
    exclude += ['ralda/ralda_energy_H2.py',
                'ralda/ralda_energy_N2.py',
                'ralda/ralda_energy_Ni.py',
                'ralda/ralda_energy_Si.py',
                'response/bse_silicon.py',
                'response/bse_MoS2_cut.py',
                'response/gwsi.py',
                'response/gw_MoS2_cut.py',
                'rpa/rpa_energy_N2.py',
                'pw/expert_diag.py',
                'pw/fulldiag.py',
                'pw/fulldiagk.py',
                'response/gw_hBN_extrapolate.py',
                'response/gw0_hBN.py',
                'response/au02_absorption.py']

if not compiled_with_sl():
    exclude += ['lcao/atomic_corrections.py',
                'response/na_plasmons.py',
                'response/na_plasmons_tetrahedron.py']

if not compiled_with_libvdwxc():
    exclude.append('vdw/libvdwxc_functionals.py')
    exclude.append('vdw/libvdwxc_h2.py')
    exclude.append('vdw/libvdwxc_mbeef.py')
    exclude.append('vdw/libvdwxc_spin.py')

if not LibElpa.have_elpa():
    exclude += ['libelpa.py',
                'parallel/lcao_elpa_kpts.py',
                'parallel/lcao_elpa.py']


if LooseVersion(np.__version__) < '1.6.0':
    exclude.append('response/chi0.py')


def get_test_path(test):
    return os.path.join(gpaw.__path__[0], 'test', test)


exclude = set(exclude)


def check_file_lists():
    for test in tests + list(exclude):
        assert os.path.exists(get_test_path(test)), \
            ('No such file: {}.  Test list or test exclusion list mentions '
             'files that do not exist.'.format(test))


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=False):
        if mpi.size > 1:
            assert jobs == 1
        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.failed = []
        self.skipped = []
        self.garbage = []
        if mpi.rank == 0:
            self.log = stream
        else:
            self.log = devnull
        self.n = max([len(test) for test in tests])
        self.setup_paths = setup_paths[:]

        # Check *all* the files, not just the ones we are supposed to be
        # running right now:
        check_file_lists()

    def run(self):
        self.log.write('=' * 77 + '\n')
        if not self.show_output:
            sys.stdout = devnull
        ntests = len(self.tests)
        t0 = time.time()
        if self.jobs == 1:
            self.run_single()
        else:
            # Run several processes using fork:
            self.run_forked()

        sys.stdout = sys.__stdout__
        self.log.write('=' * 77 + '\n')
        self.log.write('Ran %d tests out of %d in %.1f seconds\n' %
                       (ntests - len(self.tests) - len(self.skipped),
                        ntests, time.time() - t0))
        self.log.write('Tests skipped: %d\n' % len(self.skipped))
        if self.failed:
            print('Tests failed:', len(self.failed), file=self.log)
        else:
            self.log.write('All tests passed!\n')
        self.log.write('=' * 77 + '\n')
        return self.failed

    def run_single(self):
        while self.tests:
            test = self.tests.pop(0)
            try:
                self.run_one(test)
            except KeyboardInterrupt:
                self.tests.append(test)
                break

    def run_forked(self):
        j = 0
        pids = {}
        while self.tests or j > 0:
            if self.tests and j < self.jobs:
                test = self.tests.pop(0)
                pid = os.fork()
                if pid == 0:
                    exitcode = self.run_one(test)
                    os._exit(exitcode)
                else:
                    j += 1
                    pids[pid] = test
            else:
                try:
                    while True:
                        pid, exitcode = os.wait()
                        if pid in pids:
                            break
                except KeyboardInterrupt:
                    for pid, test in pids.items():
                        os.kill(pid, signal.SIGHUP)
                        self.write_result(test, 'STOPPED', time.time())
                        self.tests.append(test)
                    break
                if exitcode == 512:
                    self.failed.append(pids[pid])
                elif exitcode == 256:
                    self.skipped.append(pids[pid])
                del pids[pid]
                j -= 1

    def run_one(self, test):
        exitcode_ok = 0
        exitcode_skip = 1
        exitcode_fail = 2

        if self.jobs == 1:
            self.log.write('%*s' % (-self.n, test))
            self.log.flush()

        t0 = time.time()
        filename = get_test_path(test)

        tb = ''
        skip = False

        if test in exclude:
            self.register_skipped(test, t0)
            return exitcode_skip

        assert test.endswith('.py')
        dirname = test[:-3]
        if os.path.isabs(dirname):
            mydir = os.path.split(__file__)[0]
            dirname = os.path.relpath(dirname, mydir)

        # We don't want files anywhere outside the tempdir.
        assert not dirname.startswith('../')  # test file outside sourcedir

        if mpi.rank == 0:
            os.makedirs(dirname)
        mpi.world.barrier()
        cwd = os.getcwd()
        os.chdir(dirname)

        try:
            setup_paths[:] = self.setup_paths
            loc = {}
            with open(filename) as fd:
                exec(compile(fd.read(), filename, 'exec'), loc)
            loc.clear()
            del loc
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except SkipTest:
            skip = True
        except ImportError as ex:
            if sys.version_info[0] >= 3:
                module = ex.name
            else:
                module = ex.args[0].split()[-1].split('.')[0]
            if module == 'scipy':
                skip = True
            else:
                tb = traceback.format_exc()
        except AttributeError as ex:
            if (ex.args[0] ==
                "'module' object has no attribute 'new_blacs_context'"):
                skip = True
            else:
                tb = traceback.format_exc()
        except Exception:
            tb = traceback.format_exc()
        finally:
            os.chdir(cwd)

        mpi.ibarrier(timeout=60.0)  # guard against parallel hangs

        me = np.array(tb != '')
        everybody = np.empty(mpi.size, bool)
        mpi.world.all_gather(me, everybody)
        failed = everybody.any()
        skip = mpi.world.sum(int(skip))

        if failed:
            self.fail(test, np.argwhere(everybody).ravel(), tb, t0)
            exitcode = exitcode_fail
        elif skip:
            self.register_skipped(test, t0)
            exitcode = exitcode_skip
        else:
            self.write_result(test, 'OK', t0)
            exitcode = exitcode_ok

        return exitcode

    def register_skipped(self, test, t0):
        self.write_result(test, 'SKIPPED', t0)
        self.skipped.append(test)

    def check_garbage(self):
        gc.collect()
        n = len(gc.garbage)
        self.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], self.garbage))

    def fail(self, test, ranks, tb, t0):
        if mpi.size == 1:
            text = 'FAILED!\n%s\n%s%s' % ('#' * 77, tb, '#' * 77)
            self.write_result(test, text, t0)
        else:
            tbs = {tb: [0]}
            for r in range(1, mpi.size):
                if mpi.rank == r:
                    mpi.send_string(tb, 0)
                elif mpi.rank == 0:
                    tb = mpi.receive_string(r)
                    if tb in tbs:
                        tbs[tb].append(r)
                    else:
                        tbs[tb] = [r]
            if mpi.rank == 0:
                text = ('FAILED! (rank %s)\n%s' %
                        (','.join([str(r) for r in ranks]), '#' * 77))
                for tb, ranks in tbs.items():
                    if tb:
                        text += ('\nRANK %s:\n' %
                                 ','.join([str(r) for r in ranks]))
                        text += '%s%s' % (tb, '#' * 77)
                self.write_result(test, text, t0)

        self.failed.append(test)

    def write_result(self, test, text, t0):
        t = time.time() - t0
        if self.jobs > 1:
            self.log.write('%*s' % (-self.n, test))
        self.log.write('%10.3f  %s\n' % (t, text))
