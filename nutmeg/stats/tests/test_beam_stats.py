import scipy as sp
import scipy.io
import numpy as np
import numpy.testing as npt
import nose.tools as nt

from nutmeg.core.tfbeam import tfbeam_from_file
from nutmeg.stats.beam_stats import *
from nutmeg import utils
from nutmeg.core.tests.test_tfbeam import gen_tfbeam

def test_vox_intersection():
    # vlist1 is a subset of vlist2
    vlist1 = utils.voxel_index_list((5,5,5))
    vlist2 = utils.voxel_index_list((7,7,7))

    ivox, maps = find_vox_intersection([vlist1, vlist2])

    intersection = set([ tuple(vx) for vx in ivox ])
    v1_set = set([ tuple(vx) for vx in vlist1 ])
    
    yield nt.assert_true, len(intersection.difference(v1_set))==0, \
          'intersection failed'

    m1 = set(maps[0])
    yield nt.assert_true, len(m1)==ivox.shape[0], 'voxel map1 missing elements'
    yield nt.assert_true, len(maps[0])==len(maps[1]), 'voxel map2 missing elements'

    v2_inside = vlist2[maps[1]]
    v2_set = set([ tuple(vx) for vx in v2_inside])

    yield nt.assert_true, len(intersection.difference(v2_set))==0, \
          'intersection failed'


class testComparisons:

    @classmethod
    def setup_class(klass):
        klass.c1_beams = [gen_tfbeam(fixed_comparison='f db') for i in range(4)]
        klass.c2_beams = [gen_tfbeam(fixed_comparison='f db') for i in range(4)]
        klass.subj_labels = 'darius', 'xerxes', 'cyrus', 'artaxerxes'
        klass.c1_labels = ['mortal man']*4
        klass.c2_labels = ['god emperor']*4

class testActivationComp(testComparisons):
    
##     @classmethod
##     def setup_class(cls):
##         print 'running class setup on:', cls
##         testComparisons.setup_class()
##         cls.a_comp = BeamActivationAverager(cls.c1_beams + cls.c2_beams,
##                                             cls.subj_labels + cls.subj_labels,
##                                             cls.c1_labels + cls.c2_labels)

    def setUp(self):
        # make an activation comparison with 2 conditions
        cls = self.__class__
        self.a_comp = BeamActivationAverager(cls.c1_beams + cls.c2_beams,
                                             cls.subj_labels + cls.subj_labels,
                                             cls.c1_labels + cls.c2_labels)
        
    def test_setup_comp(self):
        assert (self.a_comp.n_subj == 4 and self.a_comp.n_cond == 2), \
               'wrong # of subjs and/or conds'
        
    def test_alignment(self):
        self.a_comp.align_voxels()
        c1, c2 = self.c1_labels[0], self.c2_labels[1]
        l = []
        for map_name in ('inter_vox', 'beam_voxel_maps'):
            map = getattr(self.a_comp, map_name)
            l.append( c1 in map and c2 in map )
        assert all(l), 'voxel maps missing one or more condtions'

    def test_comparison(self):
        c1, c2 = self.c1_labels[0], self.c2_labels[1]
        samples, avgs = self.a_comp.compare(conditions=[c1, c2])
        assert len(samples)==2 and len(avgs)==2, 'wrong # of results'

    def test_comparison_by_ref(self):
        c1, c2 = self.c1_labels[0], self.c2_labels[1]
        samples, avgs = self.a_comp.compare(conditions=[c1, c2])
        m1 = np.array([self.a_comp.beam_sig(c1, subj) for subj in self.subj_labels])
        m1 = m1.mean(axis=0)
        m2 = np.array([self.a_comp.beam_sig(c2, subj) for subj in self.subj_labels])
        m2 = m2.mean(axis=0)
        err = (avgs[0].s-m1)**2 + (avgs[1].s-m2)**2
        npt.assert_array_almost_equal(err, np.zeros_like(err))

class testContrastComp(testActivationComp):

    def setUp(self):
        # make a contrast comparison with 2 conditions
        cls = self.__class__
        self.a_comp = BeamContrastAverager(cls.c1_beams + cls.c2_beams,
                                           cls.subj_labels + cls.subj_labels,
                                           cls.c1_labels + cls.c2_labels)
    def test_comparison(self):
        c1, c2 = self.c1_labels[0], self.c2_labels[1]
        samples, avgs = self.a_comp.compare(conditions=[c1, c2])
        assert len(samples)==1 and len(avgs)==1, 'wrong # of results'

    def test_comparison_by_ref(self):
        c1, c2 = self.c1_labels[0], self.c2_labels[1]
        samples, avgs = self.a_comp.compare(conditions=[c1, c2])
        m1 = np.array([self.a_comp.beam_sig(c1, subj) for subj in self.subj_labels])
        m2 = np.array([self.a_comp.beam_sig(c2, subj) for subj in self.subj_labels])
        m = (m1 - m2).mean(axis=0)
        err = (avgs[0].s-m)**2
        npt.assert_array_almost_equal(err, np.zeros_like(err))

# Misc Utilities
# ----------------------------------------------------------------------------
def compare_to_mat_beam(pybeam, matbeam, comp='diff', return_mat=False):
    if type(pybeam) == str:
        pybeam = tfbeam_from_file(pybeam)
    if type(matbeam) == str:
        matbeam = tfbeam_from_file(matbeam)

    inter_vox, vox_indices = find_vox_intersection([pybeam.voxels,
                                                    matbeam.voxels])
    if len(inter_vox) != len(pybeam.voxels) or \
       len(pybeam.voxels) != len(matbeam.voxels):
        print 'mis-matched voxels between MATLAB and Python beams'
        # trim pybeam to the inter_vox
        s = np.array( [pybeam.s[pybeam.vox_lookup[tuple(v)]]
                       for v in inter_vox] )
        pybeam = pybeam.from_new_data(
            s, inter_vox, fixed_comparison=pybeam.uses
            )

    # need to re-order the matbeam signal since the voxels are in diff. order
    s = np.zeros_like(pybeam.s)
    for vsig, v in zip(matbeam.s, matbeam.voxels):
        try:
            i = pybeam.vox_lookup[tuple(v)]
            s[i] = vsig
        except KeyError:
            pass
    matbeam = matbeam.from_new_data(
        s, new_vox=pybeam.voxels.copy(), fixed_comparison=matbeam.uses
        )
    if comp=='diff':
        
        err = pybeam.from_new_data(
            matbeam.s - pybeam.s,
            new_vox=inter_vox,
            fixed_comparison='diff'
            )
    elif comp=='pctdiff':
        err = pybeam.from_new_data(
            100.*(matbeam.s-pybeam.s)/pybeam.s,
            new_vox=inter_vox,
            fixed_comparison='pctdiff'
            )
    elif comp=='sqrdiff':
        err = pybeam.from_new_data(
            (matbeam.s-pybeam.s)**2,
            new_vox=inter_vox,
            fixed_comparison='sqrdiff')
    if return_mat:
        return err, matbeam
    else:
        return err
