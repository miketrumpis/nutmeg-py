import numpy as np
import scipy as sp
from scipy import weave
import scipy.special
import scipy.io
import os
from nipy.core import api
from sets import Set

import nutmeg
import nutmeg.external.descriptors as desc
from nutmeg.core import TEMPLATE_MRI_PATH, full_beam_coords
from nutmeg.core import tfbeam as tfb
from nutmeg.stats import snpm_testing as snpm
from nutmeg.stats.tfstats_results import TimeFreqSnPMResults
from nutmeg.numutils import calc_fft_grid_and_map

def find_vox_intersection(vox_lists):
    """
    Find the interesction of all voxel coordinates in vox_lists.

    Parameters
    ----------
    vox_lists :
        is an iterable container of (nvox x ncoord) arrays

    Returns
    -------
    inter_vox : ndarray
        the (len(intersection) x 3) array of intersecting voxels
    vox_idx : list
        the original index of each voxel coordinate in its original list
    """
    if len(vox_lists) < 2:
        return vox_lists[0], range(len(vox_lists[0]))
    # create the lookup dictionaries for later
    lookups = []
    for vlist in vox_lists:
        d = dict(((tuple(vx), n) for (n, vx) in enumerate(vlist)))
        lookups.append(d)

    # don't bother to create the (i,j,k) lists again..
    # now they are held in the dictionary keys
    inter_set = Set(lookups[0].keys())
    for vlist in map(lambda x: getattr(x, 'keys'), lookups[1:]):
        inter_set.intersection_update(Set(vlist()))

    # make the vox coord array
    inter_vox = [ [i, j, k] for (i, j, k) in inter_set]
    inter_vox.sort()
    inter_vox = np.array(inter_vox, 'i')
    # now go through each lookup and find where inter_set voxels occur
    vox_idx = []
    for lu in lookups:
        vox_hits = [lu[tuple(vx)] for vx in inter_vox]
        #vox_hits.sort()
        vox_idx.append(vox_hits)
    return inter_vox, vox_idx

class BeamComparator(object):
    """
    This class will manage beam comparisons among conditions or across
    conditions. For any comparison specified, it will prune the voxels in
    each beam such that all voxels intersect with each other, and also with
    the MNI voxel space.
    
    """

    def __init__(self, beam_list, subject_labels, condition_labels,
                 aligned_voxels=(), **b_kws):
        """
        Parameters
        ----------
        beam_list : list
            a length-n list of beam objects of paths to beam files
        subject_labels : list
            a length-n list of labels (num, str, whatever) labeling each beam
        condition_labels : list
            a length-n list of labels (num, str, whatever) labeling each beam
        b_kws : dict, optional
            if the beam_list is supplied as a list of paths, then these are
            optional arguments to be used when loading the beams
        """
        self._compare_cache = {}
        self.subjs = np.unique(subject_labels)
        self.conds = np.unique(condition_labels)
        self.n_subj = len(self.subjs)
        self.n_cond = len(self.conds)
        self.s_labels = np.array(subject_labels)
        self.c_labels = np.array(condition_labels)
        # If the beam_list is a bunch of paths, defer loading until later
        if type(beam_list[0]) is str:
            self._blist = beam_list
            self._loaded = False
            self._b_kwargs = b_kws
        else:
            self._loaded = True
            self._blist = beam_list
            self.__beam_sanity_check(self.beams)

        # XYZ: NEVER USED PRE-ALIGNED VOXELS YET.. MAYBE KILL IT
        if aligned_voxels:
            # XYZ: STILL NEED TO CREATE THE BEAM_VOXEL_MAPS BASED ON
            # THE INTER_VOX, IF THESE PARAMETERS ARE SUPPLIED
            self.inter_vox, self.mni_voxel_map = aligned_voxels
            self.aligned = True
        else:
            self.aligned = False
##             self.align_voxels(align_all=align_all)

    def __beam_sanity_check(self, blist):
        vs = blist[0].voxelsize
        same_vox = all(filter(lambda x: (vs==x.voxelsize).all(), blist[1:]))
        sane_length = len(blist)==self.n_subj*self.n_cond
        if not same_vox:
            raise ValueError('all beams must have the same voxel resolution')
        if not sane_length:
            raise ValueError('there must be a beam for every (subj,cond) combo')

    def clear_cache(self):
        self._compare_cache = {}

    @desc.auto_attr
    def beams(self):
        if self._loaded:
            bl = self._blist
            del self._blist
            return bl
        bl = [tfb.tfbeam_from_file(f, **self._b_kwargs) for f in self._blist]
        self.__beam_sanity_check(bl)
        del self._blist
        return bl
        
    def align_voxels(self, align_all=False):
        """
        The "prototype" of align_voxels() aligns voxels across
        subjects at each condition. All resulting data are dicts keyed
        by condition.

        Parameters
        ----------
        align_all : bool
            Aligns voxels across all beams in the comparison group, rather
            than separately per condition

        Returns
        -------
        (all these items get defined on the object as a result of this method)
        inter_vox : dict
            lists of intersecting voxels
        mni_voxel_map : dict
            lists of index locations of intersecting voxels on an ordered
            list of MNI voxels
        beam_voxel_maps : dict
            lists of index locations of intersecting voxels on the ordered
            voxels list of each beam.
        """

        vs = self.beams[0].voxelsize
        mni_vox = full_beam_coords(vs)
        self.inter_vox = {}
        self.mni_voxel_map = {}
        self.beam_voxel_maps = {}
        self.aligned = True
        if align_all:
            subj_vox = [mni_vox] + [b.voxels for b in self.beams]
            ivox, bvox_map = find_vox_intersection(subj_vox)
            mni_vox_map = bvox_map.pop(0)
            # just repeat the same data over conditions
            # XYZ: THIS COULD BE RE-ENGINEERED FOR REDUNDANCY
            for c in self.conds:
                cmap = self.c_labels == c
                self.inter_vox[c] = ivox
                self.mni_voxel_map[c] = mni_vox_map
                self.beam_voxel_maps[c] = [bvox_map[cidx]
                                           for cidx, b in enumerate(cmap) if b]
            return
        for c in self.conds:
            print 'condition', c, 'comparing subjs: ',
            subj_vox = [mni_vox] + [self.beam(c,i).voxels for i in self.subjs]
            self.inter_vox[c], self.beam_voxel_maps[c] = \
                            find_vox_intersection(subj_vox)
            self.mni_voxel_map[c] = self.beam_voxel_maps[c].pop(0)
            print 'n_vox:', len(self.inter_vox[c])

    def compare(self, conditions=[]):
        """
        prototype
        
        Returns
        -------
        sample_beams : list
            a len(conditions) length list of n_subj length lists of
            comparison beams
        mni_maps : list
            a len(conditions) length list of maps to MNI voxels
        inter_vox :
            a len(conditions) length list
        avg_beams : list
            a len(conditions) length list of average comparisons
        """
        pass

    @classmethod
    def from_matlab_ptr_file(class_type, mfile, **kwargs):
        """
        This method creates a BeamComparator-type object from a MATLAB
        (or NUTMEG) pointer file.

        ex: (a shallow sublass of BeamComparitor)
        
        class MyBeamComparison (BeamComparitor):
            pass

        c = MyBeamComparison.from_matlab_ptr_file(m_file, ratio_type='f raw')
        results = c.compare()

        Parameters
        ----------
        mfile : string
            the .mat file that holds subject/condition labels and paths to
            the tfbeam mat files

        kwargs : dict
            keyword arguments for constructing TFBeams
        """        
        mat_path = os.path.abspath(mfile)
        bpath, mpath = os.path.split(mat_path)
        voi = sp.io.matlab.loadmat(mat_path,
                                   struct_as_record=True)['voi'][0,0]
        # this will need to be fixed for the RAID environment
        pnames = [os.path.join(bpath, str(p[0])) for p in voi['pathnames'][0]]
        subj_labels = voi['subjnr'][0]
        cond_labels = voi['condnr'][0]
        beam_list = []
        for p in pnames:
            pn = os.path.splitext(p)[0] + '_spatnorm.mat'
            beam_list.append(tfb.tfbeam_from_file(pn, **kwargs))
        return class_type(beam_list, subj_labels, cond_labels)


    def beam(self, cond, subj):
        """
        Returns the beam object corresponding to a given condition and subject.
        """
        c_map = self.c_labels == cond
        s_map = self.s_labels == subj
        if not (c_map & s_map).sum():
            raise ValueError('no beam with (cond, subj)=(%d, %d)'%(cond, subj))
        if (c_map & s_map).sum() > 1:
            raise ValueError('Beams not uniquely specified: ambiguously labeled BeamComparator')
        beam_idx = np.argwhere(c_map & s_map)
        return self.beams[beam_idx]
        
    def beam_sig(self, cond, subj):
        """
        Returns the signal from the beam indexed by (cond, subj).
        Importantly, returned samples of the function s(vox) ordered in
        a consistent voxel ordering
        """
        beam = self.beam(cond, subj)
        c_map = self.c_labels == cond
        s_map = self.s_labels[c_map]
        beam_idx = np.argwhere(s_map==subj)
        if beam_idx.sum() < 1:
            err = 'Subject '+str(subj)+' has no map for condition '+str(cond)
            raise ValueError(err)
        if beam_idx.sum() > 1:
            raise ValueError('Beams not uniquely specified: ambiguously labeled BeamComparator')
        m = self.beam_voxel_maps[cond][beam_idx]
        return beam.s[m]

class BeamActivationAverager(BeamComparator):
    """
    This BeamComparator looks at the activation in a given condition for
    a group of subjects.
    """

    def beam_sig(self, cond, subj):
        """
        Returns the signal from the beam indexed by (cond, subj).
        Importantly, returned samples of the function s(vox) ordered in
        a consistent voxel ordering
        """
        b = self.beam(cond, subj)
        m_idx = np.argwhere(self.subjs==subj).flat[0]
        m = self.beam_voxel_maps[cond][m_idx]
        return b.s[m]

    def compare(self, conditions=[]):
        """
        Returns a list of activations, and an average activation for
        each condition specified
        """
        if type(conditions) is not list:
            conditions = list(conditions)
        if not conditions:
            conditions = self.conds
        if not self.aligned:
            self.align_voxels()
##         if tuple(conditions) in self._compare_cache:
##             return self._compare_cache[tuple(conditions)]
##         # otherwise make new comparison
        avg_beams = []
        act_beams = []
        for c in conditions:
            condition_act_beams = []
            avg_sig = np.zeros_like(self.beam_sig(c,self.subjs[0]))
            ivox = self.inter_vox[c]
            for i in self.subjs:
                b = self.beam(c,i)
                b_sig = self.beam_sig(c,i)
                avg_sig += b_sig
                pruned_beam = b.from_new_dataset(b_sig, new_vox=ivox,
                                                 fixed_comparison=b.uses)
                condition_act_beams.append(pruned_beam)
            avg_sig /= self.n_subj
            avg_beams.append(b.from_new_dataset(avg_sig,
                                                new_vox=ivox,
                                                multi_subj=True,
                                                fixed_comparison=b.uses))
            act_beams.append(condition_act_beams)

##         self._compare_cache[tuple(conditions)] = (tuple(act_beams),
##                                                   tuple(avg_beams))
        return act_beams, avg_beams
            

class BeamContrastAverager(BeamComparator):
    """
    This BeamComparator looks at the contrast between conditions for a
    group of subjects.
    """

    # Uses default voxel alignment across condition and subject

    def compare(self, conditions=[]):
        """
        Returns a contrast beam for each subject, for each condition pair
        specified, as well as an average contrast beam for each condition pair.
        If the conditions are specified as [(a,b)], the contrast is
        beam(a, subj) - beam(b, subj)
        """
        if not conditions:
            # default to pairing off conditions in order???
            conditions = self.conds.reshape(self.n_cond/2, 2).tolist()
        if not hasattr(conditions[0], '__iter__'):
            conditions = [ conditions ]
        if len(filter(lambda x: len(x)==2, conditions)) != len(conditions):
            raise ValueError(
    """Some of the provided condition specs are not in pairs %s"""%repr(conditions)
        )
        if not self.aligned:
            self.align_voxels()

            
##         if tuple(conditions) in self._compare_cache:
##             return self._compare_cache[tuple(conditions)]
##         # otherwise make new comparison            
        avg_beams = []
        contrast_beams = []
        for pair in conditions:
            c1, c2 = pair
            avg_contrast = np.zeros_like(self.beam_sig(self.conds[0],
                                                       self.subjs[0]))
            cond_contrasts = []
            for i in self.subjs:
                b1 = self.beam(c1,i)
                s = self.beam_sig(c1,i) - self.beam_sig(c2,i)
                # XYZ: INCONVENIENT SYNTAX FOR FINDING THE CORRECT INTER_VOX
                cond_contrasts.append(
                    b1.from_new_dataset(s,
                                        new_vox=self.inter_vox[pair[0]],
                                        fixed_comparison=b1.uses)
                    )
                avg_contrast += s
                contrast_beams.append(cond_contrasts)
                avg_contrast /= self.n_subj
                avg_beams.append(
                    b1.from_new_dataset(avg_contrast,
                                        new_vox=self.inter_vox[pair[0]],
                                        multi_subj=True,
                                        fixed_comparison=b1.uses)
                    )
                
##         self._compare_cache[tuple(conditions)] = (tuple(act_beams),
##                                                   tuple(avg_beams))
        return contrast_beams, avg_beams
    
    def align_voxels(self, **kw):
        # silently ignore the kwarg
        super(BeamContrastAverager, self).align_voxels(align_all=True)
        
## XYZ: SHOULD MOVE THE FOLLOWING CLASSES INTO A TIME-FREQ STATS MODULE
        
class SnPMTester(object):
    smoothed_variance = True
    def __init__(self, beam_comp, conditions, n_perm, **kwargs):
        # need to set up:
        # self.sample_beams -- the list of beams whose signals to test
        # self.dm_gen -- the appropriate design matrix generator
        # self.test_stat -- the type of stat to calculate in snpm_testing
        # self.co -- the comparison weights for the stats design solution
        # self.inter_vox -- the mni voxels of the stat comparison
        # self.mni_vox_map -- the indices of the mni voxels of the stat comp
        pass

    def test(self):
        """
        Runs a univariate statistical test at each time-frequency-voxel,
        filling in these measures:

        T -- the statistic at each point
        maxT -- the maximum statistic of all permuted measurements
        minT -- the minimum statistic of all permuted measurements
        percentiles -- the "ranking" of the statistics in T relative to all
                       the permuted statistics

        """
        if not self._is_init:
            print 'loading and comparing beams'
            self._init_data()

        n_perm_tested = self.n_perm/2 if self.half_perms else self.n_perm
        beams = self.sample_beams
        b = beams[0]
        n_vox = len(b.voxels)
        n_bands = len(b.bands)
        n_times = len(b.timewindow)
        vox_size = b.voxelsize
        # padding the volume to be centered can't hurt, leave it for now
        grid_shape, flat_map = calc_fft_grid_and_map(b)

        # at each (t, f) pt, want to find:
        # T test
        # max T stat
        # min T stat
        # percentiles (converted to uncorrected p scores)
        T = np.empty( (n_vox, n_times, n_bands) )
        maxT = np.empty((self.n_perm, n_times, n_bands))
        minT = np.empty((self.n_perm, n_times, n_bands))
        percentiles = np.empty_like(T)
        # the array of test results
        tt = np.empty((self.n_perm, n_vox))
        for t in xrange(n_times):
            for f in xrange(n_bands):
                print 'performing stat test at (t,f) = (',t,f,')'
                X = np.array([b.s[:,t,f] for b in beams])
                snpm.run_snpm_tests(X, self.dm_gen, self.co, n_perm_tested,
                                    grid_shape, flat_map, vox_size,
                                    smoothed_variance=self.smoothed_variance,
                                    out=tt[:n_perm_tested])
                self.dm_gen.reset()
                true_t = tt[0]
                # get T distributions 
                if self.half_perms:
                    tt[n_perm_tested:] = -tt[:n_perm_tested]
                maxT[:,t,f] = tt.max(axis=1)
                minT[:,t,f] = tt.min(axis=1)
                tt = np.sort(tt, axis=0)
                # The 1st column holds the number of test values
                # smaller than true_t.
                # The 2nd column is the voxel index.
                locs = np.argwhere(true_t==tt)
                vloc = locs[:,1]
                percentiles[vloc,t,f] = locs[:,0]/float(self.n_perm)
                T[:,t,f] = true_t

        return TimeFreqSnPMResults(
            T, self.avg_beams[0].voxel_indices, percentiles, maxT, minT
            )

class SnPMOneSampT(SnPMTester):
    """
    Performs a one sample t-test on the results of
    beam_comp.compare(conditions=conditions).
    May be a test of activation significance, contrast significance, etc.

    Tests whether the mean of the comparisons is significantly different than 0.
    """

    test_stat = 'T'
    co = np.array([[1]])

    @staticmethod
    def num_observations(condition, c_labels, s_labels):
        """
        Find the number of observations in a One Sample SnPM T-test
        set up, for given specs

        Parameters
        ----------
        condition : str or iterable
            a condition or contrast label
        c_labels : iterable
            all condition labels in the comparison (IE, BeamComparator.c_labels)
        s_labels : iterable
            all subject labels in the comparison (IE, BeamComparator.s_labels)

        Returns
        -------
        the number of observations for the condition
        """
        # n_obs is the number of subjects for each condition..
        # if a contrast, safe to use first condition of pair
        while hasattr(condition, '__iter__'):
            condition = condition[0]
        return len( [n for n in xrange(len(s_labels))
                     if c_labels[n]==condition] )
            
    @staticmethod
    def num_possible_permutations(condition, c_labels, s_labels):
        return 2**SnPMOneSampT.num_observations(condition, c_labels, s_labels)

    def __init__(self, beam_comp, condition, n_perm,
                 force_half_perms=False,
                 init=True):
        """
        Sets up an SnPMTester for a 1-sample T test.

        Parameters
        ----------
        beam_comp : a BeamComparator
        condition : label, or list of labels
            If testing a BeamActivationAverager: one of beam_comp.conds
            If testing a BeamContrastAveragor: a pair drawn from beam_comp.conds
        n_perm : int
            The number of permutations, up to 2**(number-of-observations).
            If n_perm is given as the maximum, only half the permutations
            will actually be tested (from symmetry of the permutations).
        force_half_perms : bool, optional
            Ignore the n_perm argument, and test half the permutations
            (in order to compute the full set of permutations)
        init : bool, optional
            Enforce loading and computation of comparisons at construction
            time. Otherwise, initialization will be deferred until test time.
        """

        if type(beam_comp) is BeamContrastAverager and \
               len(condition) != 2:
            raise ValueError(
"""You only may test one contrast pair for a contrast test"""
        )

        if type(beam_comp) is BeamActivationAverager and \
           hasattr(condition, '__iter__'):
            raise ValueError(
"""You only may test one condition for an activation test"""
        )

        self.condition = condition
        self.comp = beam_comp
        
        n_obs = SnPMOneSampT.num_observations(condition,
                                              self.comp.c_labels,
                                              self.comp.s_labels)

        # this creates a design matrix generator with up to half the possible
        # permutations, presented in random order following the first
        # correct combination
        if force_half_perms or n_perm==(2**(n_obs-1)):
            self.n_perm = 2**n_obs
            self.dm_gen = snpm.one_sample_ttest_design_generator(
                n_obs, half=True
                )
            self.half_perms = True
        else:
            if n_perm > 2**n_obs:
                raise ValueError(
"""There are not enough possible permutations to accomodate %d permutation
tests"""%n_perm
                    )
            self.n_perm = n_perm
            self.dm_gen = snpm.one_sample_ttest_design_generator(
                n_obs, half=False
                )
            self.half_perms = False
            

##         self.sample_beams = sample_beams
        if init:
            self._init_data()
        else:
            self._is_init = False
    
    def _init_data(self):
        self.sample_beams, self.avg_beams = self.comp.compare(
            conditions=self.condition
            )
        self.sample_beams = self.sample_beams[0]
        # XYZ: I think this is redundant, given the call above?
        if not self.comp.aligned:
            self.comp.align_voxels()
        self._is_init = True
        

class SnPMUnpairedT(SnPMTester):
    """
    Performs a t-test of significance between either:
    2 conditions listed in conditions (eg: [1, 2])
    2 condition contrasts listed in conditions (eg: [[1,2],[3,4]])

    If sample_beams is provided, it is also a lenth-2 list-of-lists,
    where each sublist contains the TFBeam samples for its condition.
    """
    
    test_stat = 'T'
    co = np.array([[-1,1]])
    
    @staticmethod
    def num_observations(cpair, c_labels, s_labels):
        """
        Find the number of observations in Unpaired SnPM T-test
        set up, for given specs

        Parameters
        ----------
        cpair : iterable
            a pair of condition or contrast labels
        c_labels : iterable
            all condition labels in the comparison (IE, BeamComparator.c_labels)
        s_labels : iterable
            all subject labels in the comparison (IE, BeamComparator.s_labels)

        Returns
        -------
        (na, nb) : the numbers of observations for the each condition
        in the pair
        """
        while hasattr(cpair[0], '__iter__'):
            cpair = [c[0] for c in cpair]
        n_obs_a = SnPMOneSampT.num_observations(cpair[0], c_labels, s_labels)
        n_obs_b = SnPMOneSampT.num_observations(cpair[1], c_labels, s_labels)
        return (n_obs_a, n_obs_b)

    @staticmethod
    def num_possible_permutations(cpair, c_labels, s_labels):
        n_obs_a, n_obs_b = SnPMUnpairedT.num_observations(
            cpair, c_labels, s_labels
            )
        return int(round(sp.special.gamma(n_obs_a+n_obs_b+1) / \
                         sp.special.gamma(n_obs_a+1) / \
                         sp.special.gamma(n_obs_b+1)))

    def __init__(self, beam_comp, conditions, n_perm,
##                  sample_beams=[],
                 force_half_perms=False,
                 init=True):
        if len(conditions) != 2:
            raise ValueError(
"""Condition list must be length-2"""
        )
        if type(beam_comp) is BeamContrastAverager and len(conditions[0]) != 2:
            raise ValueError(
"""For contrasts, both test conditions must be a pair of contrast conditions"""
        )

        self.comp = beam_comp
        self.conditions = conditions
        n_obs_a, n_obs_b = SnPMUnpairedT.num_observations(
            conditions, self.comp.c_labels, self.comp.s_labels
            )
        max_perms = SnPMUnpairedT.num_possible_permutations(
            conditions, self.comp.c_labels, self.comp.s_labels
            )
        
        if force_half_perms or n_perm==(max_perms/2):
            self.n_perm = max_perms
            self.half_perms = True
            self.dm_gen = snpm.unpaired_ttest_design_generator(
                n_obs_a, n_obs_b, half=True
                )
        else:
            if n_perm > max_perms:
                raise ValueError(
"""There are not enough possible permutations to accomodate %d permutation
tests"""%n_perm
                    )
            self.n_perm = n_perm
            self.half_perms = False
            self.dm_gen = snpm.unpaired_ttest_design_generator(
                n_obs_a, n_obs_b, half=False
                )
            
        if init:
            self._init_data()
        else:
            self._is_init = False
        
    def _init_data(self):
        conditions = self.conditions
        if not self.comp.aligned or \
               len(self.comp.inter_vox[conditions[0]]) != len(self.comp.inter_vox[conditions[1]]):
            self.comp.align_voxels(align_all=True)

        sample_beams, self.avg_beams = self.comp.compare(conditions)
        self.sample_beams = sample_beams[0]+sample_beams[1]
        self._is_init = True

            
        
        
