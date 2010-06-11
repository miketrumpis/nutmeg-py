import numpy as np
import scipy as sp
import scipy.special
import os

from nutmeg.stats import snpm_testing as snpm
from nutmeg.stats.tfstats_results import TimeFreqSnPMResults
from nutmeg.numutils import calc_fft_grid_and_map
       
class SnPMTester(object):
    smoothed_variance = True
    def __init__(self, beam_comp, conditions, n_perm, **kwargs):
        # need to set up:
        # self.sample_beams -- the list of beams whose signals to test
        # self.dm_gen -- the appropriate design matrix generator
        # self.test_stat -- the type of stat to calculate in snpm_testing
        # self.co -- the comparison weights for the stats design solution
        # self.inter_vox -- the mni voxels of the stat comparison
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
        """
        Find the number of re-weighted means in a One Sample SnPM
        T-test set up, for given specs

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
        the maximum number of re-weighted means
        """

        return 2**SnPMOneSampT.num_observations(condition, c_labels, s_labels)

    def __init__(self, beam_comp, condition, n_perm,
                 force_half_perms=False,
                 init=True):
        """
        Sets up an SnPMTester for a 1-sample T test. This test is
        performed on the condition, or conditions contrast specified.

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
        Find the number of observations in an Unpaired SnPM T-test
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
        """
        Find the number of re-weighted means in an Unpaired SnPM T-test
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
        the maximum number of re-weighted means
        """        
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
