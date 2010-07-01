import os
import numpy as np
from scipy import stats, ndimage

import xipy.volume_utils as vu

from nutmeg.core import tfbeam
from nutmeg.utils import array_pickler_mixin, calc_grid_and_map
import nutmeg.stats.stats_utils as su

def load_tf_snpm_stats(snpm_arrays):
    """Loads a TimeFreqSnPMResults

    Parameters
    ----------
    snpm_arrays : a path or ndarray of SnPM data
    """
    if type(snpm_arrays) in (str, unicode):
        ext_type = os.path.splitext(snpm_arrays)[-1]
        if ext_type == '.mat':
            return adapt_mlab_tf_snpm_stats(snpm_arrays)
        elif ext_type in ('.npz', '.npy'):
            return TimeFreqSnPMResults.load(snpm_arrays)
        else:
            raise ValueError('did not recognize the file format: '+snpm_arrays)
    return TimeFreqSnPMResults(t, vox_map, p, mxt, mnt)
    

def adapt_mlab_tf_snpm_stats(combo_beam, avg_beam=None):
    mdict, mbeam, dof = su.split_combo_tfstats_matfile(combo_beam)
    stat = mdict['T test'].s
    ranks = mdict['p val neg (uncorr)'].s
    p_scores_neg = mdict['p val neg (corr)'].s
    p_scores_pos = mdict['p val pos (corr)'].s
    if avg_beam is None:
        avg_beam = mbeam
    return AdaptedTimeFreqSnPMResults(
        stat, avg_beam.voxel_indices, ranks, p_scores_pos, p_scores_neg
        )


class TimeFreqSnPMResults(array_pickler_mixin):

    threshold_types = [
        'Test score',
        'Test score (both tails)',
        'Test score (pos tail)',
        'Test score (neg tail)',
        'Cluster size (pos tail)',
        'Cluster size (neg tail)']

    # these are the names of the attributes on the object
    _argnames = ['t', 'vox_idx', 't_rank', '_max_t', '_min_t']
    
    def __init__(self,
                 vox_stat,
                 vox_map,
                 vox_stat_ranking,
                 max_stat_dist,
                 min_stat_dist):
        """Create an object that can report various statistical thresholds
        from SnPM testing results.

        Parameters
        ----------
        vox_stat : ndarray
          The statistical test value at each voxel
        vox_map : ndarray
          The (nvox x 3) voxel index map into a volume array
        vox_stat_ranking : ndarray
          The true test ranking in the sorted set of permuted test values.
          EG: if the true test value is the 75th largest among 100 permuted
          test results, vox_stat_ranking(r) = 0.75
        max_stat_dist : ndarray
          The maximal test result found at each voxel
        min_stat_dist : ndarray
          The minimal test result found at each voxel
        """
        self.vox_idx = vox_map
        self.t = vox_stat
        self.t_rank = vox_stat_ranking
        self._max_t = max_stat_dist
        self._min_t = min_stat_dist
        # the step size of significance quantiles
        self.dp = 1.0 / self._max_t.shape[0]

    def _fix_dist(self, tail, corrected_dims, pooled_dims):
        """Return the requested distribution, sorted (in the sense of
        low p value to high p value) along axis 0. Perform any correction
        or pooling across requested dimensions.

        Parameters
        ----------
        tail : str in {'pos', 'neg'}
        corrected_dims : sequence
          which dims (other than 0) to be reduced to maximal values
        pooled_dims : sequence
          which dims (other than 0) to be pooled together with the
          maximal distribution axis, in order to produce a richer distribution
        """
        i = set(pooled_dims).union(set(corrected_dims)).difference(set((1,2)))
        if i:
            dims = [d for d in i]
            raise ValueError('Dimensions out of range: '+str(dims))
        i = set(pooled_dims).intersection(set(corrected_dims))
        if i:
            d = {0: 'max_stat', 1: 'time', 2: 'frequency'}
            dims = [d[n] for n in i]
            raise ValueError('Cannot to pool and/or correct dims: '+str(dims))

        ext_t = -self._max_t if tail.lower()=='pos' else self._min_t
        # if correcting dims, simply reduce that dimension with its
        # minimal value, and replace the dimension size with 1
        for d in corrected_dims:
            s = list(ext_t.shape)
            s[d] = 1
            ext_t = ext_t.min(axis=d)
            ext_t.shape = tuple(s)
        
        # if pooling dims, roll them all to the front, and then
        # reshape to have len(pooled_dims) as the 1st dimension
        s = list(ext_t.shape)
        indpt_dims = set(range(len(s))).difference(set(pooled_dims))
        new_shape = [0] * len(s)
        
        for d in pooled_dims:
            # replace shape at these dimensions with 1
            new_shape[d] = 1
            ext_t = np.rollaxis(ext_t, d)
        for d in indpt_dims:
            # restore original shape at these dimensions
            new_shape[d] = s[d]
        # set shape of 1st dimension as the product of the pooled dims
        new_shape[0] *= np.prod(np.take(np.array(s), pooled_dims))
        ext_t = ext_t.reshape(tuple(new_shape))
        # want to return the negative tail distribution running from
        # neg-to-pos values, and the positive tail distribution running
        # from pos-to-neg values
        ext_t = np.sort(ext_t, axis=0)
        return -ext_t if tail.lower()=='pos' else ext_t

    def threshold(self, alpha, tail,
                  pooled_dims=(),
                  corrected_dims=()):
        """Find a the statistic value threshold for the significance level
        alpha, either in the positive or negative tail of the maximal
        stats distributions. Optionally pool the distributions across
        dimensions, or further correct for multiple comparisons across
        dimensions (IE, duplicate maximal stats across those dims)

        Parameters
        ----------
        alpha : float
          The significance threshold (as in p < 0.05)
        tail : str
          {'pos', 'neg'} -- return p values for whether the test statistic
          if significantly large or small, respectively
        pooled_dims : tuple, optional (can not intersect with corrected_dims)
          If not empty, pool the extreme statistics across the given dims
          to form a new maximal distribution.
        corrected_dims : tuple, optional (can not intersect with pooled_dims)
          If not empty, replace the distribution along this dimension with
          the single maximal value.

        Returns
        -------
        (threshold(t,f), exact_alpha)
        """
        dist = self._fix_dist(tail, corrected_dims, pooled_dims)
        N = dist.shape[0]
        # find highest k such that k/N < alpha
        k = min(N-1, int( alpha*N ))
        # broadcast this up to (1,ntp,nfp) dims
        tc = dist[k] * np.ones( (1,) + self.t.shape[1:])
        return tc, float(k)/N        

    def uncorrected_p_score(self, tail):
        """The t_rank represent how much of the mass of the SnPM
        distribution (at each voxel) is less than the true test value.

        Parameters
        ----------
        tail : str
          {'pos', 'neg'} -- return p values for whether the test statistic
          if significantly large or small, respectively

        Returns
        -------
        p : ndarray
          a map of P values for the chosen tail

        Note
        ----
        a p value of 0.05 represents a 5% chance that the test value
        was truly drawn from the null distribution
        """
        if tail=='neg':
            return self.t_rank
        else:
            return (1 + self.dp - self.t_rank)
        p = self.t_rank if tail=='neg' else 1-self.t_rank
        return p

    def p_score_from_maximal_statistics(self, tail, t=None,
                                        pooled_dims=(),
                                        corrected_dims=()):
        """Returns family-wise corrected p values for the requested tail
        based on the distribution of the most extreme statistics from
        each permuted test. Optionally pool the distributions across
        dimensions, or further correct for multiple comparisons across
        dimensions (IE, duplicate maximal stats across those dims)

        Parameters
        ----------
        tail : str
          {'pos', 'neg'} -- return p values for whether the test statistic
          if significantly large or small, respectively
        t : ndarray, optional
          Optionally, score these statistics against (rather than this object's
          statistics) against the maximal stat distribution.
        pooled_dims : tuple, optional (can not intersect with corrected_dims)
          If not empty, pool the extreme statistics across the given dims
          to form a new maximal distribution.
        corrected_dims : tuple, optional (can not intersect with pooled_dims)
          If not empty, replace the distribution along this dimension with
          the single maximal value.
        """
        if t is None:
            t = self.t
        else:
            assert t.shape == self.t.shape, 'shape mismatch, cannot map'

        dist = self._fix_dist(tail, corrected_dims, pooled_dims)
        p_table = np.linspace(0,1,dist.shape[0]+1,endpoint=True)
        if tail=='pos':
            dist = dist[::-1]
            p_table = p_table[::-1]

        # XYZ: maybe this can be smarter
        nt, nf = t.shape[1:]
        p_vals = np.empty_like(t)
        for tp in xrange(nt):
            for fp in xrange(nf):
                dt = 0 if dist.shape[1]==1 else tp
                df = 0 if dist.shape[2]==1 else fp
                t_indexed = su.index(t[:,tp,fp], dist[:,dt,df])
                np.take(p_table, t_indexed, out=p_vals[:,tp,fp])
                        
        
        return p_vals
        
    def map_of_significant_clusters(self, tail, alpha=0.05, gamma=0.05):
        """Make a map based on cluster-size significance.
        First find all the clusters of voxels exceeding the
        individual-statistic p value of alpha. Then, from a distribution
        of cluster sizes, return a map of clusters whose size exceed
        the p value of gamma.

        Parameters
        ----------
        tail : str in {'pos', 'neg'}
          Use p values for whether the test statistic if significantly
          large or small, respectively
        alpha : float
          Individual test statistic confidence level
        gamma : float
          Cluster size statistic confidence level

        Returns
        -------
        cluster_map : ndarray
          binary map which is non-negative at significant clusters

        """
        # map individually tested p values to a volume grid,
        # find cluster sizes,
        # determine significant cluster threshold,
        # mask out sub-threshold clusters,
        # convert back to voxel lists
        p_score = self.uncorrected_p_score(tail)
        nt, nf = p_score.shape[1:]
        ni, nj, nk = self.vox_idx.max(axis=0)+1
        cluster_vols = np.empty( (ni, nj, nk, nt, nf) )
        max_labels = np.empty((nt,nf))
        c_sizes = []
        for t in xrange(nt):
            for f in xrange(nf):
                vol = vu.signal_array_to_masked_vol(
                    p_score[:,t,f],
                    self.vox_idx,
                    fill_value=1
                    ).filled()
                cluster_vols[:,:,:,t,f], max_labels[t,f] = ndimage.label(
                    vol <= alpha
                    )
                for lval in xrange(1, int(max_labels[t,f])):
                    c_sizes.append( (cluster_vols[:,:,:,t,f]==lval).sum() )
        c_sizes = np.sort(np.array(c_sizes))
        c_sz_cutoff = c_sizes[ int(np.ceil((1-gamma) * len(c_sizes))) ]
        survivor_label = int(max_labels.max() + 1)
        # now go and re-label all super/sub threshold clusters
        for t in xrange(nt):
            for f in xrange(nf):
                for lval in xrange(1, int(max_labels[t,f])):
                    if (cluster_vols[:,:,:,t,f]==lval).sum() < c_sz_cutoff:
                        relabel = 0
                    else:
                        relabel = survivor_label
                    np.putmask(cluster_vols[:,:,:,t,f],
                               cluster_vols[:,:,:,t,f] == lval,
                               relabel)

        # now re-map to voxel list arrays
        tf_size = nt*nf
        strides = np.array([nj*nk, nk, 1]) * tf_size
        flat_map = (self.vox_idx * strides).sum(axis=1)
        flat_map = (flat_map[:,None] + np.arange(tf_size)[None,:]).ravel()
        cluster_map = cluster_vols.flat[flat_map]
        del cluster_vols
        return cluster_map.reshape(self.t.shape)

class AdaptedTimeFreqSnPMResults(TimeFreqSnPMResults):
    """This class adapts MATLAB Nutmeg results for TimeFreqSnPMResults
    functionality. 
    """

    def __init__(self, vox_stat, vox_map, vox_stat_ranking,
                 p_scores_pos, p_scores_neg):
        """Create an object that can report various statistical thresholds
        from MATLAB Nutmeg SnPM testing results. These results include
        the voxel statistic map, and the un/corrected p scores.

        Parameters
        ----------
        vox_stat : ndarray
          The statistical test value at each voxel
        vox_map : ndarray
          The (nvox x 3) voxel index map into a volume array
        vox_stat_ranking : ndarray
          This is equivalent to "uncorrected p values", which is 
          the true test ranking in the sorted set of permuted test values.
          EG: if the true test value is the 75th largest among 100 permuted
          test results, vox_stat_ranking(r) = 0.75
        p_scores_pos : ndarray
          The family-wise corrected significance scores at each voxel
          (positive tail)
        p_scores_neg : ndarray
          The family-wise corrected significance scores at each voxel
          (negative tail)        
        """
        # potentially NON-robust estimation of the quantiles..
        # could have this be a constructor argument too
        dp = np.diff(np.unique(vox_stat_ranking.ravel())).min()
        min_t, max_t = self._estimate_maximal_stats(
            vox_stat, p_scores_pos, p_scores_neg, dp
            )
        TimeFreqSnPMResults.__init__(
            self, vox_stat, vox_map, vox_stat_ranking, max_t, min_t
            )

    def _estimate_maximal_stats(self, ts, ppos, pneg, dp):
        n = int(1.0/dp)
        nt, nf = ts.shape[1:]
        max_t = np.empty((n, nt, nf))
        min_t = np.empty((n, nt, nf))
        for t in xrange(nt):
            for f in xrange(nf):
                edges, _ = su.map_t(ts[:,t,f], pneg[:,t,f], dp)
                min_t[:,t,f] = edges
                edges, _ = su.map_t(-ts[:,t,f], ppos[:,t,f], dp)
                max_t[:,t,f] = -edges
        return min_t, max_t
        
    
    def __array__(self):
        raise NotImplementedError("Adapted results will not be saved")

class TimeFreqSnPMClusters(TimeFreqSnPMResults):

    _argnames = TimeFreqSnPMResults._argnames + \
                ['all_ptail_clusters', 'all_ntail_clusters']

    threshold_types = TimeFreqSnPMResults.threshold_types + \
                      ['SnPM Cluster scores (pos tail)',
                       'SnPM Cluster scores (neg tail)']

    @staticmethod
    def from_stats_and_clusters(stats_res, pos_clusters, neg_clusters):
        # make a big nested list of all clusters at all permutations,
        # marking down empty lists as such
##         ptail_clusters = [ [ [StatCluster(*c) for c in pc] if pc else []
##                              for pc in all_perms]
##                            for all_perms in pos_clusters]
##         ntail_clusters = [ [ [StatCluster(*c) for c in pc] if pc else []
##                              for pc in all_perms]
##                            for all_perms in neg_clusters]
        args = [ getattr(stats_res, a) for a in TimeFreqSnPMResults._argnames]
        args += [pos_clusters, neg_clusters]
        return TimeFreqSnPMClusters(*args)

    def __init__(self, vox_stat, vox_map, vox_stat_ranking,
                 max_stat_dist, min_stat_dist,
                 snpm_ptail_clusters, snpm_ntail_clusters):
        """
        This class analyzes the cluster information at each permutation
        from a SnPM-style test in order to score the clusters found at
        the truly labeled test.
        
        In addition, it can make thresholds and calculate P scores based
        on an empirical null distribution (ie, everything that
        TimeFreqSnPMResults can do).
        """
        super(TimeFreqSnPMClusters, self).__init__(
            vox_stat, vox_map, vox_stat_ranking,
            max_stat_dist, min_stat_dist
            )

        t, f = self.t.shape[1:]
        assert t*f == len(snpm_ptail_clusters) and \
               t*f == len(snpm_ntail_clusters), \
            'There is not enough distribution info for every time-freq point'

        self.all_ptail_clusters = snpm_ptail_clusters
        self.all_ntail_clusters = snpm_ntail_clusters  
    
    def get_scores_and_nulls(self): #stats_res, pos_clusters, neg_clusters):
        # want to gather the corrected Pt and Ps scores from each cluster,
        # at each permutation.

        # Then, calculate the combined score W for each cluster, at each perm.
        # Then, calculate the maximal W score for each permutation
        t, f = self.t.shape[1:]
        n_perm = float(self._max_t.shape[0])

        # -- Positive tail cluster scores ------------------
        pos_clusters = self.all_ptail_clusters
        # get the maximum cluster size for each permutation, for each (t,f)
        
        p_max_csize = [ [max(ci.size for ci in pclusts) if len(pclusts) else 0
                         for pclusts in clusts] for clusts in pos_clusters ]

        p_max_csize = np.array(p_max_csize, 'd')
        p_max_csize.sort(axis=1)
        max_t = np.sort(self._max_t.reshape(n_perm, t*f), axis=0).T

        p_cluster_stats = []
        p_true_label_score = []
        for tdist, sdist, permutations in zip(max_t, p_max_csize, pos_clusters):
            scores_by_permutation = []
            for n, clusters in enumerate(permutations):
                if not clusters:
                    if n==0:
                        p_true_label_score.append( [] )
                    scores_by_permutation.append(0)
                    continue
                size = np.array([ci.size for ci in clusters], 'd')
                Ps_scores = 1.0 - sdist.searchsorted(size, side='left')/n_perm
##                 Ps_scores = (1.0 + n_perm - su.index(size, sdist))/n_perm
                intensity = np.array([ci.peak for ci in clusters])
                Pt_scores = 1.0 - tdist.searchsorted(intensity, side='left')/n_perm
##                 Pt_scores = (1.0 + n_perm - su.index(intensity, tdist))/n_perm
                # This is the combining function!
                Wscore = -2*(np.log(Pt_scores) + np.log(Ps_scores))
                scores_by_permutation.append( Wscore.max() )
                if n==0:
                    p_true_label_score.append( Wscore )
            p_cluster_stats.append( sorted(scores_by_permutation) )

        # -- Negative tail cluster scores ------------------
        neg_clusters = self.all_ntail_clusters
        n_max_csize = [ [max(ci.size for ci in pclusts) if len(pclusts) else 0
                         for pclusts in clusts] for clusts in neg_clusters ]
        n_max_csize = np.array(n_max_csize, 'd')
        n_max_csize.sort(axis=1)
        min_t = np.sort(self._min_t.reshape(n_perm, t*f), axis=0).T

        n_cluster_stats = []
        n_true_label_score = []
        for tdist, sdist, permutations in zip(min_t, n_max_csize, neg_clusters):
            scores_by_permutation = []
            for n, clusters in enumerate(permutations):
                if not clusters:
                    if n==0:
                        n_true_label_score.append( [] )
                    scores_by_permutation.append(0)
                    continue
                size = np.array([ci.size for ci in clusters], 'd')
                Ps_scores = 1.0 - sdist.searchsorted(size, side='left')/n_perm
##                 Ps_scores = (1.0 + n_perm - su.index(size, sdist))/n_perm
                intensity = np.array([ci.peak for ci in clusters])
                Pt_scores = tdist.searchsorted(intensity, side='right')/n_perm
##                 Pt_scores = su.index(intensity, tdist).astype('d')/n_perm
                # This is the combining function!
                Wscore = -2*(np.log(Pt_scores) + np.log(Ps_scores))
                scores_by_permutation.append( Wscore.max() )
                if n==0:
                    n_true_label_score.append( Wscore )

            n_cluster_stats.append( sorted(scores_by_permutation) )
        return [ (p_cluster_stats, p_true_label_score),
                 (n_cluster_stats, n_true_label_score) ]

    def score_true_clusters(self):
        p_cluster_res, n_cluster_res = self.get_scores_and_nulls()

        pos_cluster_p = []
        dists_at_tf, scores_at_tf = p_cluster_res
        n_perm = float(len(dists_at_tf[0]))
        for dist, scores in zip(dists_at_tf, scores_at_tf):
            if not len(scores):
                pos_cluster_p.append([])
                continue
            # don't know if this should be side='right' or side='left'
            pvals = 1 - np.array(dist).searchsorted(scores, side='left')/n_perm
            pos_cluster_p.append( pvals )

        neg_cluster_p = []    
        dists_at_tf, scores_at_tf = n_cluster_res
        n_perm = float(len(dists_at_tf[0]))
        for dist, scores in zip(dists_at_tf, scores_at_tf):
            if not len(scores):
                neg_cluster_p.append([])
                continue
            # don't know if this should be side='right' or side='left'
            pvals = 1 - np.array(dist).searchsorted(scores, side='left')/n_perm
            neg_cluster_p.append( pvals )

        pos_clusters = [pc[0] for pc in self.all_ptail_clusters]
        neg_clusters = [pc[0] for pc in self.all_ntail_clusters]

        return [ (pos_clusters, pos_cluster_p),
                 (neg_clusters, neg_cluster_p) ]
    

def map_of_clusters_and_labels(stats, tail, tf, pcrit):
    g, m = calc_grid_and_map(stats.vox_idx)
    
    l_img = np.zeros(g)
    t_img = np.zeros(g)
    np.put(t_img, m, stats.t)
    t, f = tf
    tf_flat = f + t*stats.t.shape[-1]
    all_scores = stats.score_true_clusters()
    if tail.lower() == 'pos':
        print 'doing pos tail'
        clusters, scores = all_scores[0]
    else:
        print 'doing neg tail'
        clusters, scores = all_scores[1]
    if not clusters[tf_flat]:
        print 'no clusters!'
        return None, None
    elif not (scores[tf_flat] < pcrit).any():
        print 'no significant clusters!'
        return None, None
    else:
        tfclusters = clusters[tf_flat]
        idx = np.argwhere(scores[tf_flat] < pcrit).reshape(-1)
        l = 1
        for i in idx:
            c = tfclusters[i]
            _, mc = calc_grid_and_map(c.voxels.T, grid=g)
            np.put(l_img, mc, l)
            l += 1
        return t_img, l_img
    
def quick_plot_clusters(stats, tail, tf, pcrit):
    t_img, l_img = map_of_clusters_and_labels(stats, tail, tf, pcrit)
    if t_img is None:
        return
    import nipy.core.api as ni_api
    import xipy.slicing.image_slicers as islicers
    from xipy.vis import quick_plot_image_slicer
    import scipy.ndimage as ndimage
    ni_image = ni_api.Image(
        np.ma.masked_array(t_img, mask=np.logical_not(l_img.astype('B'))),
        ni_api.AffineTransform.from_params('ijk', islicers.xipy_ras, np.eye(4))
        )
    isl = islicers.ResampledIndexVolumeSlicer(ni_image)
    com = ndimage.center_of_mass(
        t_img, labels=l_img, index=range(1, int(l_img.max())+1)
        )
    if l_img.max() == 1:
        com = [com]
    for center in com:
        quick_plot_image_slicer(isl, center)
    
                                               
