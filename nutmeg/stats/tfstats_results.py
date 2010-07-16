import os
import numpy as np
from scipy import stats, ndimage

import xipy.volume_utils as vu

from nutmeg.core import tfbeam
from nutmeg.utils import array_pickler_mixin, calc_grid_and_map, tablify_list
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
            return array_pickler_mixin.load(snpm_arrays)
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
        ]

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
          The significance threshold (as in p < alpha)
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
    """
    This class analyzes the cluster information at each permutation
    from a SnPM-style test in order to score the clusters found at
    the truly labeled test.

    In addition, it can make thresholds and calculate P scores based
    on an empirical null distribution (ie, everything that
    TimeFreqSnPMResults can do).
    """

    _argnames = TimeFreqSnPMResults._argnames + \
                ['ptail_clusters', 'ptail_nulls',
                 'ntail_clusters', 'ntail_nulls']

    threshold_types = TimeFreqSnPMResults.threshold_types + \
                      ['SnPM Cluster scores (pos tail)',
                       'SnPM Cluster scores (neg tail)']

    @staticmethod
    def from_stats_and_clusters(stats_res,
                                pos_clusters, pos_nulls,
                                neg_clusters, neg_nulls):
        args = [ getattr(stats_res, a) for a in TimeFreqSnPMResults._argnames]
        args += [pos_clusters, pos_nulls, neg_clusters, neg_nulls]
        return TimeFreqSnPMClusters(*args)

    def __init__(self, vox_stat, vox_map, vox_stat_ranking,
                 max_stat_dist, min_stat_dist,
                 snpm_ptail_clusters, snpm_ptail_nulls,
                 snpm_ntail_clusters, snpm_ntail_nulls):
        """

        Parameters
        ----------

        All TimeFreqSnPMResults arguments, plus

        snpm_ptail_clusters: list
           List of ScoredStatClusters, for each (t,f) point
        snpm_ptail_nulls: ndarray
           Array of positive-tail null distributions for each (t,f) point
        snpm_ntail_clusters: list
           List of ScoredStatClusters, for each (t,f) point
        snpm_ntail_nulls: ndarray
           Array of negative-tail null distributions for each (t,f) point

        """
        super(TimeFreqSnPMClusters, self).__init__(
            vox_stat, vox_map, vox_stat_ranking,
            max_stat_dist, min_stat_dist
            )

        t, f = self.t.shape[1:]
        assert t*f==len(snpm_ptail_clusters) or \
               (t==len(snpm_ptail_clusters) and \
                f==len(snpm_ptail_clusters[0])), \
                'Not enough pos-tail clusters for every time-freq point'
        assert t*f==len(snpm_ntail_clusters) or \
               (t==len(snpm_ntail_clusters) and \
                f==len(snpm_ntail_clusters[0])), \
                'Not enough neg-tail clusters for every time-freq point'


        self.ptail_clusters = tablify_list(snpm_ptail_clusters, t, f)
        self.ptail_nulls = snpm_ptail_nulls
        self.ntail_clusters = tablify_list(snpm_ntail_clusters, t, f)
        self.ntail_nulls = snpm_ntail_nulls

    def pscore_clusters(self, tail, corrected_dims=(), pooled_dims=()):
        """
        Returns
        -------

        scores

        A list of p values for each list of clusters (at each (t,f) point)
        scored against the null
        
        """
        if len(corrected_dims) or len(pooled_dims):
            raise NotImplementedError(
                'Not yet combining time-freq dimensions for correction/pooling'
                )
        
        pvalues = []
        if tail.lower()=='pos':
            clusters = self.ptail_clusters
            nulls = np.sort(self.ptail_nulls, axis=0)
        else:
            clusters = self.ntail_clusters
            nulls = np.sort(self.ntail_nulls, axis=0)

        n_perm = float(nulls.shape[0])
        nt, nf = nulls.shape[1:]
        for t in xrange(nt):
            for f in xrange(nf):
                if not clusters[t][f]:
                    pvalues.append([])
                    continue
                ntf = nulls[:,t,f]
                wscores = [ci.wscore for ci in clusters[t][f]]
                pvalues.append(
                    1.0 - ntf.searchsorted(wscores, side='left')/n_perm
                    )
        return tablify_list(pvalues, nt, nf)

    def map_of_significant_clusters(self, tail, alpha,
                                    corrected_dims=(), pooled_dims=()):
        """Make a map based on a score that mixes cluster-size
        significance and peak cluster value significance. This score
        is then compared to an empirical null distribution generated
        by permutation testing of the samples, and a map is created
        across all (voxels, time, freq) where the cluster scores
        are deemed significant compared to `alpha`.

        Parameters
        ----------

        tail : str in {'pos', 'neg'}
           Return clusters where test statistic if significantly large
           or small, respectively
        alpha : float, p < alpha significance level

        Returns
        -------

        cluster_map : ndarray
           binary map which is non-negative at significant clusters

        Notes
        -----

        This method is based on
        `Combining voxel intensity and cluster extent with permutation 
        test framework`, Hayasaka and Nichols, 2004
        """
        pvals = self.pscore_clusters(tail)
        clusters = self.ptail_clusters if tail.lower()=='pos' \
                   else self.ntail_clusters
        g, m = calc_grid_and_map(self.vox_idx)
        cmap = np.zeros_like(self.t)
        img = np.zeros(g)
        for t in xrange(cmap.shape[1]):
            for f in xrange(cmap.shape[2]):
                scores = pvals[t][f]
                c_tf = clusters[t][f]
                for p, ci in zip(scores, c_tf):
                    # if the score beats alpha, then find the
                    if p < alpha:
                        # this is actually faster than looking up
                        # where ci.voxels intersect with map "m"
                        img.fill(0)
                        np.put(img, ci.voxels, 1)
                        cmap[:,t,f] += np.take(img, m)
        return cmap
        
def map_of_clusters_and_labels(stats, tail, tf, pcrit):
    """
    Generate a statistics map and a list of clusters based on the
    presence of clusters that exceed a critical p-value.

    Parameters
    ----------

    stats: a TimeFreqSnPMClusters object
    tail: str {'pos', 'neg'}
    tf: tuple (time,freq) index
    pcrit: float in (0.0,1.0)

    Returns
    -------

    stats_image, clusters

    The `stats_image` is the image of the statistical test values.
    `clusters` is a list of the significant clusters. If there
    are no cluster scores exceeding pcrit, then clusters returns as None.
    """
    g, m = calc_grid_and_map(stats.vox_idx)
    
    l_img = np.zeros(g)
    t_img = np.zeros(g)
    t, f = tf
    np.put(t_img, m, stats.t[:,t,f])
    scores = stats.pscore_clusters(tail)
    if tail.lower() == 'pos':
        print 'doing pos tail'
        clusters = stats.ptail_clusters
    else:
        print 'doing neg tail'
        clusters = stats.ntail_clusters
    if not clusters[t][f]:
        print 'no clusters!'
        return t_img, None, 0
    crit_scores = (scores[t][f] < pcrit)
    if not crit_scores.any():
        print 'no significant clusters!'
        return t_img, None, 0
    else:
        tfclusters = clusters[t][f]
        idx = np.argwhere(crit_scores).reshape(-1)
        critical_clusters = [clusters[t][f][i] for i in idx]
        return t_img, critical_clusters

def quick_plot_top_n(stats, tail, n=3):
    scores = stats.pscore_clusters(tail)
    clusters = stats.ptail_clusters if tail.lower()=='pos' \
               else stats.ntail_clusters

    # flatten score into a flat list of cluster scores
    flattened_scores = []
    # flatten clusters into a ntime x nfreq list of cluster sublists
    flattened_clusters = []
    for crow, srow in zip(clusters, scores):
        flattened_scores = np.r_[flattened_scores,
                                 reduce(lambda x, y: x+y,
                                        [ list(e) for e in srow ])]
        flattened_clusters += crow
    tf_map = []
    cluster_lookup = dict()
    nc = 0
    for i, cl in enumerate(flattened_clusters):
        nt, nf = stats.t.shape[1:]
        t, f = i / nf, i % nf
        tf_map += [ (t,f) ] * len(cl)
        cluster_lookup.update( dict( zip(xrange(nc, nc+len(cl)), cl) ) )
        nc += len(cl)

    n_idx = flattened_scores.argsort()[:n]
    g, m = calc_grid_and_map(stats.vox_idx)
    imgs = []
    tf_pts = []
    nclusters = []
    for i in n_idx:
        clst = cluster_lookup[i]
        tf_pts.append('(t,f) = %s, %d pts, p = %1.2f'%(str(tf_map[i]),
                                                       clst.size,
                                                       flattened_scores[i]))
        nclusters.append(cluster_lookup[i])
        t_img = np.zeros(g)
        t, f = tf_map[i]
        np.put(t_img, m, stats.t[:,t,f])
        imgs.append(t_img)
    quick_plot_clusters(imgs, nclusters, titles=tf_pts)

def quick_plot_clusters(imgs, clusters, titles=None):
    """

    Parameters
    ----------

    imgs: ndarray, or list
       The stats image for each cluster (may be one or many images)
    clusters: list of StatClusters
    titles: list of figure titles

    """
    import nipy.core.api as ni_api
    import xipy.slicing.image_slicers as islicers
    from xipy.vis import quick_plot_image_slicer
    import scipy.ndimage as ndimage
    from matplotlib.colors import Normalize

    if isinstance(imgs, np.ndarray):
        imgs_gen = (imgs for i in xrange(len(clusters)))
    else:
        imgs_gen = (i for i in imgs)
    if titles is None:
        titles = [''] * len(imgs)
    for img, cluster, tt in zip(imgs_gen, clusters, titles):
##         _, cvox = calc_grid_and_map(cluster.voxels.T, grid=img.shape)
        cvox = cluster.voxels
        l_img = np.zeros_like(img)
        np.put(l_img, cvox, 1)
        
        ni_image = ni_api.Image(
            np.ma.masked_array(img, mask=np.logical_not(l_img.astype('B'))),
            ni_api.AffineTransform.from_params(
                'ijk', islicers.xipy_ras, np.eye(4)
                )
            )
        isl = islicers.ResampledVolumeSlicer(ni_image)
        vxcom = ndimage.center_of_mass(img, labels=l_img, index=1)
        com = ni_image.coordmap(vxcom)
        norm = Normalize(img.min(), img.max())
        quick_plot_image_slicer(isl, com, title=tt, norm=norm)
    
def quick_plot_critical_clusters(stats, tail, tf, pcrit):
    t_img, clusters = map_of_clusters_and_labels(stats, tail, tf, pcrit)
    if clusters is None:
        return
    titles = titles=['time-freq = %s'%str(tf)]*len(clusters)
    quick_plot_clusters(t_img, clusters, titles=titles)
    
                                               
