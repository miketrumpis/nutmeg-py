import numpy as np
import scipy.linalg as la
from scipy import ndimage
from nipy.core import api

from nutmeg.numutils import gaussian_smooth
from nutmeg.core import full_beam_volume_shape
from nutmeg.stats.stats_utils import StatCluster

def rank(a):
    svals = la.svdvals(a)
    tol = np.finfo(a.dtype).eps * np.array(a.shape).max()
    return (svals>tol).sum()

def binary_pattern(n, ncol=None):
    "Returns the first 2**n rows of a binary counting pattern"
    if ncol is None:
        ncol = n
    p = np.ones((2**n, ncol))
    switch_freq = 2**(n-1)
    for i in xrange(ncol-n, ncol):
        col = p[:,i]
        col.shape = (col.shape[0]/switch_freq, switch_freq)
        col[1::2] *= -1
        switch_freq /= 2
    return p

class ReGenerator(object):
    """
    Acts like a generator, but can be reset to the initial state
    """
    def __init__(self):
        self._generator = (f for f in [])

    def next(self):
        return self._generator.next()

    def __iter__(self):
        return iter(self._generator)

    def reset(self):
        pass

    def close(self):
        self._generator.close()

    def send(self):
        self._generator.send()
    

class one_sample_ttest_design_generator(ReGenerator):
    """
    The 1-sample T test design matrix for N observations is simply

    >>> H = np.array([ [1.0] * N ]).T

    such that H*[a] = X
    where X is [ [vox_samples(subj=0)],
                 [vox_samples(subj=1)],
                 [...] ] for all subjects
                 
    This generator yields all combinations of matrices that are re-weighted
    such that the polarity of rows are flipped. The number of toggled
    versions of H is 2**N
    """
    def __init__(self, n_obs, half=True, random_order=True):
        """
        Parameters
        ----------
        n_obs : int
          Order of the sample
        half : bool, optional
          Only compute half of the polarity flips (by symmetry, 1/2 of
          the matrices have the inverse polarity of the other half)
        random_order: bool, optional
          After the first ("true") design matrix, yield reweighted
          matrices in random order
        """
        n = n_obs-1 if half else n_obs
        p = binary_pattern(n, n_obs)
        n_rows = len(p)
        if random_order:
            # randomized the row access after row 0
            randomized_rows = np.arange(1,n_rows)
            np.random.shuffle(randomized_rows)
            randomized_rows = np.insert(randomized_rows, 0, 0)
            self._p = p[randomized_rows]
        else:
            self._p = p
        self.n_obs = n_obs
        self.dof = n_obs - 1
        self.n_tests = self._p.shape[0]
        self.reset()

    def reset(self):
        self._generator = (r.reshape(self.n_obs, 1) for r in self._p)

class unpaired_ttest_design_generator(ReGenerator):
    """
    An unpaired T-test design matrix operates as H*[alpha,beta]^T = X,
    where X is [ [vox_samples(subj=0,cond=0)],
                 [vox_samples(subj=1,cond=0)],
                 [...],
                 [vox_samples(subj=0,cond=1)],
                 [vox_samples(subj=1,cond=1)],
                 [...] ] for subjects, and conditions 0 and 1
    and H is
    >>> H = np.vstack( [ np.repeat([[1.0, 0.0]], n_obs_a, axis=0),
                         np.repeat([[0.0, 1.0]], n_obs_b, axis=0) ] )

    Each permutation of H will toggle the labels for a subset of subjects
    based on a binary counter patterned selection
    (counting from 0 to 2**(n_subjects-1)).

    The final set of permutations will then be those bit strings where the
    number of 0s is equal to n_obs_a, and the number of 1s is equal to n_obs_b.

    Furthermore, if the n_obs_a == n_obs_b, then half of the final set is
    a bit-flipped mirror of the other half. One half of the set can be
    excluded by further contraining the bit strings to begin with 1
    """
    def __init__(self, n_obs_a, n_obs_b, half=True, random_order=True):
        if n_obs_a != n_obs_b:
            half = False
        n_obs = n_obs_a + n_obs_b
        n = n_obs - 1 if half else n_obs
        p = binary_pattern(n, n_obs)
        # only keep rows where the number of flipped observations is
        # equal to n_obs_b
        good_rows = (p.sum(axis=-1) == (n_obs_a - n_obs_b))
        p = p[good_rows]
        if half:
            good_rows = (p[:,0] == 1)
            p = p[good_rows]
        if random_order:
            # randomized the row access after row 0
            randomized_rows = np.arange(1,p.shape[0])
            np.random.shuffle(randomized_rows)
            randomized_rows = np.insert(randomized_rows, 0, 0)
            self._p = p[randomized_rows]
        else:
            self._p = p
        self.n_obs = n_obs
        self.dof = n_obs - 2
        self.n_tests = self._p.shape[0]
        self.reset()

    def reset(self):
        self._generator = (self._design_mat(r) for r in self._p)
        
    def _design_mat(self, pattern):
        m = np.zeros((2, self.n_obs))
        m_alpha = m[0]; m_beta = m[1]
        ra = pattern == +1
        rb = pattern == -1
        m_alpha[ra] = 1
        m_beta[rb] = 1
        return m.T
    
    pass

class anova_design_generator(ReGenerator):
    """
    An ANOVA design matrix is H*[a,b]^T = X,
    where X is [ [tser(i=0,j=0)],
                 [tser(i=0,j=1)],
                 [...], ...] for subjects{i}, conditions{j}
    and H is [ eye(n_conditions) ] * n_subjects

    Each permutation of H will toggle the labels for a subset of subjects
    based on a binary counter patterned selection
    (counting from 0 to 2**(n_subjects-1))
    """
    def __init__(self, n_subj, n_cond, random_order=True):
        self.H = np.array(list(np.eye(n_cond))*n_subj)
        n_perm = 2**(n_subj-1)
        # create 2**(n_subj-1) rows, each with n_subj columns,
        # which indicate the bit pattern of the row number n
        ps = np.array([np.array([c for c in np.binary_repr(n,width=n_subj)],
                                dtype=np.int8)
                       for n in xrange(n_perm)])
        # convert to +1 / -1 pattern
        ps = 1 - 2*ps
        # repeat the columns by the number of contitions
        ps = np.repeat(ps, n_cond, axis=1)
        if random_order:
            # randomized the row access after row 0
            randomized_rows = np.arange(1,n_perm)
            np.random.shuffle(randomized_rows)
            randomized_rows = np.insert(randomized_rows, 0, 0)
            self._p = ps[randomized_rows]
        else:
            self._p = ps
        self.reset()

    def reset(self):
        self._generator = ( r[:,np.newaxis] * self.H for r in self._p )
    

def run_snpm_tests(samps, dgen, beta_comp, n_tests,
                   symmetry=False,
                   smooth_variance=True,
                   analyze_clusters=False,
                   # image embedding -- for clusters OR variance smoothing
                   grid_shape=None,
                   vox_map=None,
                   # variance smoothing info
                   fwhm_pix=None,
                   # cluster analysis info
                   t_crit=None,
                   fill_value=0,
                   out=None):
    """
    Parameters
    ----------
    samps : ndarray
      (samples x nvoxels) array
    dgen : design matrix generator
    beta_comp : ndarray
      comparison weight(s) for the solution b, in dgen(p)*b = samps
    n_tests : int
      number of reweighted designs to test
    symmetric : {True/False}, optional
      If the test combinations provided by dgen represent one symmetrical
      half of the possible combinations, then setting this flag will fill
      in the results from the other half of the distribution.

    smooth_variance : {True/False}, optional
      Optionally do spatial smoothing of the variance image.
      **Note**: if True, then *grid_shape*, *flat_map*, and *fwhm_pix*
      are all required parameters.

    analyze_clusters: {True/False}, optional
      Optionally gather additional cluster statistics at each permutation.
      **Note**: if True, then *grid_shape*, *flat_map*, and *t_crit*
      are all required, and *fill_value* should be provided

    grid_shape : tuple, optional
      volume shape in which to embed voxel data
    vox_map : 1D list, optional
      The mapping from tested voxels to indices in the flattened image grid
    fwhm_pix : len-3 list, optional
      edge lengths for the volume, in units of pixels
    smoothed_variance : bool, optional
      do spatial variance smoothing
    t_crit : float, optional
      The `cluster-defining threshold`. This is a statistic drawn from a
      theoretical (parametric) distribution.
    out : ndarray, optional
      a (n_permutation x voxels) array to store the
      empirical null distribution

    Returns
    -------
    test_distribution : ndarray
      A (n_permutation x voxels) shaped array of the emperical null
      distribution. The `true` value test is always in the 0th row.

    clusters : list
      If analyze_clusters is on, then this function also returns a list
      of cluster information for each permutation.
    """

    n_meas = samps.shape[0]
    n_vox = samps.shape[1]
    total_perm = n_tests*2 if symmetry else n_tests
    if out is None:
        pX = np.empty((total_perm, n_vox))
    else:
        assert out.shape == (total_perm, n_vox), \
               'output array has the wrong shape'
        pX = out

    # check the conditions specified against the arguments provided
    spatial_satisfied = grid_shape is not None and vox_map is not None
    smth_var_satisfied = fwhm_pix is not None
    cluster_satisfied = t_crit is not None

    if (analyze_clusters or smooth_variance):
        if not spatial_satisfied:
            raise ValueError(
                'No spatial mapping information provided, even though at ' \
                'least one spatially based method was requested'
                )
        stat_img = np.empty(grid_shape, 'd')
    if smooth_variance:
        if not smth_var_satisfied:
            raise ValueError(
                'FWHM parameters not provided, even though variance '\
                'smoothing was requested'
                )
        sigma_pix = np.asarray(fwhm_pix) / np.sqrt(8 * np.log(2))
    if analyze_clusters:
        if not cluster_satisfied:
            raise ValueError(
                'No cluster threshold provided, even though cluster '\
                'analysis was requestd'
                )
        ptail_clusters_list = []
        ntail_clusters_list = []
        
    for p in xrange(n_tests):
        design = dgen.next()
        # find estimate of beta, and then the error
        # of ||Y-Yhat||**2
        beta = np.dot(la.pinv(design), samps)
        err = np.dot(design, beta)
        np.subtract(samps, err, err)
        np.power(err, 2, err)
        ss_err = err.sum(axis=0)
 
        if smooth_variance:
            stat_img.fill(0)
            np.put(stat_img, vox_map, ss_err)
##             stat_img.flat[vox_map] = ss_err
            ndimage.gaussian_filter(
                stat_img, sigma_pix, mode='constant', output=stat_img
                )
            res_sm = np.take(stat_img, vox_map)
##             res_sm = stat_img.flat[vox_map]
            stat_img.fill(0)
            np.put(stat_img, vox_map, 1)
##             stat_img.flat[vox_map] = 1
            ndimage.gaussian_filter(
                stat_img, sigma_pix, mode='constant', output=stat_img
                )
            res_sm /= np.take(stat_img, vox_map)
##             res_sm /= stat_img.flat[vox_map]
        else:
            res_sm = ss_err

        # pooling : variance pooling -- c*[(X^T*X)^-1]*c^T
        pooling = np.dot(
            beta_comp,
            np.dot(la.pinv(np.dot(design.T, design)), beta_comp.T)
            )
        # res_sm := variance pooling * sigma^2
        res_sm *= np.squeeze((pooling/dgen.dof))
        np.sqrt(res_sm, res_sm)
        pX[p] = np.dot(beta_comp, beta)
        pX[p] /= res_sm

        if analyze_clusters:
            stat_img.fill(fill_value)
            np.put(stat_img, vox_map, pX[p])
##             stat_img.flat[vox_map] = pX[p]
            _, pcluster_info = label_clusters(stat_img, t_crit, tail='pos')
            _, ncluster_info = label_clusters(stat_img, -t_crit, tail='neg')
            ptail_clusters_list.append(pcluster_info)
            ntail_clusters_list.append(ncluster_info)
            if symmetry:
                # if only doing one half of symmetrical permutations,
                # then duplicate some information for free (just invert
                # the sign on the maximum cluster intensity)
                ptail_clusters_list.append(
                    [ StatCluster(
                        c.size, -c.peak, c.mass, c.voxels
                        ) for c in ncluster_info ]
                    )
                ntail_clusters_list.append(
                    [ StatCluster(
                        c.size, -c.peak, c.mass, c.voxels
                        ) for c in pcluster_info ]
                    )
    # now if only doing one half of symmetrical permutations, then
    # record the sign inverted statistics for the other half
    if symmetry:
        t = pX[:n_tests]
        pX[n_tests:] = -t
    if analyze_clusters:
        return pX, (ptail_clusters_list, ntail_clusters_list)
    else:
        return pX

# hide from Nose
run_snpm_tests.__test__ = False

def statistic_image_from_voxels(design, samps, beta_comp,
                                img_grid, vox_map,
                                fill_value=0,
                                variance_img=False,
                                fwhm_pix=None):
    """
    Calculate a statistics image from a (n x nvox) matrix of
    samples, using the design matrix and a vector of weights to combine
    the beta values.

    Parameters
    ----------
    design : ndarray, (n x p)
      The design matrix for the test
    samps : ndarray, (n x nvox)
      The samples for each of n tests at each of nvox voxels
    beta_comp : ndarray, (1 x p)
      Weights for combining the beta values
    img_grid : array shape (tuple), or 3D ndarray
      Either the array in which to store the stats image, or the shape of
      such an array
    vox_map : sequence
      The mapping from tested voxels to indices in the flattened image grid
    fill_value : float
      What value to fill in for the non tested voxels in the image
    variance_img : {True/False}, optional
      Whether or not to compute a smoothed variance image. If an array
      is provided for the image grid, that same array will be used as
      temporary storage for the variance smoothing, before having the
      statistics stored in it.
      **Note**: if computing a variance image, then fwhm_pix MUST be
      specified. 
    fwhm_pix : sequence, optional
      The FWHM of the smoothing kernel, in units of pixels, for each
      dimension of the variance image

    Returns
    -------
      The statistics image
    """

    if not isinstance(img_grid, np.ndarray):
        img_grid = np.empty(img_grid, samps.dtype)

    n_meas = samps.shape[0]
    df = n_meas - design.shape[1]
    beta_comp = np.asarray(beta_comp)

    # find estimate of beta, and then the error
    # of ||Y-Yhat||**2
    beta = np.dot(la.pinv(design), samps)
    err = np.dot(design, beta)
    np.subtract(samps, err, err)
    np.power(err, 2, err)
    ss_err = err.sum(axis=0)

##     err = ((samps - np.dot(design, beta))**2).sum(axis=0)

    if variance_img:
        if fwhm_pix is None:
            raise ValueError(
                'No smoothing parameters provided, even though variance '\
                'smoothing was requested'
                )
        img_grid.fill(0)
        sigma_pix = np.asarray(fwhm_pix) / np.sqrt(8 * np.log(2))
        img_grid.flat[vox_map] = ss_err
        ndimage.gaussian_filter(
            img_grid, sigma_pix, mode='constant', output=img_grid
            )
        res_sm = img_grid.flat[vox_map]
        img_grid.fill(0)
        img_grid.flat[vox_map] = 1
        ndimage.gaussian_filter(
            img_grid, sigma_pix, mode='constant', output=img_grid
            )
        res_sm /= img_grid.flat[vox_map]
        img_grid.fill(fill_value)
    else:
        res_sm = ss_err
        img_grid.fill(fill_value)
        
    # pooling : variance pooling -- c*[(X^T*X)^-1]*c^T
    pooling = np.dot(
        beta_comp,
        np.dot(la.pinv(np.dot(design.T, design)), beta_comp.T)
        )
    # res_sm := variance pooling * sigma^2
    res_sm *= np.squeeze((pooling/df))
    np.sqrt(res_sm, res_sm)
    stat = np.dot(beta_comp, beta)
    stat /= res_sm
    img_grid.flat[vox_map] = stat
    return img_grid

def label_clusters(stats_image, t_crit, connectivity=18, tail='pos'):
    """

    Returns a list of cluster labels, an image of clustered labels,
    as well as a dictionary of cluster information, keyed by the labels.

    dictionary 

    """

    connectivity_arg = {6:1, 18:2, 26:3}.get(connectivity,2)
    
    m = ndimage.generate_binary_structure(3, connectivity_arg)
    if tail.lower() == 'pos':
        l_image, max_label = ndimage.label(stats_image > t_crit, structure=m)
    else:
        l_image, max_label = ndimage.label(stats_image < t_crit, structure=m)
    csizes = []
    max_t = []
    t_mass = []
    c_idx = [] # this is a (3 x len-cluster) array of ijk indices
    labels = xrange(1, max_label+1)
    for l in labels:
        c_image = l_image==l
        csizes.append( c_image.sum() )
        c_idx.append( c_image.ravel().nonzero()[0] )
##         c_idx.append( np.vstack(c_image.nonzero()) )

        cluster_stats = stats_image[c_image]
        if tail.lower()=='pos':
            max_t.append( cluster_stats.max() )
            t_mass.append( cluster_stats.sum() - len(cluster_stats)*t_crit )
        else:
            max_t.append( cluster_stats.min() )
            t_mass.append( -cluster_stats.sum() + len(cluster_stats)*t_crit )

    clusters = [StatCluster(*c) for c in zip(csizes, max_t, t_mass, c_idx)]
    return l_image, clusters
    
    
def score_clusters(ext_t, permutation_clusters):
    """

    Parameters
    ----------

    ext_t: ndarray
       The extreme statistics null distribution for this permutation test
    permutation_clusters: list
       List of StatClusters at each permutation for this  test. The
       clusters of the true labeling must be in the first sub-list.

    Returns
    -------

    scores, null
       The array of scores is based on a `combining function` for each
       cluster in the true labeling. The  empirical null distribution
       is generated by the max score at each permutation.
    """
    
    p_max_csize = [max(ci.size for ci in clusts) if len(clusts) else 0
                   for clusts in permutation_clusters]

    p_max_csize = np.sort(p_max_csize).astype('d')
    ext_t = np.sort(ext_t)
    n_perm = float(len(ext_t))

    # Generate the empirical null of W (the combined cluster score)
    cluster_stats = np.empty(n_perm)
    for n, clusters in enumerate(permutation_clusters):
        if not clusters:
            cluster_stats[n] = 0
            if n==0:
                true_score = [0]
            continue
        # find the quantiles for these cluster sizes in the max sizes        
        size = np.array([ci.size for ci in clusters], 'd')
        Ps_scores = 1.0 - p_max_csize.searchsorted(size, side='left')/n_perm
        # find the quantiles for these peak-stats in the max stats
        intensity = np.array([ci.peak for ci in clusters])
        Pt_scores = 1.0 - ext_t.searchsorted(intensity, side='left')/n_perm
        
        # This is the combining function! (should be a variable)
        Wscore = -2*(np.log(Pt_scores) + np.log(Ps_scores))
        cluster_stats[n] = Wscore.max()
        if n==0:
            true_score = Wscore

    return true_score, cluster_stats
