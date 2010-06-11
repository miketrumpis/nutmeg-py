import numpy as np
import scipy.linalg as la
from scipy import ndimage
from nipy.core import api

from nutmeg.numutils import gaussian_smooth
from nutmeg.core import full_beam_volume_shape

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
            self.p = p[randomized_rows]
        else:
            self.p = p
        self.n_obs = n_obs
        self.reset()

    def reset(self):
        self._generator = (r.reshape(self.n_obs, 1) for r in self.p)

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
            self.p = p[randomized_rows]
        else:
            self.p = p
        self.n_obs = n_obs
        self.reset()

    def reset(self):
        self._generator = (self._design_mat(r) for r in self.p)
        
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
            self.p = ps[randomized_rows]
        else:
            self.p = ps
        self.reset()

    def reset(self):
        self._generator = ( r[:,np.newaxis] * self.H for r in self.p )
    

def run_snpm_tests(samps, dgen, co, n_perm, grid_shape, flat_map, vox_size,
                   smoothed_variance=True, out=None):
    """
    Parameters
    ----------
    samps : ndarray
      (samples x voxels) array
    dgen : design matrix generator
    co : ndarray
      comparison weights for the solution to dgen(p)*b = samps
    n_perm : int
      number of permutations to actual test
    grid_shape : tuple
      volume shape in which to embed voxel data, for variance smoothing
    flat_map : 1D list
      flat voxel index into the volume
    vox_size : len-3 list
      edge lengths for the volume
    smoothed_variance : bool, optional
      do spatial variance smoothing
    out : ndarray, optional
      a (n_permutation x voxels) array to store the test distribution

    Returns
    -------
    test_distribution : ndarray
      A (n_permutation x voxels) shaped array of the test distribution.
      The "true" value test is always in the 0th row.
    """

    n_meas = samps.shape[0]
    n_vox = samps.shape[1]
    if out is None:
        pX = np.empty((n_perm, n_vox))
    else:
        assert out.shape == (n_perm, n_vox), 'output array has the wrong shape'
        pX = out

    if smoothed_variance:
        fwhm_pix = np.array([20.]*3) / np.asarray(vox_size)
        sigma = np.array(fwhm_pix / np.sqrt( 8 * np.log(2) ) )
        res_sm = np.empty((n_vox,), 'd')
        r_sm = np.empty(grid_shape, 'd')
        n_sm = np.empty(grid_shape, 'd')
        
    df = -1
    for p in xrange(n_perm):
        dm = dgen.next()
        if df < 0:
            df = n_meas - rank(dm)
        beta = np.dot(la.pinv(dm), samps)
        err = ((samps - np.dot(dm, beta))**2).sum(axis=0)

        if smoothed_variance:
            r_sm[:] = 0
            n_sm[:] = 0
            r_sm.flat[flat_map] = err
            n_sm.flat[flat_map] = 1.0
            ndimage.gaussian_filter(r_sm, sigma, mode='constant', output=r_sm)
            ndimage.gaussian_filter(n_sm, sigma, mode='constant', output=n_sm)
            res_sm = r_sm.flat[flat_map] / n_sm.flat[flat_map]
        else:
            res_sm = err

        pooling = np.dot(co, np.dot(la.pinv(np.dot(dm.T, dm)), co.T))
        res_sm *= np.squeeze((pooling/df))
        np.sqrt(res_sm, res_sm)
        pX[p] = np.dot(co, beta)
        pX[p] /= res_sm

    if out is None:
        return pX

# hide from Nose
run_snpm_tests.__test__ = False
