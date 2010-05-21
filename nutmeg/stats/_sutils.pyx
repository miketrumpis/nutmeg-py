""" -*- python -*- file
"""

# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def index(np.ndarray[np.npy_double, ndim=1] t, 
          np.ndarray[np.npy_double, ndim=1] dist):
    """Pidgeon-hole the values of t (along the 0th dimension) based on
    the (arranged) quantiles in dist.

    Notes
    -----
    It is assumed that the quantized values of dist are
    ordered from low to high

    Returns
    -------
    ti : ndarray
      the indices mapping t[i] <= dist[ti[i]],
      ti's range is [ 0, len(dist) )   
    """
    cdef np.ndarray si = np.argsort(t)
    cdef Py_ssize_t i, nt, nd
    nt = t.shape[0]
    nd = dist.shape[0]
    cdef np.ndarray[np.npy_uint32, ndim=1] ti = np.empty((nt,), dtype=np.uint32)
    i = 0
    j = 0
    # create a map from t -> i
    while i < nd:
        # index into t
        k = si[j]
        while t[k] <= dist[i]:
            ti[k] = i
            j += 1
            if j >= nt:
                return ti
            k = si[j]
        i += 1
    while j < nt:
        k = si[j]
        ti[k] = nd
        j += 1
    return ti

@cython.boundscheck(False)
def map_t(np.ndarray[np.npy_double, ndim=1] t,
          np.ndarray[np.npy_double, ndim=1] pvals,
          double dp):
    """
    In a set M of maximal statistic values, we can define a cumulative
    distribution function as follows:
    
    CDF[t] --> n*(dp) := order{ tm | (tm in M) and tm < t } / order{M}

    Given quantized p values in the set { n*(dp) } for some n in [0,1/dp)
    and corresponding test values, this method attempts to roughly invert
    the relationship of the CDF.

    Parameters
    ----------
    t : ndarray
      A set of test statistic values
    pvals : ndarray
      measures of (1-CDF[t])

    Returns
    -------
    (edges, pbins)
    the inverse relationship F{pbins} = edges
    """
    nbins = int(1.0/dp)
    cdef np.ndarray[np.npy_double, ndim=1] pbins = \
         np.linspace(0.0,dp*(nbins-1),nbins)
    cdef np.ndarray[np.npy_int32, ndim=1] ranks = \
         (pvals * nbins).astype('i')
    cdef np.ndarray[np.npy_int32, ndim=1] si = np.argsort(ranks)
    cdef np.ndarray[np.npy_int32, ndim=1] s_ranks = np.take(ranks, si)
    cdef np.ndarray[np.npy_double, ndim=1] edges
    # short circuit here if s_ranks are all the same
    if (s_ranks[1:]==s_ranks[0]).all():
        mn_t = t.min()
        edges = np.ones((nbins,), dtype='d') * (mn_t-1)
        return edges, pbins

    cdef np.ndarray[np.npy_double, ndim=1] ts = np.take(t, si)
    edges = np.empty((nbins,), dtype='d')
    cdef Py_ssize_t ia, ib, ra, rb, tmp, len_rank
    cdef double tval
    len_ranks = s_ranks.shape[0]
    ia = ib = 0
    ra = 0
    rb = s_ranks[0]
    if rb > 0:
        mn_t = ts.min()
        tmp = 0
        while tmp < rb:
            edges[tmp] = mn_t - 1
            tmp += 1
    while rb < nbins:
        ra = s_ranks[ia]
        tmp = ia + 1
        while tmp < len_ranks:
            if s_ranks[tmp] > s_ranks[ia]:
                ib = tmp
                break
            tmp += 1
        if ia==ib: # in other words, if ib did not get set in the last loop
            break
        # find max in interval ts[ia:ib]
        tval = ts[ia]
        # this loop effectively sets ia=ib at the end
        while ia < ib:
            if ts[ia] > tval:
                tval = ts[ia]
            ia += 1
        # set all edges between rank_a and rank_b to tval
        rb = s_ranks[ib]
        while ra < rb:
            edges[ra] = tval
            ra += 1
    if rb < nbins:
        # this seems like an odd choice
        tval = edges[ra-1] + 1
        while ra < nbins:
            edges[ra] = tval
            ra += 1
    return edges, pbins
        
