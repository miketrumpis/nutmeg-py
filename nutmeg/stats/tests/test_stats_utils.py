import numpy as np
import scipy.stats as st
from nose.tools import assert_true, assert_equal, assert_false

import nutmeg.stats.stats_utils as su

def test_map_t():
    n = 2000
    
    p = np.random.randint(1, high=n, size=100).astype('d')
    # want to find some sampling is incomplete
    while(np.diff(p).max() < 5):
        p = np.random.randint(1, high=n, size=100).astype('d')
    p /= float(n)
    # test the neg tail, so reverse the sign of the isf function
    t = -st.norm.isf(p)
    edges, pvals = su.map_t(t, p, 1.0/n)
    # choose an alpha that is not exactly on a bin edge
    alpha = p[0] + .25/n
    k = int( alpha * n )
    thresh = edges[k]
    m = t <= thresh

    yield assert_true, p[m].max() < alpha, 'alpha mapped incorrectly'

    t *= -1
    edges *= -1
    k = int( alpha*n + .5)
    thresh = edges[k]
    m = t >= thresh

    yield assert_true, p[m].max() < alpha, 'alpha mapped incorrectly'

def test_index():
    quants = np.arange(1, 20).astype('d')
    # make values in range [0, 20)
    rvs = np.random.rand(10) * 20
    i = su.index(rvs, quants)
    yield (np.testing.assert_array_equal, i, np.floor(rvs))
    yield assert_true, i.min() >= 0, 'negative index calculated'
    yield assert_true, i.max() <= len(quants), 'too high index calculated'
    rvs = np.random.rand(100) * 20
    i = su.index(rvs, quants)
    yield (np.testing.assert_array_equal, i, np.floor(rvs))
    yield assert_true, i.min() >= 0, 'negative index calculated'
    yield assert_true, i.max() <= len(quants), 'too high index calculated'

    

def test_map_t_real_data():
    import os
    pth = os.path.join(os.path.dirname(__file__), 's_beamtf1_avg.mat')
    if not os.path.exists(pth):
        assert False, 'did not find data file'
        return
    mdict, mbeam, dof = su.split_combo_tfstats_matfile(pth)
    p_corr_pos = mdict['p val pos (corr)']
    p_corr_neg = mdict['p val neg (corr)']
    tt = mdict['T test']
    n = 2048
    nt, nf = tt.s.shape[1:]
    max_t_maps = np.empty((n, nt, nf))
    min_t_maps = np.empty((n, nt, nf))
    for t in xrange(nt):
        for f in xrange(nf):
            edges, _ = su.map_t(tt.s[:,t,f], p_corr_neg.s[:,t,f], 1.0/n)
            min_t_maps[:,t,f] = edges
            edges, _ = su.map_t(-tt.s[:,t,f], p_corr_pos.s[:,t,f], 1.0/n)
            max_t_maps[:,t,f] = -edges


    min_t_maps = np.sort(min_t_maps, axis=0)
    alpha = 0.05
    while not (p_corr_neg.s <= alpha).any():
        alpha += 0.05
    print 'testing negative tail at significance level', alpha
    # highest k index satisfying t <= tc
    k_mn = int(alpha * n)
    tc = min_t_maps[k_mn]
    m = tt.s <= tc
    yield assert_false, (p_corr_neg.s[m] > alpha).any()

    max_t_maps = np.sort(max_t_maps, axis=0)
    alpha = 0.05
    while not (p_corr_pos.s <= alpha).any():
        alpha += 0.05
    print 'testing positive tail at significance level', alpha        
    # lowest k index in max_t_maps satisfying significant t >= tc
    k_mx = int((1-alpha) * n + 0.5)
    tc = max_t_maps[k_mx]
    m = tt.s >= tc
    print m.sum()
    yield assert_false, (p_corr_pos.s[m] > alpha).any()

