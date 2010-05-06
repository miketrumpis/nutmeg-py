import numpy as np
import scipy.stats as st
from nose.tools import assert_true, assert_equal, assert_false

import nutmeg.stats.stats_utils as su
from nutmeg.stats.tfstats_results import *

def test_map_t_real_data2():
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

    # create the adapted results
    s_res = adapt_mlab_tf_snpm_stats(pth)

    alpha = 0.05
    while not (p_corr_neg.s <= alpha).any():
        alpha += 0.05
    print 'testing negative tail at significance level', alpha

    tc, alpha = s_res.threshold(alpha, 'neg')
    m = tt.s <= tc

    yield assert_false, (p_corr_neg.s[m] > alpha).any()

    alpha = 0.05
    while not (p_corr_pos.s <= alpha).any():
        alpha += 0.05
    print 'testing positive tail at significance level', alpha        

    tc, alpha = s_res.threshold(alpha, 'pos')
    m = tt.s >= tc
    yield assert_false, (p_corr_pos.s[m] > alpha).any()
