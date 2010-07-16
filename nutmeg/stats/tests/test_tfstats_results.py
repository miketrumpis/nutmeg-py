import numpy as np
import scipy.stats as st
import nose.tools as nt

from nutmeg.utils import voxel_index_list, calc_grid_and_map
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

    yield nt.assert_false, (p_corr_neg.s[m] > alpha).any()

    alpha = 0.05
    while not (p_corr_pos.s <= alpha).any():
        alpha += 0.05
    print 'testing positive tail at significance level', alpha        

    tc, alpha = s_res.threshold(alpha, 'pos')
    m = tt.s >= tc
    yield nt.assert_false, (p_corr_pos.s[m] > alpha).any()

class testTFResults:

    @classmethod
    def setup_class(cls):
        cls.dist_sz, cls.nvox, cls.nt, cls.nf = 16, 100, 10, 4
        stat = np.random.randn(cls.nvox, cls.nt, cls.nf)
        rankings = np.random.rand(cls.nvox, cls.nt, cls.nf)
        max_dist = np.random.randn(cls.dist_sz, cls.nt, cls.nf)
        min_dist = np.random.randn(cls.dist_sz, cls.nt, cls.nf)
        vox = voxel_index_list((5,5,5))[:cls.nvox]
        cls.s_res = TimeFreqSnPMResults(
            stat, vox, rankings, max_dist, min_dist
            )

    def test_pooling(self):
        t = self.s_res._fix_dist('pos', (), (1,))
        pooled_shape = (self.dist_sz*self.nt, 1, self.nf)
        print t.shape
        assert t.shape == pooled_shape, 'unexpected pooled shape'

    def test_pooling_vals(self):
        t = self.s_res._fix_dist('pos', (), (1,))
        pooled_shape = (self.dist_sz*self.nt, 1, self.nf)
        mx_t = self.s_res._max_t.reshape(pooled_shape)
        sort_t = np.sort(mx_t, axis=0)[::-1]
        assert (sort_t == t).all(), 'unexpected values'

    def test_pooling2(self):
        t = self.s_res._fix_dist('pos', (), (2,))
        pooled_shape = (self.dist_sz*self.nf, self.nt, 1)
        print t.shape
        assert t.shape == pooled_shape, 'unexpected pooled shape'


    def test_correction1(self):
        t = self.s_res._fix_dist('pos', (1,), ())
        corr_shape = (self.dist_sz, 1, self.nf)
        assert t.shape == corr_shape, 'unexpected corrected shape'
        
    def test_correction2(self):
        t = self.s_res._fix_dist('neg', (1,), ())
        corr_shape = (self.dist_sz, 1, self.nf)
        ref_t = self.s_res._min_t.min(axis=1).reshape(corr_shape)
        ref_t = np.sort(ref_t, axis=0)

        assert (t == ref_t).all(), \
               'correction did not replace axis with maximal value'

    def test_correction3(self):
        t = self.s_res._fix_dist('neg', (1,2), ())
        corr_shape = (self.dist_sz, 1, 1)
        # minimize on axis 1 twice (since the 1st time consumes axis 1)
        ref_t = self.s_res._min_t.min(axis=1).min(axis=1).reshape(corr_shape)
        ref_t = np.sort(ref_t, axis=0)

        assert (t == ref_t).all(), \
               'correction did not replace axis with maximal value'


def cluster_sanity(sres):

    def clusters_intersect(c1, c2):
        s1 = set( c1.voxels )
        s2 = set( c2.voxels )
        return len(s1.intersection(s2)) > 0

    mn_pt = 1e10
    mx_nt = -1e10
    g, m = calc_grid_and_map(sres.vox_idx)
    img = np.zeros(g)
    for i, clist in enumerate((sres.ptail_clusters, sres.ntail_clusters)):
        for t in xrange(sres.t.shape[1]):
            for f in xrange(sres.t.shape[2]):
                np.put(img, m, sres.t[:,t,f])
                c_tf = clist[t][f]
                for c in c_tf:
                    cvals = np.take(img, c.voxels)
                    if i==1 and cvals.max() > mx_nt:
                        mx_nt = cvals.max()
                    if i==0 and cvals.min() < mn_pt:
                        mn_pt = cvals.min()
                    
                if len(c_tf) > 1:
                    for c1, c2 in zip(c_tf[:-1], c_tf[1:]):
                        assert not clusters_intersect(c1, c2), \
                               'Cluster intersection at tf=(%d,%d)'%(t,f)
    print 'estimated ntail cutoff: %1.3f, estimated ptail cutoff: %1.3f'%(mx_nt, mn_pt)
