from scipy.special import gamma
from scipy import stats
from scipy.stats import distributions as dist
import numpy.testing as npt
import nose.tools as nt

from nutmeg.stats.snpm_testing import *

class Foo(ReGenerator):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._generator = ('a', 'b', 'c')

def test_regen():
    foo = Foo()
    a = [n for n in foo]
    foo.reset()
    a += [n for n in foo]
    assert a == ['a', 'b', 'c']*2, 'ReGenerator failed to reset'

def test_binary_pattern():
    # convert to 0s and 1s
    p = np.where(binary_pattern(3,4)==1, 0, 1)
    # count columns from right-to-left
    p = p[:,::-1]
    counts = np.zeros(p.shape[0])
    for i in xrange(p.shape[1]):
        counts += (2**i) * p[:,i]
    npt.assert_array_almost_equal(counts, np.arange(len(counts)))
    

def choose(a,b):
    a, b = map(int, (a,b))
    return int(gamma(a+1)/(gamma(a-b+1) * gamma(b+1)))

def test_unpaired_ttest_design():
    d = unpaired_ttest_design_generator(4, 4, half=False)
    h = d.next()
    ref1 = np.vstack( [ np.repeat([[1.0, 0.0]], 4, axis=0),
                        np.repeat([[0.0, 1.0]], 4, axis=0) ] )
    yield nt.assert_true, (h==ref1).all(), 'wrong design matrix in 1st slot'
    
    nchoices = d.n_tests
    n = choose(8, 4)
    yield nt.assert_true, n==nchoices, 'wrong # of permutations'
    yield nt.assert_true, (d._p.sum(axis=1)==0).all(), 'permutations mismatched'

    b = True
    ref = np.array([4,4])
    for h in d:
        b = b and (h.sum(axis=0) == ref).all()
    yield nt.assert_true, b, 'one or more design matrices mal-formed'

    d = unpaired_ttest_design_generator(4, 4, half=True)
    yield nt.assert_true, (d.n_tests == n/2), 'wrong # of sym. permuations'

def test_one_sample_ttest_generator():
    d = one_sample_ttest_design_generator(4, half=False)
    h = d.next()
    ref1 = np.array( [1]*4 )
    yield nt.assert_true, (h==ref1).all(), 'wrong design matrix in 1st slot'
    
    nperms = d.n_tests
    n = 2**4
    yield nt.assert_true, n==nperms, 'wrong # of permutations'

    d = one_sample_ttest_design_generator(4, half=True)
    yield nt.assert_true, (d.n_tests == n/2), 'wrong # of sym. permuations'
    
def test_1samp_statistic_image_from_voxels():
    samps = dist.norm.rvs(loc=3, size=(4,100))
    # test these samples against a null distribution with mean=0
    t_ref, _ = stats.ttest_1samp(samps, 0, axis=0)

    # stick these samples somewhere in a 10x10x10 image
    grid = (10,10,10)
    map = range(100,200)
    d_mat = np.ones(4).reshape(4,1)
    img = statistic_image_from_voxels(d_mat, samps, [1], grid, map,
                                      fill_value=1e20)
    yield nt.assert_true, np.allclose(t_ref, img.flat[map]), \
          'basic 1samp ttest differs'

    outside_map = range(100) + range(200,10*10*10)
    yield nt.assert_true, (img.flat[outside_map]==1e20).all(), \
          'untested voxels fill value is incorrect'

def test_1samp_statistic_image_from_voxels():
    samps1 = dist.norm.rvs(loc=3, size=(4,100))
    samps2 = dist.norm.rvs(loc=2, size=(3,100))
    t_ref, _ = stats.ttest_ind(samps1, samps2, axis=0)
    samps = np.vstack( (samps1, samps2) )
    design = unpaired_ttest_design_generator(4,3).next()
    grid = (10,10,10)
    map = range(100,200)
    stats_img = statistic_image_from_voxels(design, samps,
                                            np.array([1, -1]), grid, map)
    assert np.allclose(stats_img.flat[100:200], t_ref), \
           'basic 2samp Ttest differs'
    
