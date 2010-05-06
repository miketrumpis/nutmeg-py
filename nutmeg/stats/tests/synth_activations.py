import numpy as np
import scipy as sp
from scipy.stats import distributions as dist
from nutmeg.stats import beam_stats as bs

def synthetic_gaussian_activity_like(beam, tf_pts=((0,0),),
                                     locs=((-10,-10,-10),),
                                     size=20, n_beams=1, pval=.98,
                                     mode='contrast', return_labels=False):
    sigma = np.std(beam.s)*2.5
    baseline_mean = 0
    dof = n_beams-1

    # want to find the appropriate mean-difference that gives a t score such
    # that 1 - cdf(t) = pval.. use standard deviation of the signal as an
    # estimate for sample standard dev.
    t_target = dist.t.isf(1-pval, dof)
    #t_target = 5
    # t_target = (x.mean() - mu) / ( sigma / sqrt(n) )
    active_mean = t_target * (sigma / dof**0.5) + baseline_mean
    print 'mean for active voxels:', active_mean
    print 'stdev:', sigma
    
    beams = []
    locs = np.array([l for l in locs])
    voxels = beam.voxels
    for n in xrange(n_beams):
        s_baseline = np.random.normal(loc=baseline_mean, scale=sigma,
                                      size=beam.s.shape)
        s_active = np.random.normal(loc=baseline_mean, scale=sigma,
                                    size=beam.s.shape)
        for vox in locs:
            # make an ellipsoid with random radii (centered about "size"),
            # centered at this location
            ellipse = dist.norm.rvs(loc=size, scale=.25*size, size=3)
            active_vox = ( ((voxels-vox)**2 / ellipse**2).sum(axis=-1) <= 1)
            n_active = active_vox.sum()
            print 'cluster of', n_active, 'vox about', vox
            for t,f in tf_pts:
                activity = np.random.normal(loc=active_mean, scale=.1*sigma,
                                            size=n_active)
                print 'active mean for this beam:', activity.mean()
                print 'baseline mean for same vox:', s_active[active_vox,t,f].mean()
                s_active[active_vox,t,f] += activity
##                 s_active[active_vox,t,f] = activity
        if mode=='contrast':
            beams = beams + [beam.from_new_dataset(s_active,
                                                   fixed_comparison='F dB'),
                             beam.from_new_dataset(s_baseline,
                                                   fixed_comparison='F dB')]
        else:
            # convert gaussian samples to 10 ** (sig/10) -- ??
            s_baseline /= 10.0
            s_active /= 10.0
            sig = np.array( (10 ** s_active, 10 ** s_baseline) )
            beams.append( beam.from_new_dataset(sig, uses='F dB') )
    if n_beams==1:
        return beams[0]
    if return_labels:
        if mode=='contrast':
            subj_labels = np.repeat(np.arange(1,n_beams+1), 2)
            cond_labels = np.array( [1, 2]*n_beams ).flatten()
        else:
            subj_labels = np.arange(1,n_beams+1)
            cond_labels = np.array([1]*n_beams)
        return beams, subj_labels, cond_labels
    return beams


## For an activation, simulate an active-to-control ratio that follows an
## F-distribution with dof-a and dof-b equal to 200. Then a ratio with
## 2% chance of being random would be computed by
## scipy.stats.distributions.f.isf(.02, 200, 200)...

## Not sure how to simulate the active cluster, could try:
##     multiplying random signal by the ratio
##     adding random signal with mean += mean_0*ratio
    
def synthetic_activation_like(beam, tf_pts=((0,0),), locs=((-10,-10,-10),),
                              size=20, n_beams=1, pval=.98):

    # at this ratio there's a 1-pval probability of the ratio occurring
    # by the null hypothesis
    sig_ratio = sp.stats.distributions.f.isf(1-pval, 200, 200)
    baseline_mean = 10 # keep it away from 0, since we want this to be power
    active_mean = 10 + 10*sig_ratio
    locs = np.array([l for l in locs])
    voxels = beam.voxels
    for n in xrange(n_beams):
        s_baseline = np.random.normal(loc=baseline_mean, size=beam.s.shape)
        s_active = np.random.normal(loc=0, scale=sigma, size=beam.s.shape)
        for vox in locs:
            active_vox = ( ((voxels-vox)**2).sum(axis=-1) <= size**2 )
            n_active = active_vox.sum()
            print 'cluster of', n_active, 'vox about', vox
            for t,f in tf_pts:
                activity = np.random.normal(loc=active_mean, scale=sigma,
                                            size=n_active)
                s_active[active_vox,t,f] += activity
        if mode=='contrast':
            beams.append( ( beam.from_new_dataset(s_baseline),
                            beam.from_new_dataset(s_active) ) )
        else:
            sig = np.array( (s_active, s_baseline) )
            beams.append( beam.from_new_dataset(sig) )
    if n_beams==1:
        return beams[0]
    return beams
    
