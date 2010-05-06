import os
import numpy as np
import scipy.io as sio
from nutmeg.core import tfbeam
from _sutils import *

all_tfstats_maps = [
    'T test',
    'p val pos (corr)',
    'p val pos (uncorr)',
    'p val neg (corr)',
    'p val neg (uncorr)'
]

tfstats_names_to_mat_fields = dict(
    zip(all_tfstats_maps,
        ['T', 'p_corr_pos', 'p_uncorr_pos', 'p_corr_neg', 'p_uncorr_neg'])
    )

def split_combo_tfstats_file(fname):
    ext = os.path.splitext(fname)[-1]
    if ext=='.mat':
        return split_combo_tfstats_matfile(fname)
    else:
        raise ValueError("don't know how to split this file: %s"%fname)

def split_combo_tfstats_matfile(fname):
    tfb = sio.loadmat(fname, struct_as_record=True)['beam']
    stats_dict = {}
    if 'snpm' not in tfb[0,0].dtype.names:
        return stats_dict
    base_beam = tfbeam.TFBeam.from_mat_file(tfb)
    snpm = tfb[0,0]['snpm'][0,0]
    
    for stat in all_tfstats_maps:
        arr_name = tfstats_names_to_mat_fields[stat]
        if arr_name in snpm.dtype.names:
            new_beam = base_beam.from_new_dataset(snpm[arr_name],
                                                  fixed_comparison=stat)
            stats_dict[stat] = new_beam

    settings = snpm['settings'][0,0]
    subj = settings['subj'][0,0]
    dof = subj['number'][0,0]
    return stats_dict, base_beam, dof

def map_t_py(t, pvals, dp):
    """
    In a set M of maximal statistic values, we can define a cumulative
    distribution function as follows:
    
    CDF[t] --> n*(dp) := order{ tm | (tm in M) and tm < t } / order{M}

    Given quantized p values in the set { n*(dp) } for some n in [0,1/dp)
    and corresponding test values, his method attempts to roughly invert
    the relationship of the CDF.

    """
    nbins = int(1.0/dp)
    pbins = np.arange(nbins)*dp
    ranks = (pvals * nbins).astype('i').flatten()        
    si = np.argsort(ranks)
    s_ranks = np.take(ranks, si)
    # short circuit here if s_ranks are all the same
    if (s_ranks[1:]==s_ranks[0]).all():
        mn_t = t.min()
        edges = np.ones((nbins,)) * (mn_t-1)
        return edges, pbins

    ts = np.take(t, si)
    
    edges = np.empty((nbins,))
    ia = ib = 0
    ra = 0
    rb = s_ranks[0]
    if rb > 0:
        mn_t = ts.min()
        edges[:rb] = mn_t - 1    
    while rb < nbins:
        ra = s_ranks[ia]
        b = (s_ranks[ia:] > s_ranks[ia])
        if not b.any():
            break
        ib = ia + b.nonzero()[0][0]
        rb = s_ranks[ib]
        edges[ra:rb] = ts[ia:ib].max()
        ia = ib
    if rb < nbins:
        # this seems like an odd choice
        edges[ra:] = edges[ra-1] + 1
    return edges, pbins
        


