import os
import numpy as np
import scipy.io as sio
from nutmeg.utils import array_pickler_mixin
from nutmeg.core import tfbeam
from _sutils import *

# a simple class for representing clusters
class StatCluster(array_pickler_mixin):
    "A simple 'bunch' and array pickler"
    _argnames = ['size', 'peak', 'mass', 'voxels']


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
            new_beam = base_beam.from_new_data(snpm[arr_name],
                                               fixed_comparison=stat)
            stats_dict[stat] = new_beam

    settings = snpm['settings'][0,0]
    subj = settings['subj'][0,0]
    dof = subj['number'][0,0]
    return stats_dict, base_beam, dof
        


