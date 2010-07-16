import os
import numpy as np
import scipy.io as sio
from nutmeg.utils import array_pickler_mixin
from nutmeg.core import tfbeam
from _sutils import *

# a simple class for representing clusters
class StatCluster(array_pickler_mixin):
    "A simple 'bunch' and array pickler"
    _argnames = ['peak', 'mass', 'voxels']

    def __init__(self, *args):
        array_pickler_mixin.__init__(self, *args)
        self.size = self.voxels.size

    def __repr__(self):
        s = ''
        s += 'size:\t' + str(self.size) + '\n'
        for n in filter(lambda s: s!='voxels', self._argnames):
            s += (n + ':\t' + str(getattr(self, n)) + '\n')
        return s
    
class ScoredStatCluster(StatCluster):
    """
    A StatCluster that has a score based on a combination of its
    size, peak, and/or mass statistics
    """
    _argnames = StatCluster._argnames + ['wscore']

    @staticmethod
    def from_cluster(c, w):
        args = [getattr(c, n) for n in StatCluster._argnames] + [w]
        return ScoredStatCluster(*args)

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
        


