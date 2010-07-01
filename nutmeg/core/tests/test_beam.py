import numpy as np
import nose.tools as nt 
import nipy.core.api as ni_api

from xipy.utils import voxel_index_list
from nutmeg.core import beam


def gen_beam():
    # make a little offset between MEG and MRI space
    meg2mri_aff = np.eye(4)
    meg2mri_aff[:3,-1] = -5, -2.5, -2.5
    cr = beam.MEG_coreg('', '', meg2mri_aff, np.eye(3))

    # make a mapping from a 10x10x10 index grid to MEG space
    # -- each voxel edge is 2mm, and the origin is at ijk = 5,5,5
    idx2meg_aff = np.eye(4)*2
    idx2meg_aff[-1,-1] = 1
    idx2meg_aff[:3,-1] = -10,-10,-10
    cmap = ni_api.AffineTransform.from_params('ijk',
                                              beam.xipy_ras,
                                              idx2meg_aff)

    vx = cmap(voxel_index_list((10,10,10)))
    time_pts = np.arange(20)
    sig = np.random.rand(vx.shape[0], len(time_pts))
    b = beam.Beam([2.]*3, vx, 1000., time_pts, sig, cr, coordmap=cmap)
    return b

# all a Beam does now is lookup voxels

def test_beam_vox_lookup():
    b = gen_beam()
    # so.. the MRI coordinate (-5, -2.5, -2.5) maps to MEG coordinate (0, 0, 0)
    # which maps to MEG voxel index (5,5,5).. right??

    idx = b.vox_lookup_from_mr((-5,-2.5,-2.5))
    yield nt.assert_true, (b.voxels[idx]==np.zeros(3)).all(), \
          'coordinate mapping failed'
    yield nt.assert_true, (b.voxel_indices[idx]==np.array([5,5,5])).all(), \
          'index coordinate mapping failed'
