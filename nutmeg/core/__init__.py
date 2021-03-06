__docformat__ = 'restructuredtext'
import nutmeg
import os
import numpy as np
from xipy.utils import voxel_index_list
from xipy import TEMPLATE_MRI_PATH

# define some "globals"

BEAM_SPACE_LEFT = -90.; BEAM_SPACE_RIGHT = 90.
BEAM_SPACE_POST = -125.; BEAM_SPACE_ANT = 90.
BEAM_SPACE_INF = -70.; BEAM_SPACE_SUP = 105.

beam_mni_box = ( (BEAM_SPACE_LEFT, BEAM_SPACE_RIGHT),
                 (BEAM_SPACE_POST, BEAM_SPACE_ANT),
                 (BEAM_SPACE_INF, BEAM_SPACE_SUP) )

def full_beam_volume_shape(dr):
    nx = int((BEAM_SPACE_RIGHT - BEAM_SPACE_LEFT)/dr[0] + 1)
    ny = int((BEAM_SPACE_ANT - BEAM_SPACE_POST)/dr[1] + 1)
    nz = int((BEAM_SPACE_SUP - BEAM_SPACE_INF)/dr[2] + 1)
    return (nx, ny, nz)

def full_beam_coords(dr):
    dr = np.asarray(dr)
    vox = voxel_index_list(full_beam_volume_shape(dr))
    return (vox*dr+np.array([BEAM_SPACE_LEFT, BEAM_SPACE_POST, BEAM_SPACE_INF]))
