__docformat__ = 'restructuredtext'
import os
import numpy as np
from nipy.core import api as ni_api

from xipy.utils import closest_voxel

from nutmeg.utils import array_pickler_mixin, parameterize_cmap, cmap_from_array
from nutmeg.external import descriptors as desc
from nutmeg.core import BEAM_SPACE_LEFT, BEAM_SPACE_POST, BEAM_SPACE_INF

legacy_argnames = {'sig': ('sig', 's')}
legacy_kwnames = {}

def search_any_pybeam(recarr, name):
    """This searches a beam record array file for a given field name. For
    the purpose of backwards compatibility, a list of legacy names is also
    consulted.
    
    Parameters
    ----------
    recarr : ndarray (record array)
        the unpickled beam
    name : str
        the name of the field being queried

    Returns
    -------
    the field value, or None
    """
    ndicts = (legacy_argnames, legacy_kwnames)
    for d in ndicts:
        names = d.get(name, (name,))
        for n in names:
            try:
                return recarr[n][0]
            except:
                pass
    # if failing all else, return this
    return None


class MEG_coreg(array_pickler_mixin):
    """A light class to keep track of MEG-to-MRI coregistration information
    """
    _argnames = ['mrpath', 'norm_mrpath', 'affine', 'fiducials']
    
    def __init__(self, mrpath, norm_mrpath, affine, fiducials):
        """
        Parameters
        ----------
        mrpath : str
          Path to coregistered MRI file
        norm_mrpath : str
          Path to coregistered normalized MRI file (???)
        affine : ndarray
          4x4 affine transformation matrix from MEG coords to MRI coords
        fiducials : ndarray
          3 3-vectors of fiducial locations (???)
        """
        self.mrpath = str(mrpath)
        self.norm_mrpath = str(norm_mrpath)
        if not affine.dtype.isbuiltin:
            aff = np.zeros(affine.shape, 'd')
            aff[:] = affine
            affine = aff
        self.affine = affine
        self.meg2mri = ni_api.AffineTransform.from_params(
            ni_api.ras_output_coordnames,
            ni_api.ras_output_coordnames,
            affine
            )
        self.fiducials = fiducials

    @staticmethod
    def from_mat_struct(m):
        """Create a MEG_coreg object from the coreg record array created
        by scipy.io when reading in a beam file
        """
        fields = m.dtype.names
        mrpath = m['mripath'][0] if 'mripath' in fields else ''
        norm_mr = m['norm_mripath'][0] if 'norm_mripath' in fields else ''
        tfm = m['meg2mri_tfm'] if 'meg2mri_tfm' in fields else np.eye(4)
        fid = m['fiducials_mri_mm'] \
              if 'fiducials_mri_mm' in fields else np.eye(4)
        return MEG_coreg(mrpath, norm_mr, tfm, fid)

class Beam(array_pickler_mixin):
    """The basic MEG Beam object for Nutmeg.
    """
    _argnames = ['voxelsize', 'voxels', 'srate', 'timepts', 'sig', 'coreg']
    _kwnames = ['coordmap']

    @staticmethod
    def _array_from_coordmap(cmap):
        return parameterize_cmap(cmap)

    @staticmethod
    def _reconstruct_coordmap(arr):
        return cmap_from_array(arr)
    
    def __init__(self, voxelsize, voxels, srate, timepts,
                 sig, coreg, coordmap=None):
        """
        Parameters
        ----------
        voxelsize : len-3 iterable
          the voxel edge lengths
        voxels : ndarray shaped (nvox, 3)
          the voxel coordinates, in this Beam's target coordinate space
        srate : float
          the sampling rate of the MEG time series
        timepts : ndarray
          the sample times
        sig : ndarray
          the MEG signal data
        coreg : MEG_coreg object
          the MEG-to-MRI coregistration info
        coordmap : NIPY AffineTransform object
          the MEG voxel index coordinate to voxel location coordinate mapping

        """
        self.voxelsize = np.asarray(voxelsize).astype('d')
        self.voxels = voxels.astype('d')
        self._vox_lookup = dict(((tuple(vx), n) for n, vx in enumerate(voxels)))
        msg = 'the number of signal points does not match the numner of voxels'
        assert sig.shape[0]==voxels.shape[0], msg
        self.sig = sig # so far ignorant of semantics of "sig"
        self.srate = srate
        self.timepts = timepts
        self.coreg = coreg
        if coordmap is None:
            v_offset = np.array([BEAM_SPACE_LEFT, BEAM_SPACE_POST,
                                 BEAM_SPACE_INF], 'd')
            coordmap = ni_api.AffineTransform.from_start_step(
                'ijk', ni_api.ras_output_coordnames,
                v_offset,
                self.voxelsize.astype('d')
                )
        self.coordmap = coordmap

    def vox_lookup_from_mr(self, vox):
        """Given a location in the coregistered MRI space, return
        the index into this Beam's voxels list.

        Parameters
        ----------
        vox : len-3 iterable
          the MRI space location coordinate

        Returns
        -------
        voxel list index
        """
        # get the MEG coordinates from the MRI space
        meg_vox = self.coreg.meg2mri.inverse()(vox)
        # get the corresponding data volume index coordinates
        vol_idx = np.round(self.coordmap.inverse()(meg_vox)).astype('i')
        all_idx = self.voxel_indices
        # want to find the floor(vol_idx) in all_idx
        a, dist = closest_voxel(all_idx, vol_idx)
        return -1 if dist > 0 else a

    @desc.auto_attr
    def voxel_indices(self):
        """Return the indices of this object's voxel locations on the 3D
        localization grid.
        """
        return np.round(self.coordmap.inverse()(self.voxels)).astype('i')
##         # Using the implicit floor operation on the array index coordinates
##         return self.coordmap.inverse(self.voxels).astype('i')


