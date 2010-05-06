import os
import numpy as np
from nipy.core import api as ni_api
from nutmeg.external import descriptors as desc
# want to be rid of these eventually
from nutmeg.core.coordinate_grids import BEAM_space, MNI_space

__docformat__ = 'restructuredtext'

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

## class array_pickler_mixin(object):
##     """ Can save itself as a funky ass record array
##     """
    
##     _argnames = []
##     _kwnames = []

##     dt = np.dtype( [(n, object) for n in _argnames+_kwargs] )

##     def save(self, fname):
##         np.save(fname, self._to_recarray())
    
##     def _to_recarray(self):
##         a = np.empty(1, dtype=self.dt)
##         names = self._argnames + self._kwnames
##         for n in names:
##             # XYZ: NEED TO MAKE OBJECTS THAT CAN ASARRAY THEMSELVES
##             a[name][0] = np.asarray(getattr(self, name, None))
## ##             if name=='coordmap':
## ##                 obj = parameterize_cmap(getattr(self, name))
## ##                 arr[name][0] = obj
## ##                 continue
##         return a

##     @classmethod
##     def from_array(klass, a, **kwargs):
##         if a.dtype != klass.dt:
##             raise ValueError('dtype of input array does not match')

## ##         args = (search_any_pybeam(rec_arr, name)
## ##                 for name in class_type._argnames)
## ##         kws = dict( ((name,search_any_pybeam(rec_arr, name))
## ##                      for name in class_type._kwnames) )
##         args = (a[n][0] for n in klass._argnames)
##         kws = dict( ((n, a[n][0]) for n in klass._kwnames) )

##         # this way, the user-defined keywords take precedence of whatever is
##         # found (or not found) in the record array
##         kws.update(kwargs)
## ##         if kws['coordmap'] is not None:
## ##             kws['coordmap'] = cmap_from_params(kws['coordmap'])

## ##         sig_type = search_any_pybeam(rec_arr, 'sig')[0].dtype
## ##         if len(sig_type) not in (2,3) and \
## ##                kws['fixed_comparison'] is None:
## ##             kws['fixed_comparison'] = 'unknown'
##         return class_type(*args, **kws)

class coreg(object):
    _argnames = ['mrpath', 'norm_mrpath', 'affine', 'fiducials']
    
    def __init__(self, mrpath, norm_mrpath, affine, fiducials):
        self.mrpath = str(mrpath)
        self.norm_mrpath = str(norm_mrpath)
        if not affine.dtype.isbuiltin:
            aff = np.zeros(affine.shape, 'd')
            aff[:] = affine
            affine = aff
        self.affine = affine
        self.meg2mri = ni_api.Affine.from_params('xyz', 'xyz', affine)
        self.fiducials = fiducials

    def wrap_up_as_array(self):
        dt = [ (n, object) for n in self._argnames ]
        a = np.empty(1, dtype=dt)
        for n in self._argnames:
            a[n][0] = getattr(self, n)
        return a

    @staticmethod
    def from_array(a):
        a = a.reshape(1)
        args = (a[n][0] for n in coreg._argnames)
        return coreg(*args)

    @staticmethod
    def from_mat_struct(m):
##         m = m.reshape(1)
        fields = m.dtype.names
        mrpath = m['mripath'][0] if 'mripath' in fields else ''
        norm_mr = m['norm_mripath'][0] if 'norm_mripath' in fields else ''
        tfm = m['meg2mri_tfm'] if 'meg2mri_tfm' in fields else np.eye(4)
        fid = m['fiducials_mri_mm'] \
              if 'fiducials_mri_mm' in fields else np.eye(4)
        return coreg(mrpath, norm_mr, tfm, fid)

class Beam(object):

    _argnames = ['voxelsize', 'voxels', 'srate', 'timepts', 'sig', 'coreg']
    _kwnames = ['coordmap'] 
    def __init__(self, voxelsize, voxels, srate, timepts,
                 sig, coreg, coordmap=None):
        self.voxelsize = voxelsize.astype('d')
        self.voxels = voxels.astype('d')
        self.vox_lookup = dict(((tuple(vx), n) for n, vx in enumerate(voxels)))
        msg = 'the number of signal points does not match the numner of voxels'
        assert sig.shape[0]==voxels.shape[0], msg
        self.sig = sig # so far ignorant of semantics of "sig"
        self.srate = srate
        self.timepts = timepts
        self.coreg = coreg # still just a bucket!
        if coordmap is None:
            v_offset = np.array([BEAM_space.left, BEAM_space.post,
                             BEAM_space.inf], 'd')
            coordmap = ni_api.Affine.from_start_step('ijk', 'xyz',
                                                     v_offset,
                                                     self.voxelsize.astype('f'))
        self.coordmap = coordmap

    def vox_lookup_from_mr(self, vox):
        meg_vox = self.coreg.meg2mri.inverse(vox)
        vol_idx = self.coordmap.inverse(meg_vox)
        all_idx = self.voxel_indices
        # want to find the floor(vol_idx) in all_idx
        dist = np.abs(( all_idx - np.floor(vol_idx) )).sum(axis=1)
        a = np.argwhere(dist==0)
        if a.any():
            return a[0,0]
        return -1

        
##         dist = ( (self.voxels - meg_vox)**2 ).sum(axis=-1)
##         vidx = np.argmin(dist)
##         dist = np.abs(meg_vox - self.voxels[vidx])
##         print 'looking up', vox, '-->', meg_vox, 'closest to', self.voxels[vidx], dist
##         if (dist > self.voxelsize).any():
## ##             raise ValueError('voxel not close to MEG grid')
##             return -1
##         return vidx
## ##         return np.argmin(((self.voxels-meg_vox)**2).sum(axis=0))

    def save(self, fpath):
        np.save(fpath, self._to_recarray())

    @classmethod
    def from_npy_file(class_type, npyfile, **kwargs):
        rec_arr = np.load(npyfile)
        if os.path.splitext(npyfile)[-1] == '.npz':
            # this was a npz file?
            rec_arr = rec_arr[rec_arr.files[0]]

        args = [search_any_pybeam(rec_arr, name)
                for name in class_type._argnames]
        # XYZ: BIG HACK!!!
        idx = class_type._argnames.index('coreg')
        c = args[idx]
        args[idx] = coreg.from_array(c)
        kws = dict( ((name,search_any_pybeam(rec_arr, name))
                     for name in class_type._kwnames) )

        # this way, the user-defined keywords take precedence of whatever is
        # found (or not found) in the record array
        kws.update(kwargs)
        if kws['coordmap'] is not None:
            kws['coordmap'] = cmap_from_params(kws['coordmap'])

##         sig_type = search_any_pybeam(rec_arr, 'sig')[0].dtype
##         if len(sig_type) not in (2,3) and \
##                kws['fixed_comparison'] is None:
##             kws['fixed_comparison'] = 'unknown'
        return class_type(*args, **kws)
        

    def _to_recarray(self):
        all_names = self._argnames + self._kwnames
        dt = [(name, object) for name in all_names]
        arr = np.empty(1, dtype=dt)
        for name in all_names:
            if name=='coordmap':
                obj = parameterize_cmap(getattr(self, name))
            elif name=='coreg':
                obj = self.coreg.wrap_up_as_array()
            else:
                obj = getattr(self, name, None)
            arr[name][0] = obj
        return arr

    @desc.auto_attr
    def voxel_indices(self):
        """Return the indices of this object's voxel locations on the 3D
        localization grid.
        """
        return np.round(self.coordmap.inverse(self.voxels)).astype('i')
##         # Using the implicit floor operation on the array index coordinates
##         return self.coordmap.inverse(self.voxels).astype('i')


def parameterize_cmap(coordmap):
    dt = [('incoord', object), ('outcoord', object), ('affine', object)]
    a = np.zeros(1, dtype=dt)
    a['incoord'][0] = coordmap.input_coords.coord_names
    a['outcoord'][0] = coordmap.output_coords.coord_names
    a['affine'][0] = coordmap.affine
    return a

def cmap_from_params(arr):
    return ni_api.Affine.from_params(arr['incoord'][0],
                                     arr['outcoord'][0],
                                     arr['affine'][0].astype('f'))

def signal_array_to_masked_vol(sig, vox_indices,
                               grid_shape=[],
                               prior_mask=None,
                               **ma_kw):
    """Make a 3D array representing a mask for valid voxels locations,
    given an array of voxel indices.

    Parameters
    ----------
    sig : ndarray
        nvox x [num_measures] array of signal measurements, whose spatial
        order is given by the corresponding (following) voxel array
    vox_indices : ndarray
        nvox x 3 array of voxel indices (NOT voxel locations in MNI space)
    grid_shape : list (optional)
        an list of dimension extents, eg [imax, jmax, kmax]
    prior_mask : array-like (optional)
        an nvox length array indicating points to mask in the final volume
        (True = unmasked, opposite of MaskedArray convention)
    ma_kw : dict
        Keyword arguments for np.ma.masked_array

    Returns
    -------
    s_masked : a numpy MaskedArray, with non-map voxels masked out
    """
    if not grid_shape:
        if not len(sig):
            return np.ma.masked_array(np.empty((1,1,1)),
                                      mask=np.ones((1,1,1), dtype=np.bool),
                                      **ma_kw)
        ix, jx, kx = map(lambda x: x+1, vox_indices.max(axis=0))
    else:
        ix, jx, kx = grid_shape

    vmask = np.ones((ix,jx,kx) + sig.shape[1:], np.bool)
    s = np.zeros((ix,jx,kx) + sig.shape[1:], sig.dtype)

    if prior_mask is not None:
        i, j, k = vox_indices[prior_mask].T
        sig = sig[prior_mask]
    else:
        i, j, k = vox_indices.T

    flat_idx = i*(jx*kx) + j*kx + k

    if sig.shape[1:]:
        # then we need to add more indices
        vx = np.product(sig.shape[1:])
        v = np.arange(vx)
        flat_idx *= vx
        flat_idx += v

    s.flat[flat_idx] = sig
    vmask.flat[flat_idx] = False
    return np.ma.masked_array(s, mask=vmask, **ma_kw)

## def signal_array_to_masked_vol(sig, vox_indices,
##                                grid_shape=[],
##                                prior_mask=None):
##     """Make a 3D array representing a mask for valid voxels locations,
##     given an array of voxel indices.

##     Parameters
##     ----------
##     sig : ndarray
##         nvox x [num_measures] array of signal measurements, whose spatial
##         order is given by the corresponding (following) voxel array
##     vox_indices : ndarray
##         nvox x 3 array of voxel indices (NOT voxel locations in MNI space)
##     grid_shape : list (optional)
##         an list of dimension extents, eg [imax, jmax, kmax]
##     prior_mask : array-like (optional)
##         an nvox length array indicating points to mask in the final volume

##     Returns
##     -------
##     s_masked : a numpy MaskedArray, with non-map voxels masked out
##     """
##     if prior_mask is None:
##         # cheap generator to unmask every point
##         prior_mask = (False for x in xrange(len(sig)))
##     elif prior_mask.shape != (sig.shape[0],):
##         raise ValueError('mask shape does not match signal shape')
##     if not grid_shape:
##         ix, jx, kx = map(lambda x: x+1, vox_indices.max(axis=0))
##     else:
##         ix, jx, kx = grid_shape
##     vmask = np.ones((ix,jx,kx) + sig.shape[1:], np.bool)
##     s = np.empty((ix,jx,kx) + sig.shape[1:], sig.dtype)

##     # iterating over 1st dimension of sig yields blocks of all other dims,
##     # so assignment of s[i,j,k] = sig[n] works for any number of signal dims
##     for ((i,j,k), scl, block) in zip(vox_indices, sig, prior_mask):
##         vmask[i,j,k] = block
##         s[i,j,k] = scl
##     s_masked = np.ma.masked_where(vmask, s)
##     return s_masked
