import warnings
import os
import numpy as np

def loads_extension_on_call(ext_name, namespc_dict):
    """This decorates a function which requires a C extension (named ext_name).
    If the extension module is not already loaded, this decorator will try to
    load it before ungating the function. If there is a problem loading the
    extension, then the function will be redefined to do nothing.
    """
    def dec(func):
        ext_loaded = namespc_dict.get(ext_name+'_loaded', False)
        if not ext_loaded:
            try:
                exec 'import '+ext_name in namespc_dict
                namespc_dict[ext_name+'_loaded'] = True
            except:
                warnings.warn("The C extension %s must be built before using this function."%ext_name)
                def new_func(*args, **kwargs):
                    raise ImportError("The function %s passes until the appropriate extension is installed"%func)
                    pass
                return new_func
        
        return func
    return dec

def voxel_index_list(shape, order='ijk'):
    """From an array shape, return a list of voxel index-coordinates

    Parameters
    ----------
    shape : tuple
        the array shape
    order : str, optional
        Indicates whether the coordinate ordering should go 'ijk', or 'kji'.
        In either case, the i coordinate varies fasted, followed by j, k, ...

    Returns
    -------
    an ( nvox x len(shape) ) array of voxel index coordinates
    """
    if order=='ijk':
        arr = np.indices(shape[::-1])[::-1]
    else:
        arr = np.indices(shape)
    return np.array( [a.flatten() for a in arr] ).transpose()

def coord_list_to_mgrid(coords, shape, order='ijk'):
    """From a voxel coordinates list, make a meshgrid array

    Parameters
    ----------
    coords : np.product(shape) x ndim ndarray
        the voxel coordinate list -- MUST have coordinates for every
        point in the volume specified by shape
    shape : tuple
        the volume's shape
    order : str, optional
        Indicates whether the coordinate ordering is 'ijk', or 'kji'.
        In either case, the i coordinate varies fasted, followed by j, k, ...

    Returns
    -------
    a meshgrid representation of the coordinates, shaped (ndim, ni, nj, nk, ...)
    or (ndim, [...], nk, nj, ni)
    """
    ncoords = coords.shape[0]
    if np.product(shape) != ncoords:
        raise ValueError(
"""A fully specified coordinate list must be provided"""
    )
    c_t = coords.transpose()
    if order=='ijk':
        return np.array( [a.reshape(shape[::-1]).transpose() for a in c_t] )
    else:
        return np.array( [a.reshape(shape) for a in c_t] )
