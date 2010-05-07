import numpy as np
from nutmeg.fftmod import fft1, ifft1, fft2, ifft2, _fftn, _ifftn
import warnings

# a cache of pre-computed 1D arrays from the separable nD gaussian kernel,
# parameterized by dim-length and sigma
__kernel_cache = {} 

def calc_fft_grid_and_map(beam):
    """Calculate the size of the volumetric grid in which to embed this
    beam's voxel data when taking an FFT. For a spatial FFT from
    nutmeg.fftmod, the spatial/frequency origin of the input and output
    arrays should be at the center of the volume to avoid the index
    re-ordering incurred by the FFT algorithm.

    Also return a map into the flattened volume.

    Parameters
    ----------
    beam : a Beam type

    Returns
    -------
    grid_shape : tuple
        a (2L, 2M, 2N) grid shape tuple
    flat_map : ndarray
        a list of indices into the flattened grid array, corresponding to
        the beam's voxel locations
    """
    import xipy.volume_utils as vu
    import nipy.core.api as ni_api
    # need to place voxel data in a volumetric grid,
    # with "discrete" symmetry about the center voxel --
    # ie, in any dimension with 2N points, N points
    # will be interpreted as being axis points less than 0,
    # and N-1 will be greater
    cmap = beam.coordmap
    vsize = beam.voxelsize
    limits = vu.world_limits(cmap, beam.voxel_indices.max(axis=0)+1)
    sym_index_lims = []
    for dv, lim in zip(vsize, limits):
        ineg, ipos = abs(int(lim[0]/dv)), abs(int(lim[1]/dv))
        if ipos > ineg:
            mx = ipos+1
        else:
            mx = ineg
        sym_index_lims.append( (-mx, mx-1) )

    new_affine = cmap.affine.copy()
    new_affine[:3,-1] = np.array(sym_index_lims)[:,0] * vsize
    new_cmap = ni_api.Affine.from_params(
        cmap.input_coords.coord_names,
        cmap.output_coords.coord_names,
        new_affine
        )
    grid_shape = tuple( [-mn*2 for mn, _ in sym_index_lims] )
    new_idc = new_cmap.inverse(beam.voxels).astype('i')

    # now find the flat map for these voxels
    ni, nj, nk = grid_shape
    strides = np.array( [nj*nk, nk, 1] )
    # this are the map for a given volume
    flat_map = (new_idc*strides).sum(axis=1)

    return grid_shape, flat_map

def gaussian_smooth(a, fwhm=5.0, axes=(-1,), inplace=True):
    """Smooth an array by convolving with a Guassian kernel.

    Parameters
    ----------
    a : ndarray
        Array to smooth
    fwhm : float or iterable (optional)
        The FWHM kernel size, or sizes for each dimension to smooth (pix units)
    axes : iterable (optional)
        The axes over which to convolve
    inplace : bool (optional)
        Modify the array inplace, or return a new, smoothed array

    Returns
    -------
    a smoothed array, if inplace==False
    """
    
    if hasattr(fwhm, '__iter__'):
        if len(fwhm) != len(axes):
            raise ValueError('Number of smoothing kernel sizes does not match the number of smoothing dimensions')

    else:
        fwhm = [fwhm]*len(axes)

    sigma = np.array(map(lambda x: x / np.sqrt(8 * np.log(2)), fwhm))

    a_real = a.dtype.char not in ['F', 'D']
    if a_real:
        if inplace:
            warnings.warn('inplace=True not really an option yet for real types')
        a = a.astype('D')
        # but now let's work inplace on the temporary casted array
        inplace = True

    if inplace:
        _fftn(a, axes=axes, inplace=True, shift=False)
        # name-only assignment
        b = a 
    else:
        # new array assignment
        b = _fftn(a, axes=axes, inplace=False, shift=False)
    
    null_slices = [np.newaxis] * len(a.shape)
    all_slices = [slice(None)] * len(a.shape)
    for n, s in enumerate(sigma):        
        d = 2 * (a.shape[axes[n]]/2)
        g = __kernel_cache.get((d,s), None)
        if g is None:
            print 'making new kernel for', d,s
            x = np.linspace(-d/2, d/2-1, d)
            x = np.r_[x[-d/2:], x[:d/2]]
            g = np.exp(-(x**2)/(2*s**2)).astype('D')
            fft1(g, shift=False, inplace=True)
            __kernel_cache[(d,s)] = g
        ax_slice = null_slices[:]
        ax_slice[axes[n]] = slice(None)
        safe_b_slice = all_slices[:]
        safe_b_slice[axes[n]] = slice(0,d)
        b[safe_b_slice] *= g[ax_slice].real
    n = len(sigma)
    b /= np.sqrt((2*np.pi)**n * np.prod(sigma**2))
    _ifftn(b, axes=axes, inplace=True, shift=False)

    if a_real:
        br = b.real
        del b
        return br
    if not inplace:
        return b

