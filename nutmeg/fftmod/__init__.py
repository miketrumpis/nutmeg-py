__docformat__ = 'restructuredtext'

fft_notes = \
"""
This module provides N-D FFTs for functions taken on the interval
n = [-N/2, ..., N/2-1] in all transformed directions. This is accomplished
quickly by making a change of variables in the DFT expression, leading to
multiplication of exp(+/-jPIk) * DFT{exp(+/-jPIn) * [n]}. Take notice that
BOTH your input and output arrays will be arranged on the negative-to-positive
interval. To take regular FFTs, shifting can be turned off.
"""
import numpy as np
from os.path import join, split, abspath
import os, sys
#______________________ Some convenience wrappers ___________________________

def fft1(a, shift=True, inplace=False, axis=-1):
    """
    Perform a forward FFT on a given axis

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axis to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axis : int, default=-1
      the data axis to transform

    Returns
    -------
    Transformed array if inplace==False
    """
    return _fftn(a, axes=(axis,), shift=shift, inplace=inplace)

def ifft1(a, shift=True, inplace=False, axis=-1):
    """
    Perform an inverse FFT on a given axis

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axis to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axis : int, default=-1
      the data axis to transform

    Returns
    -------
    Transformed array if inplace==False
    """

    return _ifftn(a, axes=(axis,), shift=shift, inplace=inplace)

def fft2(a, shift=True, inplace=False, axes=(-2,-1)):
    """
    Perform a forward FFT on two given axes

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axes to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axes : len-2 iterable of ints, default=(-2,-1)
      the data axess to transform

    Returns
    -------
    Transformed array if inplace==False
    """
    return _fftn(a, axes=axes, shift=shift, inplace=inplace)    

def ifft2(a, shift=True, inplace=False, axes=(-2,-1)):
    """
    Perform an inverse FFT on two given axes

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axes to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axes : len-2 iterable of ints, default=(-2,-1)
      the data axess to transform

    Returns
    -------
    Transformed array if inplace==False
    """
    return _ifftn(a, axes=axes, shift=shift, inplace=inplace)

def fftn(a, shift=True, inplace=False, axes=(-1)):
    """
    Perform a forward FFT on any given axes

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axes to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axes : len-n iterable of ints, default=(-1,)
      the data axess to transform

    Returns
    -------
    Transformed array if inplace==False
    """
    return _fftn(a, axes=axes, shift=shift, inplace=inplace)

def ifftn(a, shift=True, inplace=False, axes=(-1)):
    """
    Perform an inverse FFT on any given axes

    Parameters
    ----------
    a : ndarray
      the N-dimensional data
    shift : bool, default=True
      whether to consider the origin of FFT axes to be in the center
      or the beginning.
    inplace : bool, default=False
      whether to compute the FFT inplace
    axes : len-n iterable of ints, default=(-1,)
      the data axess to transform

    Returns
    -------
    Transformed array if inplace==False
    """
    return _ifftn(a, axes=axes, shift=shift, inplace=inplace)
#____________________________________________________________________________

try:
    import fft_ext    
    def _fftn(a, axes=(-1,), shift=1, inplace=0, fft_sign=-1):
        # integer-ize these parameters
        inplace = 1 if inplace else 0

        rank = len(a.shape)
        fname = '_fft_%s_%d'%(a.dtype.char, rank)
        try:
            ft_func = getattr(fft_ext, fname)
        except AttributeError:
            raise ValueError('no transform for this type and rank: %s, %d'%(a.dtype.char, rank))

        if inplace:
            # create a very small full rank array b to make ref counts happy
            full_rank = tuple( [1] * rank )
            b = np.array([1], dtype=a.dtype).reshape(full_rank)
        else:
            b = np.empty_like(a)

        if shift:
            dims = np.array( [a.shape[d] for d in axes] )
            if (dims%2).any():
                # Not set up to quickly apply a phase shift that is not
                # simply exp(iPI*n) = {-1,+1,-1,+1,...}
                a[:] = np.fft.fftshift(a, axes=axes)
                do_fftshift = True
                shift = 0
            else:
                do_fftshift = False
                shift = 1
        else:
            do_fftshift = False
            shift = 0

        adims = np.array(axes, dtype='i')
        ft_func(a, b, adims, fft_sign, shift, inplace)

        if do_fftshift:
            arr = a if inplace else b
            a[:] = np.fft.fftshift(a, axes=axes)
        
        if not inplace:
            if do_fftshift:
                b[:] = np.fft.fftshift(b, axes=axes)
            return b

except ImportError:
    def _fftn(a, axes=(-1,), shift=1, inplace=0, fft_sign=-1):
        fft_func = np.fft.fftn if fft_sign<0 else np.fft.ifftn
        op_arr = a if inplace else a.copy()
        if shift:
            for n, d in enumerate(a.shape):
                updown = 1 - 2*(np.arange(-d/2,d/2)%2)
                slices = [np.newaxis] * len(a.shape)
                slices[n] = slice(None)
                op_arr *= updown[slices]
        #else:
        #    b = a.copy()
        b = fft_func(op_arr, axes=axes)
        del op_arr
        if shift:
            for n, d in enumerate(a.shape):
                updown = 1 - 2*(np.arange(d)%2)
                slices = [np.newaxis] * len(a.shape)
                slices[n] = slice(None)
                b *= updown[slices]
        if inplace:
            a[:] = b
            del b
            return
        return b

def _ifftn(*args, **kwargs):
    kwargs['fft_sign'] = +1
    return _fftn(*args, **kwargs)
    
    
