from nipy.core import api as ni_api
from nipy.core.reference.coordinate_map import compose
import numpy as np
import os
import scipy.io as sio
import warnings as w

from xipy.volume_utils import signal_array_to_masked_vol

from nutmeg.core.beam import Beam, search_any_pybeam, MEG_coreg
from nutmeg.external import descriptors as desc
from nutmeg.core import TEMPLATE_MRI_PATH

class TFBeam(Beam):

    # XYZ These "one time properties" still need to be disabled if they
    # are not in the signal transforms list
    
    @desc.auto_attr
    def f_raw(self):
        sa = self.sig['active']
        sc = self.sig['control']
        nz = self.sig['noise']
        sa -= nz
        sc -= nz
        return sa/sc
        
    @desc.auto_attr
    def f_db(self):
        # if I recompute all this now (as opposed to calling f = self.f_raw()),
        # there's one less array in memory after f_db is requested
        sa = self.sig['active']
        sc = self.sig['control']
        nz = self.sig['noise']
        sa -= nz
        sc -= nz
        f = sa/sc
        fdb = np.log10(f)
        fdb *= 10.0
        return fdb

    @desc.auto_attr
    def pseudo_f(self):
        f = self.f_raw
        pos_idc = (f>=1).nonzero()[0]
        neg_idc = (f<1).nonzero()[0]
        fp = np.zeros_like(f)
        np.putmask(fp, f>=1, f-1)
        np.putmask(fp, f<1, 1 - 1/f)
        return fp

    @desc.auto_attr
    def active_power(self):
        return self.sig['active']

    @desc.auto_attr
    def control_power(self):
        return self.sig['control']

    @desc.auto_attr
    def noise_power(self):
        return self.sig['noise']
    
    # Keep these in order of the __init__ call signature
    _argnames = Beam._argnames + ['bands', 'timewindow']
    _kwnames = Beam._kwnames + ['fixed_comparison', 'uses']
    
    # A descriptive name to attribute name lookup ..
    # This also controls what types of transforms are available, and
    # can be manipulated internally
    __signal_transforms = {
        'F raw' : 'f_raw',
        'F dB' : 'f_db',
        'CTF Pseudo-F' : 'pseudo_f',
        'Active Power' : 'active_power',
        'Control Power' : 'control_power',
        'Noise Power' : 'noise_power'
        }
    signal_transforms = property(lambda x: x.__signal_transforms.keys(),
                                 None, None,
                                 'available transforms of the signal data')

    @staticmethod
    def signal_transform_names():
        return TFBeam.__signal_transforms.keys()
    
    # default value
    _ratio_type = 'F dB'
    
    def __init__(self, voxelsize, voxels, srate, timepts,
                 sig, coreg, bands, timewindow,
                 coordmap=None, fixed_comparison=None, uses='F dB'):
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
        coordmap : NIPY Affine object
          the MEG voxel index coordinate to voxel location coordinate mapping

        """
        sig, voxels = self._init_signal(sig, voxels, fixed_comparison)
        Beam.__init__(self, voxelsize, voxels, srate, timepts, sig, coreg,
                      coordmap=coordmap)
        self.bands = np.atleast_2d(bands)
        self.timewindow = np.atleast_2d(timewindow)
        if not fixed_comparison:
            # still mark it down for posterity (ie, array "pickling")
            self.fixed_comparison = fixed_comparison
            self.uses = uses
        else:
            self.reinterpret_signal_as(fixed_comparison)
            
    def _init_signal(self, sig, voxels, fcomp):
        """Try to parse out the signal components and valid voxels in
        data coming from a variety of sources.
        """
        
        # In this case  there are no separable active/control components.
        if len(sig) not in (2,3) and len(sig.dtype) not in (2,3):
            if fcomp:
                return sig, voxels
            else:
                raise ValueError(
"""Ambiguous signal argument has only 1 component and no fixed active to noise
comparison has been specified. The signal should have at least 2 components
corresponding to (active, control, [noise]).
""")

        beam_dtype = np.dtype([('active', np.float64),
                               ('control', np.float64),
                               ('noise', np.float64)])

        if len(sig) in (2,3): 
            # This signal is coming from a Matlab cell type
            if len(sig) == 3:
                sa, sc, nz = map(np.atleast_3d, sig)
            elif len(sig) == 2:
                sa, sc = map(np.atleast_3d, sig)
                nz = np.zeros_like(sa)
            valid_sig = sa[:,0,0].nonzero()[0]
            sa = sa[valid_sig]
            sc = sc[valid_sig]
            nz = nz[valid_sig]
            voxels = voxels[valid_sig]
            sig = np.empty(sa.shape, dtype=beam_dtype)
            sig['active'] = sa
            sig['control'] = sc
            sig['noise'] = nz
        else:
            # This signal is coming from a npy file
            sig_fields = set(sig.dtype.fields.keys())
            beam_fields = set(beam_dtype.fields.keys())
            matching = beam_fields.intersection(sig_fields)
            nomatch = sig_fields.difference(beam_fields)
            if len(nomatch):
                new_sig = np.empty(sig.shape, dtype=beam_dtype)
                for f in matching:
                    new_sig[f] = sig[f]
                for f in nomatch:
                    new_sig[f] = np.zeros(sig.shape)
                sig = new_sig                
        return sig, voxels

    @classmethod
    def from_mat_file(class_type, matfile, **kwargs):
        if type(matfile) == np.ndarray:
            # preloaded data
            beam = matfile
        else:
            try:
                beam = sio.matlab.loadmat(matfile,
                                          squeeze_me=False,
                                          struct_as_record=True)['beam']
            except KeyError:
                print "there was no 'beam' element in this mat file"
        
        # need to fix up the shape of arrays in the struct recarray
        argdict = {}
        for name in beam.dtype.names:
            a = beam[name][0,0]
            while a.shape == (1,1):
                a = a[0,0]
            argdict[name] = np.squeeze(a)

        if len(argdict['s']) not in (2,3) and \
           'fixed_comparison' not in kwargs:
            kwargs['fixed_comparison'] = 'unknown'

        coregi = MEG_coreg.from_mat_struct(argdict['coreg'])

        return class_type(argdict['voxelsize'], argdict['voxels'],
                          argdict['srate'], argdict['timepts'],
                          argdict['s'], coregi,
                          argdict['bands'], argdict['timewindow'],
                          **kwargs)


    def reinterpret_signal_as(self, fixed_comparison):
        """Change the name of the active-to-control comparison. If this
        TFBeam is already fixed to a comparison, simply change the name of it.
        If this TFBeam has separate active and control components, then this
        method will fix the comparison as the requested one, and delete the
        old component signal data.
        """
        # Through property magic, make self.s synonymous with
        # self.sig (through association with the keyword fixed_comp),
        # and also make it the only available trasform/signal.
        # NOTE: written this way, fixed_comp can be "user defined",
        # and doesn't need to be one of the predefined options
        if len(self.sig.dtype) not in (2,3):
            self.__signal_transforms = dict(( (fixed_comparison, 'sig'), ))
            self.uses = fixed_comparison
        else:
            # In this case a fixed comparison is being used in order
            # to reduce the data (and also the memory footprint).
            # Calculate the fixed comparison once, and then throw away the
            # signal components.
            prev_attrs = self.__signal_transforms.values()
            self.uses = fixed_comparison
            self.sig = self.s
            self.__signal_transforms = dict( ((self.uses, 'sig'),) )
            for attr in prev_attrs:
                try:
                    delattr(self, attr)
                except:
                    pass
        self.fixed_comparison = fixed_comparison
            
    def get_array_by_name(self, name):
        return getattr(self, self.__signal_transforms[name])

    #### A little property magic for run-time control of behavior ####
    # The "uses" attribute
    def _get_ratio_type(self):
        return self._ratio_type
    def _set_ratio_type(self, name):
        un = name.upper()
        ul = [n.upper() for n in self.signal_transforms]
        try:
            i = ul.index(un)
            self._ratio_type = self.signal_transforms[i]
        except:
            w.warn('ratio type %s not available, see TFBeam.signal_transforms'%name)
    uses = property(_get_ratio_type, _set_ratio_type, None,
                    'what type of active-to-control ratio is exposed')

    # The "s" attribute
    def _get_ratio_signal(self):
        return self.get_array_by_name(self.uses)
    s = property(_get_ratio_signal, None, None,
                 'exposed active-to-control signal ratio')
    ##################################################################


    def to_ni_image(self, t, f, grid_shape=None, prior_mask=None):
        """Form a NIPY Image of the map of this object at (t,f). The new
        Image will have a CoordinateMap transforming the array indices into
        the target space of this Beam's coregistered MRI.
        """
        print 'slicing tf beam at t,f =', (t,f)
        if (prior_mask is not None) and (prior_mask.shape == self.s.shape):
            prior_mask = prior_mask[:,t,f]
        m_arr = signal_array_to_masked_vol(self.s[:,t,f], self.voxel_indices,
                                           grid_shape=grid_shape,
                                           prior_mask=prior_mask)
        meg2mri = self.coreg.meg2mri
        return ni_api.Image(m_arr, compose(meg2mri, self.coordmap))

    def from_new_dataset(self, new_s, new_vox=None,
                         multi_subj=False, uses=None, fixed_comparison=None):
        if new_vox is None:
            new_vox = self.voxels
        if multi_subj:
            coregi = MEG_coreg(TEMPLATE_MRI_PATH, TEMPLATE_MRI_PATH,
                               np.eye(4), self.coreg.fiducials)
        else:
            coregi = self.coreg
        return TFBeam(self.voxelsize, new_vox, self.srate, self.timepts,
                      new_s, coregi, self.bands, self.timewindow,
                      coordmap=self.coordmap, uses=uses,
                      fixed_comparison=fixed_comparison)

def tfbeam_from_file(fname, **kwargs):
    """Return a TFBeam object from a valid TFBeam file

    Parameters
    ----------
    fname : str
      path to a numpy or MATLAB TFBeam file
    """
    if not os.path.exists(fname):
        raise ValueError('no such file %s'%fname)
    if os.path.splitext(fname)[-1] in ('.npz', '.npy'):
        #return TFBeam.from_npy_file(fname, **kwargs)
        return TFBeam.load(fname, **kwargs)
    elif os.path.splitext(fname)[-1] == '.mat':
        return TFBeam.from_mat_file(fname, **kwargs)
    
def load_tfbeam(beam):
    """Return a TFBeam from various descriptions

    Parameters
    ----------
    beam : str, or TFBeam
      Either an exiting TFBeam object, or the path to a valid TFBeam file  
    """
    if type(beam) in (str, unicode):
        if os.path.splitext(beam)[-1] == '.mat':
            beam = TFBeam.from_mat_file(beam)
        elif os.path.splitext(beam)[-1] in ('.npz', '.npy'):
            #beam = TFBeam.from_npy_file(beam)
            beam = TFBeam.load(beam)
        return beam
    elif type(beam) is TFBeam:
        return beam
    else:
        raise ValueError('beam type not understood: %s'%repr(type(beam)))
    
