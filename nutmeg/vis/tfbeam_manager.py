import numpy as np
from numpy.lib.index_tricks import unravel_index

import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

# do this to kick-in ETSConfig code.. 
import nutmeg

from traits.api \
    import Instance, Enum, Dict, Constant, Str, \
    List, on_trait_change, Float, File, Array, Button, Range, Property, \
    cached_property, Event, Bool, Tuple, Int, DelegatesTo
    
from traits.ui.api \
    import Item, Group, View, VGroup, HGroup, HSplit, \
    EnumEditor, CheckListEditor, ListEditor, message, ButtonEditor, \
    RangeEditor, Include

## from traits.ui.file_dialog import open_file
from xipy.vis.qt4_widgets import browse_files
from xipy.overlay.interface import OverlayInterface, OverlayWindowInterface
from xipy.slicing.image_slicers import SampledVolumeSlicer, \
     ResampledVolumeSlicer, ResampledIndexVolumeSlicer
from xipy.volume_utils import maximum_world_distance, signal_array_to_masked_vol
from nutmeg.core.tfbeam import load_tfbeam
from nutmeg.core import tfbeam
from nutmeg.stats import tfstats_results
from nutmeg.vis.mpl_tf_plane import TimeFreqPlaneFigure
from nipy.core import api as ni_api
from nipy.core.reference.coordinate_map import compose

fill_values = {
    'F dB': 0,
    'F raw': 1,
    'T test': 0,
    'p val pos (corr)': 1,
    'p val pos (uncorr)' : 1,
    'p val neg (corr)' : 1,
    'p val neg (uncorr)' : 1,
    }


class TFBeamManager( OverlayInterface ):
    """ Manages the GUI manipulation of a TFBeam object, including:
        * Extrema localization in partitioned or all dimensions
        * Active to control ratio transforms
        * Reinterpreting data as various other useful types (NIPY Image,
          (MaskedArray, VolumeSlicer type)
    """
    # Signal manipulation--------------------
    _beam_transforms = List
    transforms = Enum(values='_beam_transforms')

    _using_extra_map = Bool(False)
    pextrabutton = Button('Plot')
    resetsignal = Button('Reset Plot')
    alt_names = Enum(*tfbeam.TFBeam.signal_transform_names())
    rbutton = Button('Rename')
    lbutton = Button('Load TF Beam')
    sbutton = Button('Load SnPM Stats Arrays')

    # Beam Stats Manager object---------------------------
    bstats_manager = Instance(
        'nutmeg.vis.tfstats_manager.TimeFreqSnPMaps', ()
        )
    bstats_changed = Event
    threshold = DelegatesTo('bstats_manager')

    # Peak finding---------------------------
    tf_idx = Tuple((0,0))
    description = Property(depends_on='overlay_updated')
    vox_idx = Int(-1)
    ana_xform = Enum('max', 'min', 'absmax')
    # seems that using "extended trait names" in the Range construction
    # requires extended names for both limits (so make a dummy trait _one)
    _one = Int(1)

    
    #### THIS SECTION IN FLUX BECAUSE OF A POSSIBLE BUG IN TRAITS
    #### WHERE PROPERTIES CONTINUALLY GET RECOMPUTED WHEN THERE ARE
    #### HIDDEN VIEW ELEMENTS. LAME!

    # Extrema in all dimensions
    _num_alldim_features = Int(150)
    all_order = Range('_one', '_num_alldim_features')
    all_button = Button('Locate In All Dimensions')
##     ordered_idx_alldim = Property(Array)
    _ordered_idx_alldim = Array(dtype='uint')
    recompute_all_order = Bool(True) # reset every time work_arr is recomputed

    # Extrema in spatial dimensions
    _num_spatial_features = Int(150)
    spatial_order = Range('_one', '_num_spatial_features')
    spatial_button = Button('Locate In Space')
##     ordered_idx_spatial = Property(Array)
    _ordered_idx_spatial = Array(dtype='uint')
    recompute_spatial_order = Bool(True) # reset every time work_arr is recomputed
    
    # Extrema in TimeFrequency space
    _num_tf_features = Int(150)
    tf_order = Range('_one', '_num_spatial_features')
    tf_button = Button('Locate In Time-Freq')
##     ordered_idx_tf = Property(Array)
    _ordered_idx_tf = Array(dtype='uint')
    recompute_tf_order = Bool(True) # reset every time work_arr is recomputed
    
    # FOV probably not more than 25cm (but could check voxels)
    _max_radius = Float
    _min_radius = Float(0.0)
    radius = Range(value='_max_radius', low='_min_radius', high='_max_radius')
    _good_vox = Array(dtype='ubyte')
    # reset when radius | vox changes (unless radius is at max)
    recompute_good_vox1 = Bool(True)
    recompute_good_vox2 = Bool(True)
    _work_arr = Array(dtype='float64')
    # reset when good_vox, ana_xform, beam_mask changes
    recompute_work_arr = Bool(True)

    # Signal and Mask------------------------
    beam_sig = Property(depends_on=['transforms', 'pextrabutton', 'resetsignal'])
    # beam mask is a negative mask (MaskedArray convention)
    _bmask_changed = Event
##     beam_mask = Property(Array, depends_on='_bmask_changed')
    beam_mask = Property(Array, depends_on='bstats_manager.thresh_changed')
##     _beam_mask = Array(dtype='B')

    # Fill value for external masked array handling (eg in Mayavi)
    fill_value = Property(Float)
    _fill_value = Float(np.nan)

    def _lame_coupling(self):
        print 'bmask changed'
        self._bmask_changed = True
    
    def __init__(self, bbox,
                 image_signal=None,
                 loc_signal=None,
                 props_signal=None,
                 beam_vox_signal=None,
                 **traits):
        """
        Create a TFBeamManager

        Parameters
        ----------
        bbox : iterable
            the {x,y,z} limits of the enrounding volume box to which TFBeam
            overlays will be mapped.
        loc_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when peak finding
            (call pattern is loc_signal.emit(x,y,z))
        props_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when image colormapping
            properties change
        image_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when updating the image
            (call pattern is image_signal.emit(self))
        beam_vox_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when the current TFBeam
            voxel list index has changed. (call pattern is beam_vox_signal(vox))
        """
        self.bbox = bbox
        self.beam = None
        # want to start out with larger alpha, unless requested differently
        if 'alpha_scale' not in traits:
            traits['alpha_scale'] = 2.0
        OverlayInterface.__init__(self,
                                  loc_signal=loc_signal,
                                  props_signal=props_signal,
                                  image_signal=image_signal,
                                  **traits)
        self.beam_vox_signal = beam_vox_signal
        self._max_radius = np.round(maximum_world_distance(bbox))
        self.__needs_refresh = False
        self.bstats_manager.tfbeam_man = self
        self.bstats_manager.on_trait_change(self._lame_coupling,
                                            'thresh_changed')

    #-------------------------------------------------------------------------
    ### PROPERTY FETCHERS AND SETTERS.. RUNTIME ATTRIBUTE LOGIC... ETC
        
    def __beam_transforms_default(self):
        return []

    def __stats_maps_default(self):
        return []

    @cached_property
    def _get_beam_sig(self):
        if self._using_extra_map:
            s = self.bstats_manager.get_map_by_name(self.stats_map)
        elif not self.transforms:
            s = np.ones((1,1,1))
        else:
            s = self.beam.s
        self.norm = (s.min(), s.max())
        return s

    def _get_fill_value(self):
        signal_type = self.stats_map if self._using_extra_map \
                      else self.transforms        
        return fill_values.get(signal_type, np.nan)
    
    @cached_property
    def _get_beam_mask(self):
        """Returns a negative MaskedArray convention binary mask
        """
        if self.threshold is None:
            return None
        m = self.threshold.create_tf_binary_mask()
        if m is None or m.shape != self.beam_sig.shape:
            return None
        return m

    @cached_property
    def _get_description(self):
        if not self.beam:
            return ''
        t, f = self.tf_idx
        time_pt = self.beam.timepts[t]
        fa, fb = self.beam.bands[f]
        if self.threshold.thresh_map_name:
            um_pts = str(self.threshold.unmasked_points)
        else:
            um_pts = 'no mask'
        mn, mx = np.ma.min(self.overlay._data), np.ma.max(self.overlay._data)
        dstr = \
"""
(t,f) --> %1.1f ms, [%1.1f Hz - %1.1f Hz] band
data range: (%1.3f, %1.3f)
unmasked pts: %s
"""%(time_pt,fa,fb,mn,mx,um_pts)
        return dstr

    #-------------------------------------------------------------------------
    ## THIS SECTION IS A LITTLE CRAZY.. DUE TO A POSSIBLE BUG IN THE
    ## TRAITS UI SYSTEM, SOME PROPERTIES ARE RE-COMPUTED INAPPROPRIATELY
    ## SO THESE ARE A BUNCH OF METHODS THAT ARE NO LONGER PROPERTIES,
    ## BUT PROPERTY-LIKE
    @on_trait_change('beam_sig')
    def _flush_all_cached(self):
        self.recompute_good_vox = True
##         # _good_vox actually should be reset here and now before
##         # the radius slider can get moved around
##         foo = self.good_vox()
        self.recompute_work_arr = True
        self.recompute_all_order = True
        self.recompute_spatial_order = True
        self.recompute_tf_order = True

    ## GOOD (SEARCHED) VOXELS
    @on_trait_change('vox_idx, radius')
    def _set_recompute_good_vox1(self):
        self.recompute_good_vox1 = True
##     @on_trait_change('radius')
##     def _set_recompute_good_vox2(self):
##         self.recompute_good_vox2 = self.radius != self._max_radius

    def good_vox(self):
        cond = self.recompute_good_vox1
        # if not set to recompute, then just return 
        if not cond:
            return self._good_vox
        
        self.recompute_good_vox1 = False
        # Only do work if radius < max_radius, or if there is a vox set.
        if self.vox_idx < 0 or self.radius == self._max_radius:
##             print 'soft computing good_vox'
##             print 'grabbing beam_sig from good_vox'
            self._good_vox = np.ones((len(self.beam_sig),), dtype=np.bool)
            return self._good_vox
##         print 'recomputing good vox'
        xyz = self.beam.voxels[self.vox_idx]
        print 'current vox', xyz, 'current radius:', self.radius
        vdist = ((self.beam.voxels - xyz)**2).sum(axis=-1)
        self._good_vox = vdist <= self.radius**2
        print '# of voxels within radius:', self._good_vox.sum()
        return self._good_vox

    ## WORK ARRAY -- USED FOR PEAK FINDING
    @on_trait_change('ana_xform, _bmask_changed, recompute_good_vox+')
    def _set_recompute_work_arr(self, name, val):
        if name.find('recompute_good_vox') == 0:
            self.recompute_work_arr = self.recompute_good_vox1
##             print 'resetting due to good vox dirt', self.recompute_work_arr
        else:
            self.recompute_work_arr = True

    def work_arr(self):
        if not self.recompute_work_arr:
            return self._work_arr
        self.recompute_work_arr = False
##         print 'recomputing work_arr'
        # this will be a masked array, where the mask is based on whatever
        # beam_mask is, and whatever the "good_vox" are
        gv = np.logical_not(self.good_vox())
##         print 'grabbing beam_mask from work_arr'
        bmask = self.beam_mask
        if bmask is not None:
            m = self.beam_mask | gv[:,None,None]
        else:
##             print 'grabbing beam_sig from work_arr1'
            bsig = self.beam_sig
            rep = np.product(bsig.shape[1:])
            m = gv.repeat(rep).reshape(bsig.shape)
##         print m.shape, gv.shape, m.sum()
##         print 'grabbing beam_sig from work_arr2'
        m_arr = np.ma.masked_array(self.beam_sig, mask=m, copy=False)
        self._work_arr = np.abs(m_arr) if self.ana_xform=='absmax' else m_arr
        return self._work_arr
    
    ## ORDERED INDICES IN VARIOUS DIMENSION PARTITIONS
    @on_trait_change('recompute_work_arr')
    def _set_recompute_orders(self):
        if self.recompute_work_arr:
            self.recompute_all_order = True
            self.recompute_spatial_order = True
            self.recompute_tf_order = True
    
    def ordered_idx_alldim(self):
        """ Create a list of sorted map indices for all dimensions
        """
        if not self.recompute_all_order:
            return self._ordered_idx_alldim
        self.recompute_all_order = False
##         print 'recomputing alldim ordered'
##         print 'grabbing work_arr from ordered_idx_alldim'
        m_arr = self.work_arr()
        sidx = m_arr.flatten().argsort()
        nz = m_arr.mask.flat[sidx].nonzero()[0]
        if nz.shape not in ( (), (0,) ):
            last_good = nz[0]
        else:
            # the mask is all False (meaning all unmasked)
            last_good = len(sidx)    
        self._num_alldim_features = last_good
        self._ordered_idx_alldim = sidx[:last_good]
        return self._ordered_idx_alldim

    def ordered_idx_spatial(self):
        if not self.recompute_spatial_order:
            return self._ordered_idx_spatial
        self.recompute_spatial_order = False
##         print 'recomputing spatial ordered'
        ti, fi = self.tf_idx
##         print 'grabbing work_arr form ordered_idx_spatial'
        m_arr = self.work_arr()[:,ti,fi]
        sidx = m_arr.flatten().argsort()
        nz = m_arr.mask.flat[sidx].nonzero()[0]
        if nz.shape not in ( (), (0,) ):
            last_good = nz[0]
        else:
            # the mask is all False (meaning all unmasked)
            last_good = len(sidx)    
        self._num_spatial_features = last_good
        self._ordered_idx_spatial = sidx[:last_good]
        return self._ordered_idx_spatial

    def ordered_idx_tf(self):
        if not self.recompute_tf_order:
            return self._ordered_idx_tf
        self.recompute_tf_order = False
##         print 'recomputing tf ordered'
        v_idx = self.vox_idx
##         print 'grabbing work_arr from ordered_idx_Tf'
        m_arr = self.work_arr()[v_idx]
        sidx = m_arr.flatten().argsort()
        nz = m_arr.mask.flat[sidx].nonzero()[0]
        if nz.shape not in ( (), (0,) ):
            last_good = nz[0]
        else:
            # the mask is all False (meaning all unmasked)
            last_good = len(sidx)    
        self._num_spatial_features = last_good
        self._ordered_idx_tf = sidx[:last_good]
        return self._ordered_idx_tf

    #### END OF CRAZY SECTION!
    
    #-------------------------------------------------------------------------
    ### TRAITS CALLBACKS
    def _lbutton_fired(self):
        f = browse_files(None, dialog='Select File',
                         wildcard='*.mat *.npy *.npz')
        if f:
            self.update_beam(f)

    def _sbutton_fired(self):
        f = browse_files(None, dialog='Select SnPM Arrays',
                         wildcard='*.mat *.npy *.npz')
        if f:
            sres = tfstats_results.load_tf_snpm_stats(f)
            self.bstats_manager.stats_results = sres

    def _rbutton_fired(self):        
        self.beam.fix_comparison(self.alt_names)
        self.update_beam(self.beam)

    def _pextrabutton_fired(self):
        self._using_extra_map = True
        self.signal_new_image()
##         self.comm_with_listener('signal')

    def _resetsignal_fired(self):
        self._using_extra_map = False
        self.signal_new_image()
##         self.comm_with_listener('signal')

    def _all_button_fired(self):
        self.find_alldim_peak()

    def _spatial_button_fired(self):
        self.find_spatial_peak()

    def _tf_button_fired(self):
        self.find_tf_peak()

    #-------------------------------------------------------------------------
    ### COUPLING WITH LISTENING/INTERACTING OBJECTS

    # This update may be used as a PyQt4 callback with signature (x,y,z)
    def main_xyz_changed(self, *mri_xyz):
        print mri_xyz
        self.world_position = mri_xyz

    # This update is a traditional Traits callback
    @on_trait_change('world_position')
    def _change_vox_idx(self):
        if self.beam is None:
            return
        self.vox_idx = self.beam.vox_lookup_from_mr(self.world_position)

    @on_trait_change('transforms, _bmask_changed')
    def signal_new_image(self):
        if self.beam is None:
            return
        self.beam.uses = self.transforms
        self.overlay = self.to_overlay(*self.tf_idx)
        # fire the Traits Event
        self.overlay_updated = True
        if self.image_signal is not None:
            print 'signalling new image'
            self.image_signal.emit(self)
        # is this needed??
        self.__needs_refresh = False

    def signal_new_vox(self):
        meg_xyz = self.beam.voxels[self.vox_idx]
        mri_xyz = self.beam.coreg.meg2mri(meg_xyz)
        print 'signalling new world position'
        self.world_position_updated = True
        if self.loc_signal is not None:
            self.loc_signal.emit(*mri_xyz)

    @on_trait_change('vox_idx')
    def signal_new_beam_vox_idx(self):
        if self.vox_idx>=0:
            self.trait_setq(world_position = self.beam.voxels[self.vox_idx])
        if self.beam_vox_signal:
            self.beam_vox_signal.emit(self.vox_idx)

    @on_trait_change('tf_idx')
    def signal_new_tf(self):
        self.signal_new_image()

##         if self._tf_signal is not None:
##             ti, fi = self.tf_idx
##             emitting = (self.beam.timepts[ti],) + tuple(self.beam.bands[fi])
##             self._tf_signal.emit(*emitting)

    #-------------------------------------------------------------------------
    ### UTILITY METHODS
    def alpha(self, scale=None):
##         half_alpha_db = 0.5
##         pc = np.polyfit([-half_alpha_db, 0, half_alpha_db],
##                         [192., 64., 192.], 2)
        if not self.beam:
            return 1.0
        if scale is None:
            scale = self.alpha_scale
        sig_range = np.linspace(self.norm[0], self.norm[1], 256)
        sig_range[np.argmin(sig_range)] = 0
        sig_range *= (scale * 2*np.pi/max(abs(self.norm[0]), abs(self.norm[1])))

        signal_type = self.stats_map if self._using_extra_map \
                      else (self.transforms or '')
        print signal_type, self.fill_value
        if signal_type in ('F dB' 'T test'):
            # de-emphasize 0, emphasize neg and pos
            #f = lambda x: np.polyval(pc, scale*x)
            f = lambda x: np.abs(np.arctan(sig_range)) * (2/np.pi)
        elif signal_type == 'F raw':
            # de-emphasize 1, emphasize less than and greater than
            #f = lambda x: np.polyval(pc, 10*np.log10(scale*x))
            sig_range = 10*np.log10(sig_range)
            f = lambda x: np.abs(np.arctan(sig_range)) * (2/np.pi)            
        elif signal_type.find('p val') >= 0:
            # ramp down from 1 to 0
            f = lambda x: np.linspace(1, 0, 256)*scale
        else:
            # ramp up from 0 to 1
            f = lambda x: np.linspace(0, 1, 256)*scale
        a = np.clip(f(sig_range), 0, 1)
##         # this should also hide NaNs???
##         a[0] = 0
##         if threshold and self.threshold.thresh_map_name == '':
##             tval, comp = self.threshold
##             mn, mx = self.norm
##             lut_map = int(255 * (tval - mn)/(mx-mn))
##             if comp == 'greater than':
##                 a[:lut_map] = 0
##             else:
##                 a[lut_map+1:] = 0
        return a

    ### PEAK FINDING
    @on_trait_change('all_order')
    def find_alldim_peak(self):
##         print 'grabbing ordered_idx_alldim from find_alldim_peak'
        o_idx = self.ordered_idx_alldim()
        if o_idx.shape in ( (), (0,) ):
            print 'no features, all masked'
            return
        if self.ana_xform in ('absmax', 'max'):
            # find the (last-order) unmasked ordered index
            all_pk_idx = o_idx[-self.all_order]
        else:
            # find the lowest order unmasked index
            all_pk_idx = o_idx[self.all_order-1]
##         print 'grabbing beam_sig from find_alldim_peak'
        vx, ti, fi = unravel_index(all_pk_idx, self.beam_sig.shape)

        # update state and send off the new image signal
        self.vox_idx = vx
        self.tf_idx = ti, fi
        self.signal_new_vox()
        self.signal_new_image()
##         self.comm_with_listener(tuple(vox_loc) + (ti, fi))

    @on_trait_change('spatial_order')
    def find_spatial_peak(self):
##         print 'grabbing ordered_idx_spatial from find_spatial_peak'
        o_idx = self.ordered_idx_spatial()
        if o_idx.shape in ( (), (0,) ):
            print 'no features, all masked'
            return
        if self.ana_xform in ('absmax', 'max'):
            # find the (last-order) unmasked ordered index
            spatial_pk_idx = o_idx[-self.spatial_order]
        else:
            # find the lowest order unmasked index
            spatial_pk_idx = o_idx[self.spatial_order-1]

        # update state of vox_idx and send new vox signal
        self.vox_idx = int(spatial_pk_idx)
        self.signal_new_vox()
##         self.comm_with_listener(vox)

    @on_trait_change('tf_order')
    def find_tf_peak(self):
##         print 'grabbing ordered_idx_tf from find_tf_peak'
        o_idx = self.ordered_idx_tf()
        if o_idx.shape in ( (), (0,) ):
            print 'no features, all masked'
            return        
        if self.ana_xform in ('absmax', 'max'):
            tf_pk_idx = o_idx[-self.tf_order]
        else:
            tf_pk_idx = o_idx[self.tf_order-1]
##         print 'grabbing beam_sig from find_tf_peak'
        ti, fi = unravel_index(tf_pk_idx, self.beam_sig.shape[1:])

        # update state of tf_idx and send new image signal
        self.tf_idx = ti, fi
##         self.signal_new_image()
##         self.comm_with_listener((ti,fi))
        
    #-------------------------------------------------------------------------
    ### DATA EXPORT FUNCTIONS
    
    def to_masked_array(self, t_idx, f_idx, grid_shape=None):
        sig = self.beam_sig[:,t_idx,f_idx]
        vox = self.beam.voxel_indices
        bmask = self.beam_mask
        if bmask is not None:
            vox_mask = bmask[:,t_idx,f_idx]
            if not (vox_mask==self.threshold.binary_mask).all():
                print 'warning! beam mask not synced with threshold mask!'
        else:
            vox_mask = None
        
        m_arr = signal_array_to_masked_vol(sig, vox,
                                           grid_shape=grid_shape,
                                           prior_mask=vox_mask,
                                           fill_value=self.fill_value)
        return m_arr
    
    def to_ni_image(self, t_idx, f_idx, grid_shape=None):
        m_arr = self.to_masked_array(t_idx, f_idx, grid_shape=grid_shape,
                                     fill_value=self.fill_value)
        meg2mri = self.beam.coreg.meg2mri
        return ni_api.Image(m_arr.filled(),
                            compose(meg2mri, self.beam.coordmap))

    def to_overlay(self, t_idx, f_idx, grid_spacing=None):
        m_arr = self.to_masked_array(t_idx, f_idx)
        meg2mri = self.beam.coreg.meg2mri
        img = ni_api.Image(m_arr,
                           compose(meg2mri, self.beam.coordmap))
        return img
        
    #-------------------------------------------------------------------------
    ### DATA REFRESH
    def update_beam(self, b):
        try:
            self.beam = load_tfbeam(b)
        except Exception, e:
            print 'could not load beam %s'%repr(b)
            raise e
        print 'beam JUST loaded'
        self.__needs_refresh = True
        self._beam_transforms = self.beam.signal_transforms
        self.trait_setq(
            transforms=self.beam.uses,
            vox_idx = self.beam.vox_lookup_from_mr(self.world_position)
            )
        # do this last to hopefully trigger new image signal
        self.resetsignal = True
        if self.__needs_refresh:
            print 'kicking tardy signal out'
            self.signal_new_image()


    def make_panel(self, parent=None):
        tab = QtGui.QTabWidget(parent)
        i = tab.addTab(
            self.edit_traits(
                parent=parent,
                kind='subpanel').control,
            'Nutmeg TFBeam Manager')
        tab.addTab(self.bstats_manager.edit_traits(
            parent=parent,
            kind='subpanel'
            ).control,
                   'TFBeam Stats Manager')
        tab.setCurrentIndex(i)
        return tab


    bgroup = VGroup(
        Item('lbutton', show_label=False),
        Item('sbutton', show_label=False),
        Item('transforms', label='Beam Transforms', width=40),
        HGroup(Item('alt_names', label='Rename Transform', width=40),
               Item('rbutton', show_label=False),
               visible_when='transforms=="unknown"'),
        HGroup(Item('stats_map', label='Other Stats Maps', width=40),
               Item('pextrabutton', show_label=False),
               visible_when='len(object._stats_maps) > 1'),
        Group(Item('resetsignal', show_label=False),
              visible_when='object._using_extra_map'),
        Item('cmap_option', label='Colormap', width=20),
        Item('alpha_scale', label='Alpha Scaling')
        )

    loc_group = VGroup(
        Item('ana_xform', label='Find Features', width=10),
        HGroup(
            Item('all_button', show_label=False),
            Item('all_order', show_label=False, style='simple')
            ),
        HGroup(
            Item('spatial_button', show_label=False),
            Item('spatial_order', show_label=False, style='simple')
            ),
        HGroup(
            Item('tf_button', show_label=False),
            Item('tf_order', show_label=False, style='simple')
            ),
        Item('radius', style='simple',
             label='searched radius (mm)')
        )

                    
    view = View(
        Group(
            HGroup(
                bgroup,
                loc_group,
                ),
            ),
        resizable=True,
        title='Beam Signals List'
        )

#--------------- QT4 OVERLAY WINDOW INTERFACE CLASS --------------------------
from PyQt4 import QtGui, QtCore

class NmTimeFreqWindow(OverlayWindowInterface):
    
    tool_name = 'Nutmeg Time Freq MEG Window'

    bvox_signal = QtCore.pyqtSignal(int)
    
    def __init__(self, loc_connections, image_connections,
                 image_props_connections, bbox,
                 functional_manager=None,
                 external_loc=None, parent=None, main_ref=None,
                 tfbeam=None, **kwargs):
        """
        Creates a new MplQT4TimeFreqWindow, which controls interaction
        with Nutmeg TFBeam objects. This window is a QT4TopLevelAuxiliaryWindow,
        giving it top level status. It contains:
         *a 2D image of the time-frequency plane with selectable bins (emits a
          functional image update event and a tf event, described below)
         *a TFBeam management panel, with data management and feature
          localizations (events described below)
         *a threshold management panel, to create different mask criteria

         Time-frequency point updates cause emission of the tf_point signal.
         To cause methods to be connected to this signal, include them in the
         tf_connections iterable argument. Methods will have this signature
         pattern:
         meth(timept, freq_a, freq_b)

         Overlay image updates are also signalled. To cause methods to be
         connected, include them in the image_connections iterable argument.
         Methods will have this signature pattern:
         meth(obj)
          * obj will give access to obj.overlay, obj.alpha,
          * obj.norm, obj.fill_value

         Spatial location updates cause emission of the xyz_point signal. To
         cause methods to be connected to this signal, include them in the
         loc_connections interable argument. Methods will have this signature
         pattern:
         meth(x, y, z)

         Parameters
         ----------
         loc_connections: iterable
             Methods to connect to the xyz_point signal
         image_connections: iterable
             Methods to connect to the new_image signal
         parent: QObject (optional)
             A QT4 QObject to set as the parent of this widget (thus causing
             this widget NOT to run as a top level window)
         main_ref: QObject (optional)
             A QT4 QObject to be considered as the "parent" of this widget,
             when this widget is running as a top level window
         tfbeam: TFBeam (optional)
             A TFBeam with which to preload the beam manager.
         **kwargs:
             figsize, dpi for the time-frequency plane figure
         """

        
        figsize = kwargs.pop('figsize', (6,4))
        dpi = kwargs.pop('dpi', 100)
        # insert some local callbacks into the connection lists
        image_connections = (self._update_from_new_overlay,) + \
                            image_connections
        image_props_connections = (self.update_tf_figure,) + \
                                  image_props_connections
        OverlayWindowInterface.__init__(self,
                                        loc_connections,
                                        image_connections,
                                        image_props_connections,
                                        bbox,
                                        external_loc=external_loc,
                                        parent=parent,
                                        main_ref=main_ref,
                                        functional_manager=functional_manager)

        # make sure that when the tfbeam manager's voxel index changes,
        # the tfplane image gets updated
        self.bvox_signal.connect(self.update_tf_figure)
        if functional_manager is None or \
           not isinstance(functional_manager, TFBeamManager):
            self.func_man = TFBeamManager(
                bbox, image_signal = self.image_changed,
                loc_signal = self.loc_changed,
                props_signal = self.image_props_changed,
                beam_vox_signal=self.bvox_signal
                )

        self.func_man.beam_vox_signal = self.bvox_signal
        
        vbox = QtGui.QVBoxLayout(self)

        # set up figure
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.canvas = Canvas(fig)
        QtGui.QWidget.setSizePolicy(fig.canvas,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        Canvas.updateGeometry(fig.canvas)
        fig.canvas.setParent(self)
        self.fig = fig

        # fake plane at first
        self.tf_plane = self._initialize_tf_plane()
        vbox.addWidget(self.fig.canvas)

        vbox.addWidget(self.func_man.make_panel(parent=self))

        QtGui.QWidget.setSizePolicy(self,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        if tfbeam is not None:
            self.func_man.update_beam(tfbeam)

    def _initialize_tf_plane(self):
        t_ax = np.array([-1,0,1,2])
        f_ax = np.array([-1,0,1,2])
        self.fig.clear()
        tf_plane = TimeFreqPlaneFigure(self.fig, t_ax, f_ax)
        tf_plane.draw()
        return tf_plane

    def activate(self):
        if self._activated:
            return
        # call parent class functionality first
        OverlayWindowInterface.activate(self)
        # connect a button press on the TF plane to an update MEG image event
        self._cx_id = self.tf_plane.canvas.mpl_connect(
            'button_press_event',
            self._coordinate_tf_index
            )
        # connect buttons presses on the ortho figures to an update voxel event
        try:
            self.external_loc.connect(self.func_man.main_xyz_changed)
        except:
            print 'not connected to external location signal'
        self._activated = True

    def deactivate(self, strip_overlay=False):
        if not self._activated:
            return
        OverlayWindowInterface.deactivate(self)
        if strip_overlay:
            self.tf_plane = self._initialize_tf_plane()
        self.tf_plane.canvas.mpl_disconnect(self._cx_id)
        try:
            self.external_loc.disconnect(self.func_man.main_xyz_changed)
        except:
            print 'could not disconnect external location callbacks'
            pass
        self._activated = False

    def _update_from_new_overlay(self):
        # this is only called from the func_man
        beam = self.func_man.beam
        t_ax = np.linspace(beam.timewindow.min(), beam.timewindow.max(),
                           len(beam.timewindow)+1)
        f_ax = np.array([beam.bands[0,0]] + [b[-1] for b in beam.bands])
        # only updating axes if this is a new beam with new dimensions
        if not np.allclose(t_ax, self.tf_plane.t_ax) or \
           not np.allclose(f_ax, self.tf_plane.f_ax):
            self.tf_plane.reset_plane(t_ax, f_ax)
        print 'new image update of tf figure:', self.func_man.vox_idx
        self.update_tf_figure(self.func_man.vox_idx)
        self._coordinate_tf_index()
    
    def update_tf_figure(self, *args):
        """ Updates the TF plane properties and array. Connect this to
        beam_vox_signal and image_props_changed
        """

        try:
            beam = self.func_man.beam
        except:
            print 'returning because no bman'
            return

        cmap = self.func_man.colormap
        norm = mpl.colors.normalize(*self.func_man.norm)

        if type(args[0]) is not TFBeamManager:
            vidx = args[0]
            if vidx < 0:
                self.tf_plane.clear_plane()
                print 'cleared plane and returning because vidx<0'
                return
            print vidx
            plane = self.func_man.beam_sig[vidx,:,:].T
        else:
            # just update the properties
            plane = None
        # update tf plane plot
        self.tf_plane.update_tf_plane(plane, cmap=cmap, norm=norm)

    def _coordinate_tf_index(self, *args):
        if len(args) == 1:
            # listening for an MPL event in the TF plane
            # 1st check if user clicked outside the axes
            mpl_event = args[0]
            if mpl_event is not None and mpl_event.inaxes is None:
                return
            ti, fi = self.tf_plane.get_plane_index(mpl_event.xdata,
                                                   mpl_event.ydata)
            self.func_man.tf_idx = (ti, fi)
        elif len(args) == 0:
            # listens to tf_point emissions, just check if beam_man.tf_idx
            # matches tf_plane.current_index
            ti, fi = self.func_man.tf_idx
            if (fi, ti) != self.tf_plane.selected_tf_index:
                print 'pushing func_man tf_idx to tf_plane'
                self.tf_plane.update_selected_tf((fi, ti))

if __name__=='__main__':
    pass
