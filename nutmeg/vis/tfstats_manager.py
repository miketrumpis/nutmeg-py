# want to make a class that uses TimeFreqSnPMResults to create
# different maps and thresholds -- 
# available functions map(alpha) :
#
# map array |   threshold low    |  threshold high |  threshold mode
# test stat | neg tail cutoff   | pos tail cutoff  | mask between
# test pos  | pos tail cutoff   | max test value   | mask lower
# test neg  | min test value    | neg tail cutoff  | mask higher
# corr ppos | 0                 | alpha            | mask higher
# corr pneg | 0                 | alpha            | mask higher
# cluster   | 0                 | 0.5              | mask lower
import numpy as np
import traits.api as t_api
import traits.ui.api as ui_api
import traits.api as traits
from traits.api \
    import Instance, Enum, Tuple, Event, \
    String, List, on_trait_change, File, Array, Button, Range, \
    Property, cached_property, Bool, Int
    
from traits.ui.api import Item, Group, View, VGroup, \
     HGroup, EnumEditor, ListEditor, RangeEditor, CheckListEditor

from xipy.overlay.interface import ThresholdMap
from nutmeg.stats.tfstats_results import TimeFreqSnPMResults


class TimeFreqThresholdMap(ThresholdMap):
    """This class offers adaptation from Time x Freq x Spatial maps
    to simple spatial maps (controlled by a (time, freq) index)
    """

    _tf_idx = t_api.Tuple((0,0))

    _tf_map_scalars = t_api.Array
    map_scalars = t_api.Property(
        t_api.Array, depends_on='_tf_idx, _tf_map_scalars'
        )

    _tf_thresh_limits = t_api.Array
    thresh_limits = t_api.Property(
        t_api.Tuple #, depends_on='_tf_idx, _tf_thresh_limits'
        )

    @t_api.on_trait_change('_tf_idx')
    def _change_index(self):
        self.map_changed = True

    @t_api.cached_property
    def _get_map_scalars(self):
        if self._tf_map_scalars.size==0:
            return None
        t, f = self._tf_idx
        return self._tf_map_scalars[:,t,f]

##     @t_api.cached_property
    def _get_thresh_limits(self):
        # _tf_thresh_limits should be shaped (2, ntp, nfp)
        t, f = self._tf_idx
        if self._tf_thresh_limits.size==0:
            return (0,0)
        return self._tf_thresh_limits[:,t,f]

    def create_tf_binary_mask(self, type='negative'):
        """Create a binary mask in the shape of _tf_map_scalars for the
        current threshold conditions.

        Parameters
        ----------
        type : str, optional
            By default, make a MaskedArray convention mask ('negative').
            Otherwise, set mask to True where values are unmasked ('positive')
        """
        if not self.thresh_map_name:
            return None
        mode = self.thresh_mode
        limits = self._tf_thresh_limits
        map = self._tf_map_scalars
        if mode=='mask lower':
            m = (map < limits[0]) if type=='negative' else (map >= limits[0])
        elif mode=='mask higher':
            m = (map > limits[1]) if type=='negative' else (map <= limits[1])
        elif mode=='mask between':
            m = ( (map > limits[0]) & (map < limits[1]) ) \
                if type=='negative' \
                else ( (map <= limits[0]) | (map >= limits[1]) )
        else: # mask outside
            m = ( (map < limits[0]) | (map > limits[1]) ) \
                if type=='negative' \
                else ( (map >= limits[0]) & (map <= limits[1]) )
        return m

class TimeFreqSnPMaps(t_api.HasTraits):
    """This class interfaces with a TimeFreqSnPMResults object in
    order to provide different maps and thresholds for time-frequency
    MEG data.

    It also allows the user to create custom simple thresholds based
    on either the test statistic level, or the level of the MEG
    signal itself.

    """

    stats_results = Instance('nutmeg.stats.tfstats_results.TimeFreqSnPMResults')

    # Provide for a little talk-back with a TFBeamManager
    tfbeam_man = Instance('nutmeg.vis.tfbeam_manager.TFBeamManager')
    _tf_idx = Tuple((0,0))

    map_types = Property(depends_on='stats_results')
    map_type = Enum(values='map_types')
    
    clear_button = Button('Clear Mask')

    create_mask = Button('Create Threshold')
    user_thresholds = List
    # Provide a name of all the defined thresholds
    user_thresholds_names = List 

    thresh_changed = Int(1)
    threshold = Instance(TimeFreqThresholdMap)

    def __init__(self, **traits):
        t_api.HasTraits.__init__(self, **traits)
        # create default
        self.threshold

    def _threshold_default(self):
        return TimeFreqThresholdMap(thresh_map_name='')
    
    @on_trait_change('stats_results')
    def _clear_maps(self):
        self.threshold = self._threshold_default()
        self.user_thresholds = []
        self.user_thresholds_names = []

    @on_trait_change('thresh_changed')
    def foo(self):
        print 'thresh changed!'

    @on_trait_change('tfbeam_man')
    def _sync_up(self):
        # push these names onto tfbeam manager object
        self.sync_trait('user_thresholds_names', self.tfbeam_man,
                        alias='_stats_maps', mutual=False)

    def get_map_by_name(self, name):
        """Given a threshold name, look up the threshold object in
        the user_thresholds list, and return the map
        """
        try:
            names = [t.thresh_map_name for t in self.user_thresholds]
            i = names.index(name)
            return self.user_thresholds[i]._tf_map_scalars
        except ValueError:
            return None
    
    @cached_property
    def _get_map_types(self):
        default_list = ['MEG map']
        if self.stats_results is not None:
            default_list += self.stats_results.threshold_types[:]
        return default_list
            
    def _create_mask_fired(self):
        vox = self.tfbeam_man.beam.voxels
        if self.map_type in ('MEG map', 'Test score'):
            t = SimpleThresholdMask(stats_manager=self,
                                    map_voxels=vox)
        else:
            t = StatsThresholdMask(stats_manager=self,
                                   map_voxels=vox)
        # push tfbeam manager's tf_idx onto the threshold
        self.tfbeam_man.sync_trait('tf_idx', t, alias='_tf_idx', mutual=False)
        t.thresh_map_name = self.map_type
        self.user_thresholds.append(t)

    def _clear_button_fired(self):
        self.threshold = self._threshold_default()
        self.thresh_changed *= -1

    @on_trait_change('user_thresholds[]')
    def _tlist_bookkeeping(self, name, added):
        print 'added:', added, 'in', name
        # if a threshold tab was removed, and it defined the active
        # user mask, then be sure to clear that mask
        mnames = [t.thresh_map_name for t in self.user_thresholds]
        active_name = self.threshold.thresh_map_name
        if active_name and active_name not in mnames:
            self.clear_button = True
        self.user_thresholds_names = mnames

    
    view = View(
        VGroup(
            HGroup(
                Item('map_type', label='Stats Maps', width=50)
                ),
            HGroup(
                Item('clear_button', show_label=False),
                Item('create_mask', show_label=False)
                ),
            HGroup(
                Item('user_thresholds', style='custom',
                     editor=ListEditor(use_notebook=True,
                                       deletable=True,
                                       page_name='.thresh_map_name'))
                )
            ),
        resizable=True,
        title='Stats Thresholding'
        )

tf_axes_to_array = {
    'time' : 1,
    'freq' : 2
    }
array_to_tf_axes = dict( ((v,u) for u, v in tf_axes_to_array.iteritems()) )

class StatsThresholdMask(TimeFreqThresholdMap):
    """ A StatsThresholdMask creates masking parameters based on an
    analysis of significance of a certain test statistic
    """
    stats_results = Instance('nutmeg.stats.tfstats_results.TimeFreqSnPMResults')
    stats_manager = Instance('nutmeg.vis.tfstats_manager.TimeFreqSnPMaps')
    available_maps = Property(depends_on='stats_manager')

    #mask_type = Enum(values='available_maps')
    thresh_map_name = Enum(values='available_maps')

    alpha = Range(low=0.0, high=1.0, value=0.05,
                  editor=RangeEditor(format='%1.3f'))

    _tf_map_scalars = Property(depends_on='thresh_map_name, alpha')
    _tf_thresh_limits = Property(
        depends_on='alpha, thresh_map_name, _pooled_dims, _corrected_dims'
        )
    thresh_mode = Property(depends_on='thresh_map_name')

    pooled_dims = Property
    _pooled_dims = List(
        editor=CheckListEditor(
            values= ['time', 'freq'], cols=1
            )
        )

    corrected_dims = Property
    _corrected_dims = List(
        editor=CheckListEditor(
            values= ['time', 'freq'], cols=1
            )
        )


    mask_button = Button('Apply Stats Mask')

    # XXX: this seems odd.. why not just have it be provided by the
    # constructor?
    @on_trait_change('stats_manager')
    def _get_copy_of_results(self):
        self.stats_results = self.stats_manager.stats_results

    def _get_pooled_dims(self):
        pdims = set([tf_axes_to_array[d] for d in self._pooled_dims])
        cdims = set([tf_axes_to_array[d] for d in self._corrected_dims])
        pdims.difference_update(cdims)
        self._pooled_dims = [array_to_tf_axes[d] for d in pdims]
        return tuple(pdims)

    def _get_corrected_dims(self):
        cdims = set([tf_axes_to_array[d] for d in self._corrected_dims])
        return tuple(cdims)
        
    @cached_property
    def _get_available_maps(self):
        if self.stats_manager and self.stats_manager.map_types:
            l = self.stats_manager.map_types[:]
            l.remove('MEG map')
            l.remove('Test score')
            return l
        return []

    @cached_property
    def _get__tf_map_scalars(self):
        if self.thresh_map_name.find('Test score') >= 0:
            return self.stats_results.t
        elif self.thresh_map_name.find('Cluster') >= 0:
            tail = 'pos' if self.thresh_map_name.lower().find('pos')>=0 \
                   else 'neg'
            cdims = self.corrected_dims
            pdims = self.pooled_dims
            return self.stats_results.map_of_significant_clusters(
                tail, alpha=self.alpha, pooled_dims=pdims, corrected_dims=cdims
                )
        else:
            print 'did not understand map type:', self.thresh_map_name
            return None
    
    @cached_property
    def _get__tf_thresh_limits(self):
        pdims = self.pooled_dims
        cdims = self.corrected_dims
        print pdims, cdims
        if self.thresh_map_name.find('Test score')>=0:
            if self.thresh_map_name.find('both tails')>=0:
                # be careful to test for p < alpha/2 if looking at
                # both tails simultaneously
                spec_alpha = self.alpha/2.0
            else:
                spec_alpha = self.alpha
            # find pos tail cutoff if computing for both tails, or just pos
            if self.thresh_map_name in ('Test score (both tails)',
                                 'Test score (pos tail)'):
                # want to get pos tail cutoff for alpha
                pos_cutoff, alpha = self.stats_results.threshold(
                    spec_alpha, 'pos',
                    pooled_dims=pdims,
                    corrected_dims=cdims
                    )
                pos_cutoff = pos_cutoff[0]
                self.trait_setq(alpha=alpha)
            # find neg tail cutoff if computing for both tails, or just neg
            if self.thresh_map_name in ('Test score (both tails)',
                                 'Test score (neg tail)'):
                neg_cutoff, alpha = self.stats_results.threshold(
                    spec_alpha, 'neg',
                    pooled_dims=pdims,
                    corrected_dims=cdims
                    )
                neg_cutoff = neg_cutoff[0]
                self.trait_setq(alpha=alpha)
            if self.thresh_map_name == 'Test score (both tails)':
                # reset self.alpha to 2x what was returned before
                self.trait_setq(alpha = 2*alpha)
                return np.array( [neg_cutoff, pos_cutoff] )
            if self.thresh_map_name == 'Test score (pos tail)':
                return np.array([pos_cutoff, self.stats_results.t.max(axis=0)])
            if self.thresh_map_name == 'Test score (neg tail)':
                return np.array([self.stats_results.t.min(axis=0), neg_cutoff])
        elif self.thresh_map_name.find('Cluster')>=0:
            ntp, nfp = self.stats_results.t.shape[1:]
            thresh = np.empty((2,ntp,nfp))
            thresh[0] = 0.5
            thresh[1] = 1
            return thresh
    
    @cached_property
    def _get_thresh_mode(self):
        if self.thresh_map_name == 'Test score (both tails)':
            return 'mask between'
        if self.thresh_map_name == 'Test score (neg tail)':
            return 'mask higher'
        else:
            return 'mask lower'

    def _mask_button_fired(self):
        # set all the props in the stats_manager to our properties
        self.map_changed = True
        self.stats_manager.threshold = self
        self.stats_manager.thresh_changed *= -1

    view = View(
        VGroup(
            HGroup(
                Item('mask_button', show_label=False),
                Item('alpha'),
                ),
            HGroup(
                Item(
                    '_pooled_dims', label='Pooled Stats', style='custom'
                    ),
                Item(
                    '_corrected_dims', label='Corrected Stats', style='custom'
                    ),
                Item('unmasked_points',
                     label='Unmasked Points at Current Image',
                     style='readonly')
                )
            )
##         resizable=True
        )
        


class SimpleThresholdMask(TimeFreqThresholdMap):
    """ A SimpleThresholdMask creates masking parameters based on a function
    level comparison. It can only threshold on the MEG activation signal
    and the stats test score.
    """

    stats_manager = Instance('nutmeg.vis.tfstats_manager.TimeFreqSnPMaps')
    available_functions = Property(depends_on='stats_manager')
    thresh_map_name = Enum(values='available_functions')
    tval = Range(low='min_t', high='max_t',
                       editor=RangeEditor(low_name='min_t', high_name='max_t',
                                    format='%1.2f'))
    comp = Enum('greater than', 'less than')
    max_t = Property(depends_on='thresh_map_name')
    min_t = Property(depends_on='thresh_map_name')

    mask_button = Button('Apply Simple Mask')

    _tf_map_scalars = Property(depends_on='stats_manager, thresh_map_name')
    _tf_thresh_limits = Property
    thresh_mode = Property(depends_on='comp')

    view = View(
        VGroup(
            HGroup(
##                 Item('thresh_map_name',
##                      editor=EnumEditor(name='available_functions'),
##                      label='Threshold Type' ),
                Item('comp', label='Mask Out'),
                Item('mask_button', show_label=False)
            ),
                Item('tval', style='custom', label='Threshold Value'),
            )
        )

    @cached_property
    def _get__tf_map_scalars(self):
        if not self.thresh_map_name or \
               not self.stats_manager:
            return np.zeros(1)
        if self.thresh_map_name == 'MEG map':
            return self.stats_manager.tfbeam_man.beam_sig
        if self.stats_manager.stats_results and \
               self.thresh_map_name == 'Test score':
            return self.stats_manager.stats_results.t
    @cached_property
    def _get_max_t(self):
        mx = self._tf_map_scalars.max()
        return int(1000*mx)/1000.0
    @cached_property
    def _get_min_t(self):
        mn = self._tf_map_scalars.min()
        return int(1000*mn)/1000.0
    def _get__tf_thresh_limits(self):
        bcaster = np.ones((self._tf_map_scalars.shape[1:]))
        # if masking greater than tval
        if self.comp=='greater than':
            return np.array([bcaster*self.min_t, bcaster*self.tval])
        # if masking less than tval
        else:
            return np.array([bcaster*self.tval, bcaster*self.max_t])
    @cached_property
    def _get_thresh_mode(self):
        if self.comp=='greater than':
            return 'mask higher'
        else:
            return 'mask lower'
    
    @cached_property
    def _get_available_functions(self):
        if self.stats_manager and self.stats_manager.map_types:
            return ['MEG map', 'Test score']
        return []

    def _mask_button_fired(self):
        self.map_changed = True
        self.stats_manager.threshold = self
        self.stats_manager.thresh_changed *= -1
