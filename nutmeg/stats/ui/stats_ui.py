import os
from xipy.vis.qt4_widgets import browse_files, browse_multiple_files, \
     get_saveas_file
## from nutmeg.vis import ortho_plot_window_qt4 as plotter
from nutmeg.stats import beam_stats as bstats
from nutmeg.stats import tfbeam_stats as tfbstats
from nutmeg.core import tfbeam
from nutmeg.vis import plot_tfbeam

import numpy as np
from traits.api import Str, List, HasTraits, Button, File, \
     Instance, Int, Range, Property, Any, Event, Enum, Bool
from traits.api import on_trait_change, cached_property
from traits.ui.api import View, Item, Group, VSplit, \
     VGroup, HGroup, RangeEditor, FileEditor, TextEditor, SetEditor, \
     TabularEditor, InstanceEditor, ListEditor, ListStrEditor, message, \
     spring, CheckListEditor
from traits.ui.api import TableEditor

from traits.ui.table_column \
    import ObjectColumn, ExpressionColumn, ListColumn
from traits.ui.menu \
    import Menu, Action
from traits.ui.tabular_adapter \
    import TabularAdapter

from traits.ui.qt4.tabular_editor import TabularEditorEvent
## from traits.ui.qt4.table_editor import TableEditor

sep_char = os.path.sep

class StringAdapter ( TabularAdapter ):
    
    value_text = Property

    def _get_value_text ( self ):
        return self.item

    def __init__(self, table_name, **kwargs):
        TabularAdapter.__init__(self, **kwargs)
        self.columns = [ (table_name, 'value') ]

class Subject(HasTraits):

    fpath = File
    fname = Str
    name = Str

    @on_trait_change('fpath')
    def _change_fname(self):
        self.fname = os.path.split(self.fpath)[-1]
        print self.fname

##     view = View('fpath')
    view = View('name', 'fpath', 'fname')
    file_view = View('fpath', buttons = ['OK', 'Cancel'])

class Condition(HasTraits):
    subjs = List(Subject)
    cname = Str
    cond = 1
    nsubj = Range(0,200)

    load_many = Button('Load Many Subjs')
    tab_event = Instance(TabularEditorEvent)

    def _cname_default(self):
        return 'cond_'+str(Condition.cond)

    def _subjs_default(self):
        #return [Subject(name='subj%d'%(n+1,)) for n in range(self.nsubj)]
        return []

    def _load_many_fired(self):
        f = browse_multiple_files(None)
        subj_start = self.nsubj+1
        added = len(f)
        new_subjs = [Subject(name='subj%d'%s, fpath=fp)
                     for s, fp in zip(range(subj_start, subj_start+added), f)]
        self.subjs = self.subjs + new_subjs
        self.nsubj += added
                                      

    @on_trait_change('nsubj')
    def _update_list(self):
        while len(self.subjs) > self.nsubj:
            self.subjs.pop(-1)
        while len(self.subjs) < self.nsubj:
            n = len(self.subjs)
            self.subjs.append(Subject(name='subj%d'%(n+1,)))

    @on_trait_change('tab_event')
    def _edit_subject_trait(self):
        if self.tab_event is None:
            return
        subj_attrs = ['name', 'fname', 'fpath']
        print self.tab_event.row, self.tab_event.column
        if self.tab_event.column==2:
            subj = self.subjs[self.tab_event.row]
##             subj.configure_traits(view='file_view', kind='modal')
            subj.fpath = browse_files(None)
        print getattr(self.subjs[self.tab_event.row],
                      subj_attrs[self.tab_event.column])

    def _on_tab_event_fired(self):
        self._edit_subject_trait
    
    def __init__(self, **traits):
        # if incoming args have a subject list, ignore any nsubj spec
        if 'subjs' in traits:
            traits['nsubj'] = len(traits['subjs'])
        HasTraits.__init__(self, **traits)
        print self.cname # force default
        if 'cname' not in traits:
            Condition.cond += 1

    subj_table_editor = TableEditor(
        columns = [ObjectColumn(name='name', label='Subject Name', width=.25),
                   ExpressionColumn(label='Subject File',
                                    width=.5,
                                    expression="object.fpath.split('%s'%sep_char)[-1]"),
                   ObjectColumn(name='fpath', label='File', width=.25,
                                editor=FileEditor(),
                                #editor=InstanceEditor(),
                                style='simple',
                                droppable=True)],
        editable=False,
        sortable=False,
        auto_size=False,
        )

    subj_tabular_editor = TabularEditor(
        adapter=TabularAdapter(
            columns=[ ('Subject Name', 'name'),
                      ('File Name', 'fname'),
                      ('File Path', 'fpath') ],
            can_drop=True
            ),
        dclicked = 'object.tab_event',
        editable = True,
        multi_select = False,
        auto_update = True
        )
        
            
                   

    view = View(VGroup(
        HGroup(Item('cname', style='simple'),
               Item('load_many', show_label=False),
               Item('nsubj', style='simple', label='Number of Subjects')
               )
        ),
        #Item('subjs', editor=subj_table_editor, style='custom'),
        Item('subjs', editor=subj_tabular_editor),
                resizable=True)

class ComparisonResult(HasTraits):
    comp_name = Str
    beams = List(Str)
    beam = Enum(values='beams')
    plot = Button('Plot Beam')
    
    def __init__(self, avgbeam, sampbeams, subjs, **traits):
        HasTraits.__init__(self, **traits)
        self.bdict = dict( zip(subjs, sampbeams) )
        self.bdict['Average'] = avgbeam
        self.beams = list(subjs) + ['Average']

    def _plot_fired(self):
        print 'would plot:', self.beam, self.bdict[self.beam]
        plot_tfbeam(self.bdict[self.beam])

    view = View(
        Item('comp_name', show_label=False, style='readonly'),
        Item('_'),
        Group(
            Item('beam', style='custom'),
            spring,
            Item('plot', show_label=False)
            )
        )
from nutmeg.stats import stats_utils, tfstats_results
class StatsResult(HasTraits):
    comp_name = Str
    _avg_beams = List
    avg_beam = Enum(values='_avg_beams')
    plot_avg = Button('Plot Average')
    save_avg = Button('Save Average')
    # this doesn't actually do anything yet
    stats_maps = Enum(*tfstats_results.TimeFreqSnPMResults.threshold_types)
    plot_stat = Button('Plot Stats Map')
    save_stat = Button('Save Stats Results')
    
    def __init__(self, condition, avg_beams, stats_results, **traits):
        HasTraits.__init__(self, **traits)
        self._avg_dict = {}
        self.stats_results = stats_results
        
        for c, b in zip(condition, avg_beams):
            self._avg_beams.append(str(c))
            self._avg_dict[str(c)] = b

    def _plot_avg_fired(self):
        plot_tfbeam(self._avg_dict[self.avg_beam], stats=self.stats_results)

    def _plot_stat_fired(self):
        pass

    def _save_avg_fired(self):
        f = get_saveas_file(
            None, dialog='Select Filename to Save', wildcard='*.npy'
            )
        f, ext = os.path.splitext(f)
        self._avg_dict[self.avg_beam].save(f)

    def _save_stat_fired(self):
        f = get_saveas_file(
            None, dialog='Select Filename to Save', wildcard='*.npy'
            )
        f, ext = os.path.splitext(f)
        self.stats_results.save(f)
        
    view = View(
        HGroup(
            VGroup(
                Item('avg_beam', label='Average beams', style='custom'),
                Item('plot_avg', show_label=False),
                Item('save_avg', show_label=False)
                ),
            VGroup(
                Item('stats_maps', label='Stats Maps', style='custom'),
                Item('plot_stat', show_label=False),
                Item('save_stat', show_label=False)
                )
            ),
        resizable=True
        )

class SnPMTesterUI(HasTraits):
    comp = Instance('nutmeg.stats.beam_stats.BeamComparator')

    all_conditions = List(Str)
    _ctable_title = Property(Str)
    
    comp_results = List
    stat_results = List

    beam_transform = Enum(*tfbeam.TFBeam.signal_transform_names())
    run_comp = Button('Make Comparison')

    snpm_test = Enum(values='_available_tests')
    _available_tests = Property(List, depends_on='active_conditions')
    _one = Int(1)
    n_perm = Range(low='_one', high='_max_perm', value=1)
    _max_perm = Property(Int, depends_on='snpm_test, active_conditions')
    sym_perms = Bool(True)
    minimum_pval = Property(Str, depends_on='n_perm')
    analyze_clusters = Bool(True)
    cluster_connectivity = Enum(['sides', 'edges', 'corners'])
    cluster_critical_pval = Range(
        low=0.0, high=1.0, editor=RangeEditor(format='%1.4f')
        )
    test_kwargs = Property

    run_stats = Button('Run Stats Test')
    
    def __init__(self, klass, blist, subj_labels, cond_labels):
        self._comp_args = (blist, subj_labels, cond_labels)
        self._comp_class = klass
        HasTraits.__init__(self)
        for c in np.unique(cond_labels):
            self.all_conditions.append(str(c))
        # an ordered list of conditions to use in comparisons
        self.add_trait(
            'active_conditions',
            List(editor=CheckListEditor(cols=1, values=self.all_conditions))
            )

        self.__iscontrast = klass is bstats.BeamContrastAverager
        self.on_trait_change(self._make_comp, 'run_comp', dispatch='new')
        self.on_trait_change(self._make_test, 'run_stats', dispatch='new')

    def _get__ctable_title(self):
        if not self.__iscontrast:
            return 'Select one or more conditions'
        else:
            return 'Select two conditions to contrast'

    def _get_test_kwargs(self):
        kws = dict(analyze_clusters = self.analyze_clusters)
        if self.analyze_clusters:
            connectivity_rule = {
                'sides': 6,
                'edges': 18,
                'corners': 26
                }[self.cluster_connectivity]
            kws['cluster_connectivity'] = connectivity_rule
            kws['cluster_critical_pval'] = self.cluster_critical_pval
        return kws
        
    @cached_property
    def _get__available_tests(self):
        tests = ['One sample T test']
        if (self.__iscontrast and len(self.active_conditions) > 3) or \
           (not self.__iscontrast and len(self.active_conditions) > 1):
            tests += ['Unpaired T test']
        return tests
    
    @cached_property
    def _get__max_perm(self):
        if not self.active_conditions:
            return 1
        _, conds = self._make_comp_params()
        if not conds:
            return 1
        s_labels, c_labels = self._comp_args[1:]
        if self.snpm_test=='One sample T test':
            return tfbstats.SnPMOneSampT.num_possible_permutations(
                conds, c_labels, s_labels
                )
        if self.snpm_test=='Unpaired T test':
            return tfbstats.SnPMUnpairedT.num_possible_permutations(
                conds, c_labels, s_labels
                )

    @cached_property
    def _get_minimum_pval(self):
        if self.sym_perms:
            nperm = self._max_perm
        else:
            nperm = self.n_perm
        return '%1.4f'%(1.0/nperm)
        
    @on_trait_change('_max_perm')
    def _change_n_perm(self):
        self.n_perm = self._max_perm
        
    def _make_comp_params(self):
        if not self.active_conditions:
            return None, None
        if self.__iscontrast:
            n_cond = len(self.active_conditions)
            if n_cond % 2:
                #message('contrast conditions must be listed in pairs')
                return None, None
            conditions = [ [ self.active_conditions[2*n],
                             self.active_conditions[2*n+1] ]
                           for n in xrange(n_cond/2) ]
            cond_titles = [ '%s to %s contrast'%(a,b) for (a,b) in conditions]
        else:
            conditions = self.active_conditions
            cond_titles = [ '%s activation'%a for a in conditions]
        return cond_titles, conditions

    def _make_comp(self):
        cond_titles, conditions = self._make_comp_params()
        if not conditions:
            return
        existing_comps = [c.comp_name for c in self.comp_results]
        print cond_titles
        cond_titles = filter(lambda x: x not in existing_comps, cond_titles)
        print cond_titles
        if not cond_titles:
            return
        kws = dict(fixed_comparison=self.beam_transform)
        comp = self._comp_class(*self._comp_args, **kws)
        print 'testing conditions:', conditions
        samps, avg = comp.compare(conditions=conditions)
        for ab, sb, ctitle in zip(avg, samps, cond_titles):
            self.comp_results.append( ComparisonResult(ab, sb,
                                                       comp.subjs,
                                                       comp_name=ctitle) )
        del comp

    def _make_test(self):
        cond_titles, conditions = self._make_comp_params()
        if not conditions:
            return
        existing_comps = [c.comp_name for c in self.comp_results]
        old_comp_idx = [n for n in xrange(len(conditions))
                        if cond_titles[n] in existing_comps]
        new_comp_idx = [n for n in xrange(len(conditions))
                        if cond_titles[n] not in existing_comps]
        
        if self.snpm_test == 'One sample T test':
            test_class = tfbstats.SnPMOneSampT
        elif self.snpm_test == 'Unpaired T test':
            test_class = tfbstats.SnPMUnpairedT
            conditions = [[a] + [b] for a, b in zip(conditions[0::2],
                                                    conditions[1::2])]
            cond_titles = [[a]+[b] for a, b in zip(cond_titles[0::2],
                                                   cond_titles[1::2])]
            
        kws = dict(fixed_comparison=self.beam_transform)
        comp = self._comp_class(*self._comp_args, **kws)
        for n in xrange(len(conditions)):
            condition = conditions[n]
            c_title = cond_titles[n]

            test = test_class(comp, condition, self.n_perm,
                              force_half_perms=self.sym_perms,
                              init=True)
            
            # if unpaired test, then there are two averages and two conditions
            # if one-samp, then there is one average, and either:
            #  * a contrast pair
            #  * an activation condition
            if self.snpm_test == 'Unpaired T test':
                cond = [ c_title[0], c_title[1] ]
                avgs = test.avg_beams
                ctitle = c_title[0] + ' vs ' + c_title[1]
            else:
                avgs = test.avg_beams
                cond = [ cond_titles[n] ]
                ctitle = cond_titles[n]

            print cond, ctitle
            stats = test.test(**self.test_kwargs)
            
            sresult = StatsResult(cond, avgs, stats, comp_name=ctitle)
            self.stat_results.append(sresult)

    cond_selector = TabularEditor(
        adapter=StringAdapter('Conditions'),
        show_titles=False,
        multi_select=False,
##         selected='object.active_conditions',
        editable=False
        )            
##     cond_selector = ListStrEditor(
##         selected='object.active_conditions',
##         editable=False,
##         multi_select=True
##         )

    view = View(
        VGroup(
            HGroup(
                VGroup(
                    Item('_ctable_title', show_label=False,
                         style='readonly', width=75),
                    Item('active_conditions', show_label=False,
                         style='custom'),
                    Item('active_conditions', show_label=False,
                         editor=cond_selector, width=75)
                    ),
                VGroup(
                    Item('beam_transform', style='custom',
                         label='Signal Transforms'),
                    Item('snpm_test', label='SnPM test type', style='simple'),
                    HGroup(
                        Item('n_perm', label='Number of permutation tests'),
                        Item('sym_perms', label='Symmetric tests',
                             help='Perform symmetrical half of permutations'),
                        Item('minimum_pval', label='Minimum uncorrected P val',
                             help='This is the minimum p score that can be achieved with the current settings',
                             style='readonly'),
                        ),
                    HGroup(
                        Item('analyze_clusters', label='Analyze Clusters'),
                        Group(
                            Item(
                                'cluster_connectivity',
                                label='Connectivity Rule'
                                ),
                            Item(
                                'cluster_critical_pval',
                                label='Defining cluster P-value'
                                ),
                            enabled_when='object.analyze_clusters'
                            )
                        ),
                    HGroup(
                        Item('run_comp', show_label=False),
                        Item('run_stats', show_label=False)
                        )
                    )
                ),
            HGroup(
                Group(Item('comp_results', show_label=False, style='custom',
                           editor=ListEditor(
                               use_notebook=True,
                               page_name='.comp_name'
                               ),
                           height=100
                           ),
                      label='Comparison Results'
                      ),
                Group(Item('stat_results', show_label=False, style='custom',
                           editor=ListEditor(
                               use_notebook=True,
                               page_name='.comp_name'
                               ),
                           height=100
                           ),
                      label='Stats Results'
                      ),
                layout='tabbed'
                )
            ),
        resizable=True,
        title='SnPM Tester'
        )
                

class ListOfConditions(HasTraits):

    conditions = List(Condition, [Condition()])
    add_condition = Button('Append Condition')
    del_condition = Button('Remove Condition')
    selected_condition = Instance(Condition)
    compare_conditions = List(Condition, [])
    contrast_button = Button('Create Contrasts')
    activation_button = Button('Create Activations')

    cond_table_editor = TableEditor(
        columns=[ObjectColumn(label='Conditions',
                              name='cname',
                              width=1,
                              editor=TextEditor(enter_set=True))],
        deletable = True,
        auto_add = False,
        editable=True,
        edit_on_first_click = False,
        show_toolbar = True,
        row_factory = Condition,
        selected='selected_condition'
        )

    cond_tabular_editor = TabularEditor(
        adapter=TabularAdapter(
            columns=[ ('Conditions', 'cname') ]
            ),
        editable=True,
        multi_select=False,
        operations=['insert', 'delete', 'append', 'edit'],
        auto_update=True,
        selected='selected_condition'
        )

    def _add_condition_fired(self):
        self.conditions.append(Condition())
    def _del_condition_fired(self):
        if self.selected_condition:
            # choose an in-bounds index to reset after the list pop
            idx = max(0,
                      min(len(self.conditions)-2,
                          self.conditions.index(self.selected_condition))
                      )
            self.conditions.remove(self.selected_condition)
            if not self.conditions:
                self.selected_condition = None
            else:
                self.selected_condition = self.conditions[idx]
    def _contrast_button_fired(self):
        self._tempcomp = self._build_comparison(bstats.BeamContrastAverager)
    def _activation_button_fired(self):
        self._tempcomp = self._build_comparison(bstats.BeamActivationAverager)

    def _build_comparison(self, klass):
        conds = self.compare_conditions
        # need to find matched lists of
        # -- file paths
        # -- subject labels
        # -- condition labels
        files = []
        clabels = []
        slabels = []
        for c in conds:
            files += [s.fpath for s in c.subjs]
            slabels += [s.name for s in c.subjs]
            clabels += [c.cname] * len(c.subjs)
        #blist = [tfbeam.tfbeam_from_file(f) for f in files]
        tester = SnPMTesterUI(klass, files, slabels, clabels)
        tester.edit_traits()
        self.tester = tester
        return tester

    view = View(
        VGroup(
            HGroup(
                VGroup(
                    Item('conditions', show_label=False,
                         editor=cond_tabular_editor),
                    HGroup(Item('add_condition', show_label=False),
                           Item('del_condition', show_label=False))
                    ),
                Item('selected_condition',
                     show_label=False, style='custom')
                ),
            Item('_'),
            HGroup(
                Item('conditions',
                     editor=TabularEditor(
                         selected='compare_conditions',
                         editable=False,
                         multi_select=True,
                         auto_update=True,
                         adapter=TabularAdapter(
                             columns=[('Available Condtions', 'cname')])
                         ),
                     width=150),
                Item('compare_conditions',
                     editor=TabularEditor(
                         editable=False,
                         auto_update=True,
                         adapter=TabularAdapter(
                             columns=[('Conditions to Compare', 'cname')])
                         ),
                     width=150),
                VGroup(Item('contrast_button', show_label=False),
                       Item('activation_button', show_label=False))
                )
            ),
        title='Table of Conditions And Subjects',
        resizable = True
        )



def _main(mode='real'):
    condition_1_spec = [
    ('subj1', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub1/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj2', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub2/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj3', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub3/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj4', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub4/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj5', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub5/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj6', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub6/s_beamtf_listenfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ]
    condition_2_spec = [
    ('subj1', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub1/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj2', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub2/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj3', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub3/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj4', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub4/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj5', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub5/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ('subj6', '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/sub6/s_beamtf_speakfirst75Lhemi_firlsbp200_bf_SAM_all_spatnorm.mat'),
    ]
    if mode=='real':
        c1 = Condition(subjs=[Subject(name=nm, fpath=fp)
                              for nm, fp in condition_1_spec],
                       cname='listen_first')
        c2 = Condition(subjs=[Subject(name=nm, fpath=fp)
                              for nm, fp in condition_2_spec],
                       cname='speak_first')
        demo = ListOfConditions(conditions=[c1, c2])
        return demo
    elif mode=='synth_contrast':
        from nutmeg.stats.tests import synth_activations
        n_subjs = 6
        def save_beams(bl, sl, cl):
            pl = []
            for b, s, c in zip(bl, sl, cl):
                p = 'synth_beam_s%d_c%d'%(s, c)
                pl.append(os.path.abspath(p)+'.npy')
                b.save(p)
            return pl
        beam = tfbeam.tfbeam_from_file(condition_1_spec[0][1],
                                       fixed_comparison='F dB')
        c1b = synth_activations.synthetic_gaussian_activity_like(
            beam, n_beams=n_subjs,
            tf_pts=[ (t,1) for t in xrange(4,13) ],
            locs = ( (-20,-30,-30), (-15,3,15), (-18,15,0) ),
            size=14.,
            mode='activation'
            )
        c1_subjs = range(1,n_subjs+1)
        c1_labels = [1] * n_subjs
        c1_paths = save_beams(c1b, c1_subjs, c1_labels)

        c2b = synth_activations.synthetic_gaussian_activity_like(
            beam, n_beams=n_subjs,
            tf_pts=[ (t,1) for t in xrange(4,13) ],
            locs = ( (-20,-30,-30), (-15,3,15), (-18,15,0) ),
            size=14.,
            mode='activation', pval=.1
            )
        c2_subjs = range(1,n_subjs+1)
        c2_labels = [2] * n_subjs
        c2_paths = save_beams(c2b, c2_subjs, c2_labels)

        c3b = synth_activations.synthetic_gaussian_activity_like(
            beam, n_beams=n_subjs, mode='activation',
            tf_pts=[ (t,1) for t in xrange(4,13) ],
            locs = ( (-20,-30,-30), (-15,3,15), (-18,15,0) ),
            size=14.            
            )
        c3_subjs = range(1,n_subjs+1)
        c3_labels = [3] * n_subjs
        c3_paths = save_beams(c3b, c3_subjs, c3_labels)
        
        c4b = synth_activations.synthetic_gaussian_activity_like(
            beam, n_beams=n_subjs, mode='activation', pval=.6,
            tf_pts=[ (t,1) for t in xrange(4,13) ],
            locs = ( (-20,-30,-30), (-15,3,15), (-18,15,0) ),
            size=14.            
            )
        c4_subjs = range(1,n_subjs+1)
        c4_labels = [4] * n_subjs
        c4_paths = save_beams(c4b, c4_subjs, c4_labels)

        conds = [ Condition(subjs=[Subject(name=str(nm), fpath=fp)
                                   for nm, fp in zip(subjs, paths)],
                            name=cname)
                  for subjs, paths, cname in zip( [c1_subjs, c2_subjs,
                                                   c3_subjs, c4_subjs],
                                                  [c1_paths, c2_paths,
                                                   c3_paths, c4_paths],
                                                  ['1', '2', '3', '4'] ) ]
        demo = ListOfConditions(conditions=conds)
        
        return demo

condition_args = (['/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_DS_01_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_HJ_07_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_HL_02_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_MA_01_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_MP_04_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_NS_01_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_NS_10_RD2_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_RB_01_RD2_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_WC_01_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_YC_motor03_firlsbp200cn_SAM_all_spatnorm.mat',
  '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf_ZJ_01_firlsbp200cn_SAM_all_spatnorm.mat'],
 ['subj1',
  'subj2',
  'subj3',
  'subj4',
  'subj5',
  'subj6',
  'subj7',
  'subj8',
  'subj9',
  'subj10',
  'subj11'],
 ['cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1',
  'cond_1']
                  )


if __name__=='__main__':
    import sys
    mode = sys.argv[1]
    demo = _main(mode=mode)
    demo.edit_traits()
##     demo = ListOfConditions(conditions=[c1,c2])
##     demo.configure_traits()

