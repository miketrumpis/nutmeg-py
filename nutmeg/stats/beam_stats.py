import numpy as np
import scipy.io as sio
import os

import nutmeg.external.descriptors as desc
from nutmeg.core import beam_mni_box
from nutmeg.core import tfbeam as tfb

def find_vox_intersection(vox_lists, trimmed_boundary=None):
    """
    Find the interesction of all voxel coordinates in vox_lists.
    Optionally, trim all voxels to be within the specified boundary

    Parameters
    ----------
    vox_lists :
      is an iterable container of (nvox x ncoord) arrays

    trimmed_boundary : iterable, optional
      a list of (min, max) pairs for the X, Y, Z dimensions. If given,
      then each coordinate component must be inside its range.

    Returns
    -------
    inter_vox : ndarray
      the (len(intersection) x 3) array of intersecting voxels
    vox_idx : list
      the original index of each voxel coordinate in its original list
    """
    if len(vox_lists) < 2:
        return vox_lists[0], range(len(vox_lists[0]))
    # create the lookup dictionaries for later
    lookups = []

    def in_bounds(vx):
        if trimmed_boundary is None:
            return True
        b = True
        for i in xrange(3):
            bounds = trimmed_boundary[i]
            b = b and (vx[i] >= bounds[0] and vx[i] <= bounds[1])
        return b
    
    for vlist in vox_lists:
        # only count the voxel if it's in_bounds
        pairs = ( (tuple(vx), n) for (n, vx) in enumerate(vlist)
                  if in_bounds(vx) )
        d = dict(pairs)
        lookups.append(d)

    # don't bother to create the (i,j,k) lists again..
    # now they are held in the dictionary keys
    inter_set = set(lookups[0].keys())
    for vlist in map(lambda x: getattr(x, 'keys'), lookups[1:]):
        inter_set.intersection_update(set(vlist()))

    # make the vox coord array
    inter_vox = [ [i, j, k] for (i, j, k) in inter_set]
    inter_vox.sort()
    inter_vox = np.array(inter_vox, 'i')
    # now go through each lookup and find where inter_set voxels occur
    vox_idx = []
    for lu in lookups:
        vox_hits = [lu[tuple(vx)] for vx in inter_vox]
        #vox_hits.sort()
        vox_idx.append(vox_hits)
    return inter_vox, vox_idx

class BeamComparator(object):
    """
    This class will manage beam comparisons among conditions or across
    conditions. For any comparison specified, it will prune the voxels in
    each beam such that all voxels intersect with each other, and also with
    the MNI voxel space.
    
    """

    def __init__(self, beam_list, subject_labels, condition_labels,
                 aligned_voxels=(), **b_kws):
        """
        Parameters
        ----------
        beam_list : list
            a length-n list of beam objects of paths to beam files
        subject_labels : list
            a length-n list of labels (num, str, whatever) labeling each beam
        condition_labels : list
            a length-n list of labels (num, str, whatever) labeling each beam
        b_kws : dict, optional
            if the beam_list is supplied as a list of paths, then these are
            optional arguments to be used when loading the beams
        """
        self._compare_cache = {}
        self.subjs = np.unique(subject_labels)
        self.conds = np.unique(condition_labels)
        self.n_subj = len(self.subjs)
        self.n_cond = len(self.conds)
        self.s_labels = np.array(subject_labels)
        self.c_labels = np.array(condition_labels)
        # If the beam_list is a bunch of paths, defer loading until later
        if type(beam_list[0]) is str:
            self._blist = beam_list
            self._loaded = False
            self._b_kwargs = b_kws
        else:
            self._loaded = True
            self._blist = beam_list
            self.__beam_sanity_check(self.beams)

        self.aligned = False

    def __beam_sanity_check(self, blist):
        vs = blist[0].voxelsize
        same_vox = all(filter(lambda x: (vs==x.voxelsize).all(), blist[1:]))
        sane_length = len(blist)==self.n_subj*self.n_cond
        if not same_vox:
            raise ValueError('all beams must have the same voxel resolution')
        if not sane_length:
            raise ValueError('there must be a beam for every (subj,cond) combo')

    def clear_cache(self):
        self._compare_cache = {}

    @desc.auto_attr
    def beams(self):
        if self._loaded:
            bl = self._blist
            del self._blist
            return bl
        bl = [tfb.tfbeam_from_file(f, **self._b_kwargs) for f in self._blist]
        self.__beam_sanity_check(bl)
        del self._blist
        return bl
        
    def align_voxels(self, align_all=False):
        """
        The "prototype" of align_voxels() aligns voxels across
        subjects at each condition. All resulting data are dicts keyed
        by condition.

        Parameters
        ----------
        align_all : bool
            Aligns voxels across all beams in the comparison group, rather
            than separately per condition

        (all these items get defined on the object as a result of this method)
        inter_vox : dict
            lists of intersecting voxels
        beam_voxel_maps : dict
            lists of index locations of intersecting voxels on the ordered
            voxels list of each beam.
        """

        vs = self.beams[0].voxelsize
        self.inter_vox = {}
        self.beam_voxel_maps = {}
        self.aligned = True
        if align_all:
            subj_vox = [b.voxels for b in self.beams]
            ivox, bvox_map = find_vox_intersection(subj_vox)
            # just repeat the same data over conditions
            # XYZ: THIS COULD BE RE-ENGINEERED FOR REDUNDANCY
            for c in self.conds:
                cmap = self.c_labels == c
                self.inter_vox[c] = ivox
                self.beam_voxel_maps[c] = [bvox_map[cidx]
                                           for cidx, b in enumerate(cmap) if b]
            return
        for c in self.conds:
            print 'condition', c, 'comparing subjs: ',
            subj_vox = [self.beam(c,i).voxels for i in self.subjs]
            self.inter_vox[c], self.beam_voxel_maps[c] = \
                            find_vox_intersection(subj_vox,
                                                  trimmed_boundary=beam_mni_box)
            print 'n_vox:', len(self.inter_vox[c])

    def compare(self, conditions=[]):
        """
        prototype
        
        Returns
        -------
        sample_beams : list
            a len(conditions) length list of n_subj length lists of
            comparison beams
        avg_beams : list
            a len(conditions) length list of average comparisons
        """
        pass

    @classmethod
    def from_matlab_ptr_file(class_type, mfile, **kwargs):
        """
        This method creates a BeamComparator-type object from a MATLAB
        (or NUTMEG) pointer file.

        ex: (a shallow sublass of BeamComparitor)
        
        class MyBeamComparison (BeamComparitor):
            pass

        c = MyBeamComparison.from_matlab_ptr_file(m_file, ratio_type='f raw')
        results = c.compare()

        Parameters
        ----------
        mfile : string
            the .mat file that holds subject/condition labels and paths to
            the tfbeam mat files

        kwargs : dict
            keyword arguments for constructing TFBeams
        """        
        mat_path = os.path.abspath(mfile)
        bpath, mpath = os.path.split(mat_path)
        voi = sio.matlab.loadmat(mat_path, struct_as_record=True)['voi'][0,0]
        # this will need to be fixed for the RAID environment
        pnames = [os.path.join(bpath, str(p[0]))
                  for p in voi['pathnames'][0]]
        subj_labels = voi['subjnr'][0].astype('i')
        cond_labels = voi['condnr'][0].astype('i')
        beam_list = []
        for p in pnames:
            if p.find('spatnorm') < 0:
                beam_list.append(os.path.splitext(p)[0] + '_spatnorm.mat')
            else:
                beam_list.append(p)
        
        return class_type(beam_list, subj_labels, cond_labels)


    def beam(self, cond, subj):
        """
        Returns the beam object corresponding to a given condition and subject.
        """
        c_map = self.c_labels == cond
        s_map = self.s_labels == subj
        if not (c_map & s_map).sum():
            raise ValueError('no beam with (cond, subj)=(%d, %d)'%(cond, subj))
        if (c_map & s_map).sum() > 1:
            raise ValueError('Beams not uniquely specified: ambiguously labeled BeamComparator')
        beam_idx = np.argwhere(c_map & s_map)
        return self.beams[beam_idx]
        
    def beam_sig(self, cond, subj):
        """
        Returns the signal from the beam indexed by (cond, subj).
        Importantly, returned samples of the function s(vox) ordered in
        a consistent voxel ordering
        """
        beam = self.beam(cond, subj)
        c_map = self.c_labels == cond
        s_map = self.s_labels[c_map]
        beam_where = s_map==subj
        if beam_where.sum() < 1:
            err = 'Subject '+str(subj)+' has no map for condition '+str(cond)
            raise ValueError(err)
        if beam_where.sum() > 1:
            raise ValueError('Beams not uniquely specified: ambiguously labeled BeamComparator')
        beam_idx = np.argwhere(beam_where)
        m = self.beam_voxel_maps[cond][beam_idx]
        return beam.s[m]

class BeamActivationAverager(BeamComparator):
    """
    This BeamComparator looks at the activation in a given condition for
    a group of subjects.
    """

    def beam_sig(self, cond, subj):
        """
        Returns the signal from the beam indexed by (cond, subj).
        Importantly, returned samples of the function s(vox) ordered in
        a consistent voxel ordering
        """
        b = self.beam(cond, subj)
        m_idx = np.argwhere(self.subjs==subj).flat[0]
        m = self.beam_voxel_maps[cond][m_idx]
        return b.s[m]

    def compare(self, conditions=[]):
        """
        Returns a list of activations, and an average activation for
        each condition specified
        """
        if not hasattr(conditions, '__iter__'):
            conditions = [conditions]
        if not conditions:
            conditions = self.conds
        if not self.aligned:
            self.align_voxels()
##         if tuple(conditions) in self._compare_cache:
##             return self._compare_cache[tuple(conditions)]
##         # otherwise make new comparison
        avg_beams = []
        act_beams = []
        for c in conditions:
            condition_act_beams = []
            avg_sig = np.zeros_like(self.beam_sig(c,self.subjs[0]))
            ivox = self.inter_vox[c]
            for i in self.subjs:
                b = self.beam(c,i)
                b_sig = self.beam_sig(c,i)
                avg_sig += b_sig
                pruned_beam = b.from_new_data(b_sig, new_vox=ivox,
                                              fixed_comparison=b.uses)
                condition_act_beams.append(pruned_beam)
            avg_sig /= self.n_subj
            avg_beams.append(b.from_new_data(avg_sig,
                                             new_vox=ivox,
                                             multi_subj=True,
                                             fixed_comparison=b.uses))
            act_beams.append(condition_act_beams)

##         self._compare_cache[tuple(conditions)] = (tuple(act_beams),
##                                                   tuple(avg_beams))
        return act_beams, avg_beams
            

class BeamContrastAverager(BeamComparator):
    """
    This BeamComparator looks at the contrast between conditions for a
    group of subjects.
    """

    # Uses default voxel alignment across condition and subject

    def compare(self, conditions=[]):
        """
        Returns a contrast beam for each subject, for each condition pair
        specified, as well as an average contrast beam for each condition pair.
        If the conditions are specified as [(a,b)], the contrast is
        beam(a, subj) - beam(b, subj)
        """
        if not conditions:
            # default to pairing off conditions in order???
            conditions = self.conds.reshape(self.n_cond/2, 2).tolist()
        if not hasattr(conditions[0], '__iter__'):
            conditions = [ conditions ]
        if len(filter(lambda x: len(x)==2, conditions)) != len(conditions):
            raise ValueError(
    """Some of the provided condition specs are not in pairs %s"""%repr(conditions)
        )
        if not self.aligned:
            self.align_voxels()

            
##         if tuple(conditions) in self._compare_cache:
##             return self._compare_cache[tuple(conditions)]
##         # otherwise make new comparison            
        avg_beams = []
        contrast_beams = []
        for pair in conditions:
            c1, c2 = pair
            avg_contrast = np.zeros_like(self.beam_sig(self.conds[0],
                                                       self.subjs[0]))
            cond_contrasts = []
            b1 = self.beam(c1,self.subjs[0])
            for i in self.subjs:
                s = self.beam_sig(c1,i) - self.beam_sig(c2,i)
                # XYZ: INCONVENIENT SYNTAX FOR FINDING THE CORRECT INTER_VOX
                cond_contrasts.append(
                    b1.from_new_data(s,
                                     new_vox=self.inter_vox[pair[0]],
                                     fixed_comparison=b1.uses)
                    )
                avg_contrast += s
            contrast_beams.append(cond_contrasts)
            avg_contrast /= self.n_subj
            avg_beams.append(
                b1.from_new_data(avg_contrast,
                                 new_vox=self.inter_vox[pair[0]],
                                 multi_subj=True,
                                 fixed_comparison=b1.uses)
                )
                
##         self._compare_cache[tuple(conditions)] = (tuple(act_beams),
##                                                   tuple(avg_beams))
        return contrast_beams, avg_beams
    
    def align_voxels(self, **kw):
        # silently ignore the kwarg
        super(BeamContrastAverager, self).align_voxels(align_all=True)
        

            
        
        
