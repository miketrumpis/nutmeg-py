==================
Beam Stats Example
==================

The :mod:`~nutmeg.stats.beam_stats` module in Nutmeg-Py provides objects to align data across subjects and conditions, and then perform non-parametric statstical analyses on the data comparisons.

Comparing Data
^^^^^^^^^^^^^^

Comparing data starts with the :class:`~nutmeg.stats.beam_stats.BeamComparator` type. The common functionality of a BeamComparator is:

* to align voxels across subjects (and potentially across conditions)
* to index beams by (condition, subject)
* to index beam signals by (condition, subjet), and return the signals aligned voxel-wise across subjects
* to return per-subject samples and an average, according to the comparison being made

There are two flavors of comparison, the :class:`~nutmeg.stats.beam_stats.BeamActivationAverager`, and the :class:`~nutmeg.stats.beam_stats.BeamContrastAverager`. The BeamActivationAverager only looks at functional activation (some transform of the active-to-control ratio), while the BeamContrastAverager looks at the difference in activations across two conditions.

Here are some examples for constructing comparators::

    >>> from nutmeg.stats import beam_stats as bstats
    >>> from glob import glob

A BeamComparator can be constructed either with a list of paths to tfbeam mat files, or with actual :class:`~nutmeg.core.tfbeam.TFBeam` objects. Choosing paths constructs the comparator quickly, since the data loads only on call.
::

    >>> bfiles = glob('NC_RightDom_Digits_New/s_beamtf*spatnorm.mat')
    >>> import os
    >>> subjs = [os.path.split(b)[1].split('_')[2] for b in bfiles]
    >>> subjs
    ['DS', 'HJ', 'HL', 'MA', 'MP', 'NS', 'NS', 'RB', 'WC', 'YC', 'ZJ']
    >>> conds = ['cond1'] * len(subjs)
    >>> conds
    ['cond1', 'cond1', 'cond1', 'cond1', ...]

To cut down on the size in memory of all the subject TFBeams, I'll choose to fix the active-to-control ratio to be 'F dB'::

    >>> comp = bstats.BeamActivationAverager(bfiles, subjs, conds, fixed_comparison='F dB')

Now it's possible  to perform the comparison. The :meth:`~nutmeg.stats.beam_stats.BeamComparator.compare` method returns the per-subject samples and the across subject average for each condition configuration given. By default, it forms comparisons on all the conditions available. For example, the :meth:`~nutmeg.stats.beam_stats.BeamActivationAverager.compare` method defined on BeamActivationAverager will returns comparisons for all conditions by default::

    >>> samples, avgs = comp.compare()
    >>> len(samples), len(avgs)
    (1, 1)
    >>> cond1_avg = avgs[0]
    >>> cond1_samps = samples[0]

From Matlab-Nutmeg Pointers
***************************

*This functionality is not fully implemented for the RAID environment!*

It is also possible to construct comparators from "pointers" created in Matlab-Nutmeg::

    >>> comp = bstats.BeamContrastAverager.from_matlab_ptr_file('meg_6subj2cond/PM_HL_OJ_CZ_TC_WK_first75L_new_ptr.mat', fixed_comparison='F dB')

SnPM Stats Analysis
^^^^^^^^^^^^^^^^^^^

The :mod:`~nutmeg.stats.tfbeam_stats` module also includes a number of objects of the general type :class:`~nutmeg.stats.tfbeam_stats.SnPMTester`. These objects are created using a BeamComparator and perform a test on the condition, or conditions pair specified.

**Note: the currently implementation of these tests requires that the null distribution is mean zero and symmetrical! The "F dB" comparison satisfies this assumption**

There are two subclasses of SnPMTester implemented: 

:class:`~nutmeg.stats.tfbeam_stats.SnPMOneSampT`

* Compare group activity or group contrast to the null distribution
* The condition argument must be a single condition label (when using a BeamActivationAverager), or a conditions contrast pair (when using a BeamContrastAverager)

:class:`~nutmeg.stats.tfbeam_stats.SnPMUnpairedT`

* Compare two groups of activity or contrast to each other
* The condition argument must be two condition labels (when using a BeamActivationAverager), or two conditions contrast pairs (when using a BeamContrastAverager)

Using the BeamContrastAverager "comp" from above, I will set up a SnPMOneSampT tester. When setting the number of permutations, there are a few options. To find out how many permutations are possible, you can use this method (defined on both classes)::

    >>> from nutmeg.stats import tfbeam_stats as tfbstats
    >>> tfbstats.SnPMOneSampT.num_possible_permutations(comp.conds, comp.c_labels, comp.s_labels)
    64

This number effectively determines the courseness of the estimation of the null distribution, and more permutations provide greater significance. Of course, the testing is slower for more permutations. 

There is a possible savings of a factor of 2 when testing all possible permutations, since half of the re-weightings are mirrors of the other half. To enforce testing in this mode, you can set the flag "force_half_perms" in the tester constructor, like this (in this case, the n_perm argument is ignored)::

    >>> comp.conds
    array([1, 2])
    >>> tester = tfbstats.SnPMOneSampT(comp, [1,2], 1234, force_half_perms=True)
    >>> tester.n_perm
    64

If truncating the number of permutations, then the statistical test reweightings will be presented in random order::

    >>> tester = tfbstats.SnPMOneSampT(comp, [1,2], 20)
    >>> tester.n_perm
    20

Results
*******

To make the test, simply run the "test" method on the SnPMTester::

    >>> results = tester.test()
    performing stat test at (t,f) = ( 0 0 )
    performing stat test at (t,f) = ( 0 1 )
    performing stat test at (t,f) = ( 0 2 )
    ...
    >>> results
    <nutmeg.stats.tfstats_results.TimeFreqSnPMResults object at 0x33c9ff0>

The results object is a :class:`~nutmeg.stats.tfstats_results.TimeFreqSnPMResults` object. This object holds both the maximal and minimal stats distributions, and the test's true statistic from the data. With this information, I can perform the following tasks:

:meth:`~nutmeg.stats.tfstats_results.TimeFreqSnPMResults.threshold`, example::

    >>> tmap, alpha = results.threshold(.23, 'neg')
    >>> alpha
    0.20000000000000001
    >>> map = results.t <= tmap
    >>> tmap, alpha = results.threshold(.23, 'neg', corrected_dims=(1,2))
    >>> tmap
    array([[[-6.9315619, -6.9315619, -6.9315619, -6.9315619],
	    [-6.9315619, -6.9315619, -6.9315619, -6.9315619], ....

:meth:`~nutmeg.stats.tfstats_results.TimeFreqSnPMResults.uncorrected_p_score`, example::

    >>> pmap = results.uncorrected_p_score('pos')
    >>> (pmap < .15).any()
    True
    >>> (pmap < .15).sum()
    34513

:meth:`~nutmeg.stats.tfstats_results.TimeFreqSnPMResults.p_score_from_maximal_statistics`, example::

    >>> pmap = results.p_score_from_maximal_statistics('pos')
    >>> (pmap < .15).sum()
    35
    >>> pmap = results.p_score_from_maximal_statistics('pos', corrected_dims=(1,))
    >>> pmap.shape
    >>> (pmap < .15).sum()
    0
    >>> pmap = results.p_score_from_maximal_statistics('pos', pooled_dims=(1,))
    >>> (pmap < .15).sum()
    141

:meth:`~nutmeg.stats.tfstats_results.TimeFreqSnPMResults.map_of_significant_clusters`, example::

    >>> (results.uncorrected_p_score('pos') > .15).any()
    True
    >>> cmap = results.map_of_significant_clusters('pos', alpha=.15, gamma=.01)
    >>> mask = cmap > 0
    >>> mask.sum()
    9976

Coercing Matlab-Nutmeg Results
******************************

There is a subclass of TimeFreqSnPMResults called :class:`~nutmeg.stats.tfstats_results.AdaptedTimeFreqSnPMResults`. This class attempts to reconstruct the maximal statistic distribution based on the true statistic function and the corresponding corrected p values. The method :meth:`~nutmeg.stats.tfstats_results.adapt_mlab_tf_snpm_stats` will return an AdaptedTimeFreqSnPMResults object, given a path to a mat file.

Viewing The Results
^^^^^^^^^^^^^^^^^^^

::

    >>> from nutmeg.vis import plot_tfbeam
    >>> avg = tester.avg_beams[0]
    >>> plot_tfbeam(avg, stats=results, with_3d=True)


Saving/Loading The Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    >>> avg.save('cond12_contrast_avg')
    >>> results.save('cond12_contrast_stats')
    >>> ls
    cond12_contrast_avg.npy		cond12_contrast_stats.npy	stats_tutorial.py
    >>> res2 = tfstats_results.load_tf_snpm_stats('cond12_contrast_stats.npy')
    >>> from nutmeg.core import tfbeam
    >>> avg2 = tfbeam.tfbeam_from_file('cond12_contrast_avg.npy')

