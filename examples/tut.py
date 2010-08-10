import numpy as np
from nutmeg.stats import beam_stats as bstats
from nutmeg.stats import tfbeam_stats
import os

# Little utility to walk the directory tree and find all "*spatnorm.mat"
def find_spatnorm_mats(pth):
    def is_spatnorm_mat(f):
        return f.find('spatnorm.mat') > 0
    snorms = []
    for root, dirs, files in os.walk(pth):
        snorms += [os.path.join(root, f)
                   for f in filter(is_spatnorm_mat, files)]
    return snorms

# Little utility to segment the file paths according to conditions
def split_conds(c1, c2, files):
    def is_c1(f):
        return f.find(c1) > 0
    def is_c2(f):
        return f.find(c2) > 0
    return filter(is_c1, files), filter(is_c2, files)

cond1 = 'listenfirst'
cond2 = 'speakfirst'

spatnorms = find_spatnorm_mats('meg_6subj')

cond1_beams, cond2_beams = split_conds(cond1, cond2, spatnorms)

# Assuming a path structure like /some/path/subject{n}/spatnormfile.mat
# Split file path 2ce off the end to get the subject name
subj_labels = [
    os.path.split(
        os.path.split(f)[0]
        )[1] for f in cond1_beams
    ]
n_subj = len(subj_labels)

# Now arange the arguments to form a BeamContrastAverager
all_beams = cond1_beams + cond2_beams
all_subjs = subj_labels * 2 # list-arithmetic: repeats the subject list twice
all_conds = [cond1] * n_subj + [cond2] * n_subj
# Use a "fixed_comparison" to cut down on the memory footprint of all
# the loaded TFBeams
bcomp = bstats.BeamContrastAverager(
    all_beams, all_subjs, all_conds, fixed_comparison='f db'
    )

# the TFBeams are loaded on call.. the first time the bcomp.beam attribute
# is requested

# For Example, with the BeamComparators, you can index a beam by
# condition and subject label
beam_c1_s3 = bcomp.beam(cond1, subj_labels[2])

# This function returns one list of sample beams and one list of average beams
# per conditions compared. There is only one pair of conditions compared here,
# so len(samps) and len(avg) are both == 1
samps, avg = bcomp.compare()
samps = samps[0]
avg = avg[0]

# This TFBeam has its voxels pruned to the intersection of all compared
# TFBeams, and its signal is aligned voxel-wise with all the other samples
s0 = samps[0]
s0.coreg.mrpath # <-- probably points to a path on the RAID
s0.coreg.mrpath = os.path.abspath('../data/mri/ChadickZack/wChadickZack_V2.hdr')
# can save and view s0
s0.save('tfbeam0')

tester = tfbeam_stats.SnPMOneSampT(bcomp, [cond1, cond2], 10)
stats_res = tester.test(analyze_clusters=True)

# Make a maximal-stats corrected threshold at alpha = .5.. true_alpha is
# based on the actual level of the empirical CDF, which is a function of
# the number of bins in the empirical distribution
threshold, true_alpha = stats_res.threshold(.5, 'neg')

# this is the map of the t test that survives the threshold
surviving_t_map = stats_res.t < threshold
vox_map_at_tf_10_1 = stats_res.vox_idx[surviving_t_map[:,10,1]]

p_map = stats_res.p_score_from_maximal_statistics(.5, 'pos')

avg.save('avgbeam')
stats_res.save('onesampTstats')
