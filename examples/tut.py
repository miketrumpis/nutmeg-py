from nutmeg.stats import beam_stats as bstats
bstats.BeamContrastAverager.from_matlab_ptr_file('meg_6subj/PM_HL_OJ_CZ_TC_WK_first75L_new_ptr.mat', fixed_comparison='F dB')
bcomp = bstats.BeamContrastAverager.from_matlab_ptr_file('meg_6subj/PM_HL_OJ_CZ_TC_WK_first75L_new_ptr.mat', fixed_comparison='F dB')


samps, avg = bcomp.compare()


s0 = samps[0][0]
s0.coreg.mrpath
import os
s0.coreg.mrpath = os.path.abspath('../data/mri/ChadickZack/wChadickZack_V2.hdr')
s0.save('tfbeam0')
# can save and view s0

tester = tfbeam_stats.SnPMOneSampT(bcomp, [1,2], 10)
stats_res = tester.test(analyze_clusters=True)

threshold, alpha = stats_res.threshold(.5, 'neg')
p_map = stats_res.p_score_from_maximal_statistics(.5, 'pos')

avg.save('avgbeam')
stats_res.save('onesampTstats)
