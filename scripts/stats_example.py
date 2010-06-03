#!/usr/bin/env python
from nutmeg.core.tfbeam import tfbeam_from_file
from nutmeg.stats import beam_stats as bstat
import glob
import os, sys

def perform_activation_ttest(dir=None, ptr=None):
    if dir is not None:
        glob_str = os.path.join(dir, 's_beam*spatnorm*mat')
        mat_files = glob.glob(glob_str)
        # get all beams in dir, with subject and condition labels for each beam
        beam_list = [tfbeam_from_file(mfile, fixed_comparison='F dB')
                     for mfile in mat_files]
        subj_labels = range(1,len(beam_list)+1)
        cond_labels = [1] * len(subj_labels)

        a_comp = bstat.BeamActivationAverager(beam_list,
                                              subj_labels,
                                              cond_labels)
    elif ptr is not None:
        a_comp = bstat.BeamActivationAverager.from_matlab_ptr_file(ptr,
                                                            ratio_type='F dB')

    # do an activation test on condition 1
    one_samp_t = bstat.SnPMOneSampT(a_comp, 1, 0, force_half_perms=True)
    s_res = one_samp_t.test()
    return s_res, one_samp_t.avg_beams[0]

if __name__=='__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]

    if os.path.splitext(arg)[1] == '.mat':
        results, avgbeam = perform_activation_ttest(ptr=arg)
        dir = os.environ['PWD']
    else:
        results, avgbeam = perform_activation_ttest(dir=arg)

    results.save('test_results')
    avgbeam.save('avg_activation')
        
