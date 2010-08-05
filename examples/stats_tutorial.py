from glob import glob
from nutmeg.stats import beam_stats as bstats
from nutmeg.stats import tfbeam_stats

# -- Example 1, from "pointer"
def run1():
    m_path = '/Users/mike/workywork/nutmeg-py/data/meg_6subj2cond/PM_HL_OJ_CZ_TC_WK_first75L_new_ptr.mat'

    comp = bstats.BeamContrastAverager.from_matlab_ptr_file(
        m_path, fixed_comparison='F dB'
        )
    tester = tfbeam_stats.SnPMOneSampT(comp, [1,2], 20)

    results = tester.test()
    avg = tester.avg_beams
    return avg, results

# -- Example 2, from path list
def run2():
    bfiles = glob('/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/s_beamtf*spatnorm.mat')
    cond_labels = [1] * len(bfiles)
    subj_labels = range(len(cond_labels))
    comp = bstats.BeamActivationAverager(
        bfiles, subj_labels, cond_labels, fixed_comparison='F dB'
        )

    tester = tfbeam_stats.SnPMOneSampT(comp, [1], 20)
    results = tester.test()
    avg = tester.avg_beams
    return avg, results


# -- Example 3, from GUI
def run3():
    from nutmeg.stats.ui import stats_ui
    clist = stats_ui.ListOfConditions()
    clist.edit_traits()
    
