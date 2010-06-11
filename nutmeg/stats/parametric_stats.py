import numpy as np
import scipy as sp
import scipy.stats

def one_sample_ttest(beam_comp): #, conditions):
    (sample_beams, comp_avg) = beam_comp.compare() #conditions=conditions)
##     inter_vox = 

    n_tests = len(sample_beams)
    t_scores = []
    p_pos_vals = []
    p_neg_vals = []
    for clabel, comp in zip(beam_comp.conds, sample_beams):
        # sample axis is now 0
        a = np.array([beam.s for beam in comp])
        n = float(a.shape[0])
        df = n - 1
        d = np.mean(a, axis=0)
        v = np.var(a, axis=0, ddof=1)
        t = d / np.sqrt(v/n)
        # prob of score >= t if contrasting same distributions
        ppos = sp.stats.distributions.t.sf(t, df)
        # prob of score <= t if contrasting same distributions
        pneg = sp.stats.distributions.t.sf(-t, df)
        inter_vox = beam_comp.inter_vox[clabel]
        t_beam = beam.from_new_data(
            t, new_vox=inter_vox, multi_subj=True,
            fixed_comparison='parametric T score'
            )
        ppos_beam = beam.from_new_data(
            ppos, new_vox=inter_vox, multi_subj=True,
            fixed_comparison='parametric pval pos tail'
            )
        pneg_beam = beam.from_new_data(
            pneg, new_vox=inter_vox, multi_subj=True,
            fixed_comparison='parametric pval neg tail'
            )
        t_scores.append(t_beam)
        p_pos_vals.append(ppos_beam)
        p_neg_vals.append(pneg_beam)
    return t_scores, p_pos_vals, p_neg_vals
