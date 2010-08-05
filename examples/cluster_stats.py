from glob import glob
from nutmeg.stats import beam_stats as bstats
from nutmeg.stats import tfbeam_stats
from nutmeg.stats import tfstats_results


bfiles = glob('../data/NC_RightDom_Digits_New/*spatnorm.mat')
conds = [1] * len(bfiles)
subjs = range(1, len(bfiles)+1)

comp = bstats.BeamActivationAverager(
        bfiles, subjs, conds, fixed_comparison='F dB'
    )

tester = tfbeam_stats.SnPMOneSampT(comp, [1], -1, force_half_perms=True)
## tester = tfbeam_stats.SnPMOneSampT(comp, [1], 200, force_half_perms=False)
sres = tester.test(analyze_clusters=True,
                   cluster_critical_pval=.001,
                   cluster_connectivity=18)
avg = tester.avg_beams[0]
avg.save('avgbeam')
sres.save('stats_with_clusters')

tfstats_results.quick_plot_top_n(sres, 'neg', n=10)
