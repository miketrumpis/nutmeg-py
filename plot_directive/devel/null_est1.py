import numpy as np
import nutmeg.stats.stats_utils as su
import matplotlib.pyplot as pp

# simulate some maximal null distribution from 100 permutations
null = np.sort(np.random.normal(size=(100,), loc=4, scale=1.2))
# create test scores from our experiment
# (but for ease of demonstration, clip the values to be within the null dist)
t_scores = np.clip(
    np.random.normal(size=(20,), loc=5, scale=0.8),
    null.min(), null.max()*.99
    )

# Score the t_scores based on the empirical null distribution
index = su.index(t_scores, null)
p_table = np.linspace(1,0,len(null)+1,endpoint=True)
p_scores = np.take(p_table, index)

# Now, try to recover the null distribution, based on these few
# T-score and P-score pairs -- use the negative edges
edges, pbins = su.map_t(-t_scores, p_scores, 1/100.)

edges = -edges[::-1]

p_recovered = np.array(
    [ ( edges > t_scores[i] ).sum() for i in xrange(len(t_scores))]
    )
assert (np.round(p_scores*100) == p_recovered).all()

probability = np.arange(100)/100.

si = np.argsort(p_scores)
print np.round(p_scores[si]*100), t_scores[si]
pp.plot(probability, null)
pp.plot(probability, edges, 'r')

x_locs = 1 - 1/100. - p_scores
y_mins = [0] * len(x_locs)
y_maxs = t_scores

pp.vlines(x_locs, y_mins, y_maxs, colors='g')
pp.plot(x_locs, y_maxs, 'go')
pp.gca().set_title('estimated null with 20 samples')
pp.show()

