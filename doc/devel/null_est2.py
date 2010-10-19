import numpy as np
import nutmeg.stats.stats_utils as su
import matplotlib.pyplot as pp

# simulate some null distribution from 100 permutations
null = np.sort(np.random.normal(size=(100,), loc=4, scale=1.2))

# simulate a larger sampling of natural T scores
t_scores = np.clip(np.random.normal(size=(200,), loc=5, scale=0.8), null.min(), null.max()*.99)
index = su.index(t_scores, null)
p_table = np.linspace(1,0,len(null)+1,endpoint=True)
p_scores = np.take(p_table, index)
edges, pbins = su.map_t(-t_scores, p_scores, 1/100.)

edges = -edges[::-1]

probability = np.arange(100)/100.
pp.plot(probability, null)
pp.plot(probability, edges, 'r')
pp.gca().set_title('estimated null with 200 samples')
pp.show()
