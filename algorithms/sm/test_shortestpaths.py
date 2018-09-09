from scipy.sparse import save_npz, load_npz
from shortest_paths import dijkstra
from os.path import join, abspath, expanduser
import numpy as np
HOME = abspath(expanduser('~/Documents/streamminer/data/'))
from rel_closure_2 import relational_closure_sm as relclosure

if __name__ == "__main__":
	source  = 392035
	target = 2115741
	G_fil_rel = load_npz(join(HOME, 'sm', 'G_fil_rel.npz'))
	G_fil_val = load_npz(join(HOME, 'sm', 'G_fil_val.npz'))
	print "Start mining"
	weight_stack, path_stack, rel_stack = relclosure(G_fil_val, G_fil_rel, source, 599, target, kind='metric', linkpred = False)
	print path_stack, rel_stack, weight_stack
	# dist_matrix, predecessors = dijkstra(G_fil_val, directed=True, indices=source,
	#              return_predecessors=False,
	#              unweighted=False, limit=np.inf, target=target)
