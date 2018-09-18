# coding: utf-8
import heapq
import sys
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import ujson as json
import logging as log
from copy import copy
from tqdm import tqdm
import gc
###### Cython benign warning ignore ##########################
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
###############################################################
from pandas import DataFrame, Series
from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from time import time
from datetime import date
import cPickle as pkl
#####################################
from datastructures.rgraph import Graph, weighted_degree
#####################################
from time import time
from os.path import exists, join, abspath, expanduser, basename, dirname, \
	isdir, splitext
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.sparse import save_npz, load_npz, csr_matrix
from datastructures.rgraph import make_graph, Graph
from datastructures.relationalpath import RelationalPath
from datastructures.relationalpath_sm import RelationalPathSM
##############################################
from algorithms.sm.rel_closure_2 import relational_closure_sm as relclosure_sm
from algorithms.pra.pra_mining import find_best_model
##############################################

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float
inf = float('inf')

# Date
DATE = '{}'.format(date.today())

########################################################
################# DATABASE SETUP #######################
########################################################
# KG - DBpedia
HOME = abspath(expanduser('~/Documents/streamminer/data/'))
if not exists(HOME):
	print 'Data directory not found: %s' % HOME
	print 'and enter the directory path below.'
	data_dir = raw_input('\nPlease enter data directory path: ')
	if data_dir != '':
		data_dir = abspath(expanduser(data_dir))
	if not os.path.isdir(data_dir):
		raise Exception('Entered path "%s" not a directory.' % data_dir)
	if not exists(data_dir):
		raise Exception('Directory does not exist: %s' % data_dir)
	HOME = data_dir
	# raise Exception('Please set HOME to data directory in algorithms/__main__.py')
PATH = join(HOME, 'kg/_undir/')
assert exists(PATH)
SHAPE = (6060993, 6060993, 663)
WTFN = 'logdegree'

##############################################################
RELSIMPATH = join(HOME, 'relsim/coo_mat_sym_2016-10-24_log-tf_tfidf.npy')
assert exists(RELSIMPATH)
##############################################################

# ██    ██ ████████ ██ ██      ██ ████████ ██    ██
# ██    ██    ██    ██ ██      ██    ██     ██  ██
# ██    ██    ██    ██ ██      ██    ██      ████
# ██    ██    ██    ██ ██      ██    ██       ██
#  ██████     ██    ██ ███████ ██    ██       ██

def weighted_degree(arr, weight='logdegree'):
	"""Returns a weighted version of the array."""
	if weight == 'degree':
		arr = 1./(1 + arr)
	elif weight == 'logdegree':
		arr = 1./(1 + np.log(arr))
	else:
		raise ValueError('Unknown weight function.')
	return arr

def delete_node(Gv, Gr, s):
    # for now it just deletes outward edges from the s node
    s = int(s)
    deletedNodes = []
    start = Gr.indptr[s]
    end = Gr.indptr[s+1]
    tmp = Gv.data[start:end]
    # deleting data values
    Gv.data[start:end] = np.inf
    deletedNodes.append((s, tmp))
    return deletedNodes

def add_node(Gv, Gr, removedNodes):
    for removedNode in removedNodes:
        start = Gr.indptr[removedNode[0]]
        end = Gr.indptr[removedNode[0]+1]
        Gv.data[start:end] = removedNode[1]

def delete_edge(Gv, Gr, s, p, o):
	s, p, o = int(s), int(p), int(o)
	deletedEdges = []

	# deleting the edge: s --> o
	start = Gr.indptr[s]
	end = Gr.indptr[s+1]
	neighbors = Gr.indices[start:end]
	rels = Gr.data[start:end]
	pos = start + np.where(np.logical_and(neighbors == o, rels == p))
	deletedEdges.append((s, o, p, Gv.data[pos]))
	Gv.data[pos] = np.inf

	# deleting the edge: o --> s
	start = Gr.indptr[o]
	end = Gr.indptr[o+1]
	neighbors = Gr.indices[start:end]
	rels = Gr.data[start:end]
	pos = start + np.where(np.logical_and(neighbors == s, rels == p))

	deletedEdges.append((o, s, p, Gv.data[pos]))
	Gv.data[pos] = np.inf

	return deletedEdges

def add_edge(Gv, Gr, removed_edges):
    for removed_edge in removed_edges:
        s, o, p, cost = removed_edge
        start = Gr.indptr[s]
        end = Gr.indptr[s+1]
        neighbors = Gr.indices[start:end]
        rels = Gr.data[start:end]
        Gv.data[start + np.where(np.logical_and(neighbors == o, rels == p))] = cost

# ██████   █████  ███████ ███████     ███████ ███    ███
# ██   ██ ██   ██ ██      ██          ██      ████  ████
# ██████  ███████ ███████ █████       ███████ ██ ████ ██
# ██   ██ ██   ██      ██ ██               ██ ██  ██  ██
# ██████  ██   ██ ███████ ███████     ███████ ██      ██

def train_model_sm(G, triples, relsim, use_interpretable_features=False, cv=10):
	"""
	Entry point for building a fact-checking classifier.
	Performs three steps:
	1. Path extraction (features)
	2a. Path selection using information gain
	2b. Filtering most informative discriminative predicate paths
	3. Building logistic regression model

	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	use_interpretable_features: bool
		Whether or not to perform 2b.
	cv: int
		Number of cross-validation folds.

	Returns:
	--------
	vec: DictVectorizer
		Useful for preprocessing future triples.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.
	"""
	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].to_dict(orient='records')

	pid = triples[0]['pid']
	log.info('PID is: {}, with type: {}'.format(pid, pid.dtype))
	#print 'PID is: {}, with type: {}'.format(pid, pid.dtype)

	if np.DataSource().exists(join(HOME, "sm", "G_fil_val_{}.npz".format(int(pid)) ))\
	   and np.DataSource().exists(join(HOME, "sm", "G_fil_rel_{}.npz".format(int(pid)) )):
		Gr = load_npz(join(HOME, 'sm', 'G_fil_rel_{}.npz'.format(int(pid)) ))
		Gv = load_npz(join(HOME, 'sm', 'G_fil_val_{}.npz'.format(int(pid)) ))
	else:
		# set weights
		indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
		indegsim = indegsim.ravel()
		targets = G.csr.indices % G.N
		relations = (G.csr.indices - targets) / G.N
		relsimvec = np.array(relsim[int(pid), :]) # specific to predicate p
		relsim_wt = relsimvec[relations] # with the size of relations as the number of relations
		######################################################
		specificity_wt = indegsim[targets] # specificity

		## Removing all the edges with the predicte p in between any nodes.
		log.info('=> Removing predicate {} from KG.\n'.format(pid))
		eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
		specificity_wt[eraseedges_mask] = 0
		relsim_wt[eraseedges_mask] = 0
		G.csr.data = specificity_wt.copy()
		print ''

		G.csr.data = np.multiply(relsim_wt, G.csr.data)
		log.info("Constructing adjacency matrix for: {}".format(pid))
		adj_list_data = []
		adj_list_s = []
		adj_list_p = []
		adj_list_o = []
		sel_data = np.array([])
		sel_relations = np.array([])
		dicti = {}
		num_nodes = len(G.csr.indptr)-1
		for node in tqdm(xrange(num_nodes)):
		    dicti = {}
		    start = G.csr.indptr[node]
		    end = G.csr.indptr[node+1]

		    sel_data = G.csr.data[start:end]
		    sel_relations = relations[start:end]
		    for i, sel_tar in enumerate(targets[start:end]):
		        if sel_tar in dicti:
		            if dicti[sel_tar][0] < sel_data[i]:
		                dicti[sel_tar] = (sel_data[i], sel_relations[i])
		        else:
		            dicti[sel_tar] = (sel_data[i], sel_relations[i])
		    for key, value in dicti.iteritems():
		        if value[0] != 0:
		            adj_list_data.append(value[0])
		            adj_list_s.append(node)
		            adj_list_p.append(value[1])
		            adj_list_o.append(key)
		Gr = csr_matrix((adj_list_p, (adj_list_s, adj_list_o)), shape=(num_nodes, num_nodes))
		Gv = csr_matrix((adj_list_data, (adj_list_s, adj_list_o)), shape=(num_nodes, num_nodes))
		save_npz(join(HOME, 'sm', 'G_fil_rel_{}.npz'.format(int(pid))), Gr)
		save_npz(join(HOME, 'sm', 'G_fil_val_{}.npz'.format(int(pid))), Gv)

	############# Path extraction ###################
	log.info('=> Path extraction..(this can take a while)')
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths_sm(Gv, Gr, triples, y)
	log.info('P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features)))
	vec = DictVectorizer()
	X = vec.fit_transform(measurements)
	n, m = X.shape
	log.info('Time taken: {:.2f}s\n'.format(time() - t1))
	print ''

	########### Path selection ###############
	log.info('=> Path selection..')
	t1 = time()
	pathselect = SelectKBest(mutual_info_classif, k=min(100, m))
	X_select = pathselect.fit_transform(X, y)
	selectidx = pathselect.get_support(indices=True) # selected feature indices
	vec = vec.restrict(selectidx, indices=True)
	select_pos_features, select_neg_features = set(), set()
	for feature in vec.get_feature_names():
		if feature in pos_features:
			select_pos_features.add(feature)
		if feature in neg_features:
			select_neg_features.add(feature)
	log.info('D: +:{}, -:{}, tot:{}'.format(len(select_pos_features), len(select_neg_features), X_select.shape[1]))
	log.info('Time taken: {:.2f}s\n'.format(time() - t1))
	print ''

	# Fact interpretation
	if use_interpretable_features and len(select_neg_features) > 0:
		log.info('=> Fact interpretation..')
		t1 = time()
		theta = 10
		select_neg_idx = [i for i, f in enumerate(vec.get_feature_names()) if f in select_neg_features]
		removemask = np.where(np.sum(X_select[:, select_neg_idx], axis=0) >= theta)[0]
		restrictidx = select_neg_idx[removemask]
		keepidx = []
		for i, f in enumerate(vec.get_feature_names()):
			if i not in restrictidx:
				keepidx.append(i)
			else:
				select_neg_features.remove(f)
		vec = vec.restrictidx(keepidx, indices=True)
		X_select = X_select[:, keepidx]
		log.info('D*: +:{}, -:{}, tot:{}'.format(len(select_pos_features), len(select_neg_features), X_select.shape[1]))
		log.info('Time taken: {:.2f}s\n'.format(time() - t1))

	# Model creation
	log.info('=> Model building..')
	#print '=> Model building..'
	t1 = time()
	model = find_best_model(X_select, y, cv=cv)
	log.info('#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score']))
	#print '#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score'])
	log.info('Time taken: {:.2f}s\n'.format(time() - t1))

	return vec, model

# ███████ ██   ██ ████████ ██████   █████   ██████ ████████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██
# █████     ███      ██    ██████  ███████ ██         ██
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██

def extract_paths_sm(Gv, Gr, triples, y, features=None):
    return_features = False
    if features is None:
        return_features = True
        features, pos_features, neg_features = set(), set(), set()
    measurements = []

    for idx, triple in enumerate(tqdm(triples)):
        sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
        label = y[idx]
        triple_feature = dict()
        discovered_paths = yenKSP5(Gv, Gr, sid, pid, oid, K = 5)
        for path in discovered_paths:
            log.info(path)
            ff = tuple(path.relational_path)
            if ff not in features:
                features.add(ff)
                if label == 1:
                    pos_features.add(ff)
                elif label == 0:
                    neg_features.add(ff)
                else:
                    raise Exception("Unknown class label: {}".format(label))
            triple_feature[ff] = triple_feature.get(ff, 0) + 1
        measurements.append(triple_feature)
    print ''
    if return_features:
        return features, pos_features, neg_features, measurements
    return measurements

# ██   ██ ███████ ██████
# ██  ██  ██      ██   ██
# █████   ███████ ██████
# ██  ██       ██ ██
# ██   ██ ███████ ██

def yenKSP5(Gv, Gr, sid, pid, oid, K = 5):
	discovered_paths = []
	weight_stack, path_stack, rel_stack = relclosure_sm(Gv, Gr, int(sid), int(pid), int(oid), kind='metric', linkpred = False)
	if rel_stack == [-1]:
		## if the first shortest path is empty, retuen empty discoverd_paths
		return discovered_paths
	A = [{'path_total_cost': np.sum(weight_stack[:-1]),
		'path': path_stack,
		'path_rel': rel_stack,
		'path_weights': weight_stack}]
	B = []
	removed_edges = []
	removed_nodes = []
	for k in xrange(1, K): #for the k-th path, it assumes all paths 1..k-1 are available
		for i in xrange(0, len(A[-1]['path'])-1):
			gc.collect()
			# the spurnode ranges from first node of the previous (k-1) shortest path to its next to last node.
			spurNode = A[-1]['path'][i]
			rootPath = A[-1]['path'][:i+1]
			rootPathRel = A[-1]['path_rel'][:i+1]
			rootPathWeights = A[-1]['path_weights'][:i+1]
			# print "SpurNode: {}, Rootpath: {}".format(spurNode, rootPath)
			removed_edges[:] = []
			removed_nodes[:] = []
			for path_dict in A:
				if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
					removed_edges.extend( delete_edge(Gv, Gr, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1]) )
			for rootPathNode in rootPath[:-1]:
				removed_nodes.extend( delete_node(Gv, Gr, rootPathNode) )
			spurPathWeights, spurPath, spurPathRel = relclosure_sm(Gv, Gr, int(spurNode), int(pid), int(oid), kind='metric', linkpred = False)
			if spurPath and spurPathRel != [-1]:
				totalPath = rootPath[:-1] + spurPath
				totalDist = np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights[:-1])
				totalWeights = rootPathWeights[:-1] + spurPathWeights[:]
				totalPathRel = rootPathRel[:] + spurPathRel[1:]
				potential_k = {'path_total_cost': totalDist,
								'path': totalPath,
								'path_rel': totalPathRel,
								'path_weights': totalWeights}
				if not (potential_k in B or potential_k in A):
					# removes repititive paths in A & B
					B.append(potential_k)
			removed_nodes.reverse()
			add_node(Gv, Gr, removed_nodes)
			removed_edges.reverse()
			add_edge(Gv, Gr, removed_edges)
		if len(B):
			B = sorted(B, key=lambda k: k['path_total_cost'])
			A.append(B[0])
			B.pop(0)
		else:
			break
	for path_dict in A:
		discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
	gc.collect()
	return discovered_paths


# ███████ ██ ███    ██ ██████      ██████  ███████ ███████ ████████
# ██      ██ ████   ██ ██   ██     ██   ██ ██      ██         ██
# █████   ██ ██ ██  ██ ██   ██     ██████  █████   ███████    ██
# ██      ██ ██  ██ ██ ██   ██     ██   ██ ██           ██    ██
# ██      ██ ██   ████ ██████      ██████  ███████ ███████    ██


def find_best_model(X, y, scoring='roc_auc', cv=10):
	"""
	Fits a logistic regression classifier to the input data (X, y),
	and returns the best model that maximizes `scoring` (e.g. AUROC).

	Parameters:
	-----------
	X: sparse matrix
		Feature matrix.
	y: array
		A vector of ground truth labels.
	scoring: str
		A string indicating the evaluation criteria to use. e.g. ROC curve.
	cv: int
		No. of folds in cross-validation.

	Returns:
	--------
	best: dict
		Best model key-value pairs. e.g. classifier, best score on
		left out data, optimal parameter.
	"""
	steps = [('clf', LogisticRegression())]
	pipe = Pipeline(steps)
	params = {'clf__C': [1, 5, 10, 15, 20]}
	grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, refit=True, scoring=scoring)
	grid_search.fit(X, y)
	best = {
		'clf': grid_search.best_estimator_,
		'best_score': grid_search.best_score_,
		'best_param': grid_search.best_params_
	}
	return best

# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████



def main(args=None):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-d', type=str, required=True,
			dest='dataset', help='Dataset to test on.')
	parser.add_argument('-o', type=str, required=True,
			dest='outdir', help='Path to the output directory.')
	parser.add_argument('-m', type=str, required=True,
			dest='method', help='Method to use: stream, relklinker, klinker, \
			predpath, sm')
	args = parser.parse_args()

	relsim = np.load(RELSIMPATH)

	outdir = abspath(expanduser(args.outdir))
	assert exists(outdir)
	args.outdir = outdir
	datafile = abspath(expanduser(args.dataset))
	assert exists(datafile)
	args.dataset = datafile
	LOGPATH = join(HOME, '../logs')
	assert exists(LOGPATH)
	base = splitext(basename(args.dataset))[0]
	log_file = join('logs/', 'log_{}_{}_{}.log'.format(args.method, base, DATE))
	log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = log_file, level=log.DEBUG)
	log.getLogger().addHandler(log.StreamHandler())
	log.info('Launching {}..'.format(args.method))
	log.info('Dataset: {}'.format(basename(args.dataset)))
	log.info('Output dir: {}'.format(args.outdir))

	# read data
	df = pd.read_table(args.dataset, sep=',', header=0)
	log.info('Read data: {} {}'.format(df.shape, basename(args.dataset)))
	spo_df = df.dropna(axis=0, subset=['sid', 'pid', 'oid'])
	log.info('Note: Found non-NA records: {}'.format(spo_df.shape))
	df = spo_df[['sid', 'pid', 'oid']].values
	subs, preds, objs  = df[:,0].astype(_int), df[:,1].astype(_int), df[:,2].astype(_int)

	# load knowledge graph
	G = Graph.reconstruct(PATH, SHAPE, sym=True) # undirected
	assert np.all(G.csr.indices >= 0)

	t1 = time()

	if args.method == 'sm':
		vec, model = train_model_sm(G, spo_df, relsim) # train
		log.info('Time taken: {:.2f}s\n'.format(time() - t1))
		# save model
		predictor = { 'dictvectorizer': vec, 'model': model }
		try:
			outpkl = join(args.outdir, 'out_streamminer_{}_{}.pkl'.format(base, DATE))
			with open(outpkl, 'wb') as g:
				pkl.dump(predictor, g, protocol=pkl.HIGHEST_PROTOCOL)
			log.info('Saved: {}'.format(outpkl))
		except IOError, e:
			raise e
	log.info('\nDone!\n')

if __name__ == '__main__':
	main()
