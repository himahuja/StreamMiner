# coding: utf-8
"""
PredPath (PP) model building and prediction.

Source: 'Discriminative Predicate Path Mining for Fact-Checking
in Knowledge Graphs' by Baoxu Shi and Tim Weninger.

Performs three things:
- Path extraction: Extracts anchored predicate paths as features, and constructs feature matrix
for a given set of triples.
- Path selection: Computes mutual information / information gain between features and label
for identifying discriminative predicate paths.
- Model building: Trains a logistic regression model that optimizes AUROC and empirically
sets a threshold 'delta' for retaining most informative feature paths.
"""
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

from datastructures.rgraph import make_graph, Graph
from datastructures.relationalpath import RelationalPath
from datastructures.relationalpath_sm import RelationalPathSM
from pathenum import get_paths as c_get_paths
## for streamminer,
from pathenum import get_paths_sm as c_get_paths_sm

##############################################
from algorithms.mincostflow.ssp import succ_shortest_path, disable_logging
from algorithms.relklinker.rel_closure import relational_closure as relclosure
from algorithms.sm.rel_closure import relational_closure_sm as relclosure_sm
from algorithms.klinker.closure import closure
# from algorithms.sm.ksp import k_shortest_paths
##############################################

###################################################################
################# DATABASE and RELSIM SETUP #######################
###################################################################
# KG - DBpedia
HOME = abspath(expanduser('~/Documents/streamminer/data/'))
if not exists(HOME):
	print 'Data directory not found: %s' % HOME
	print 'Download data per instructions on:'
	print '\thttps://github.com/shiralkarprashant/knowledgestream#data'
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
# WTFN = 'logdegree'
WTFN = 'logdegree'

# # relational similarity using TF-IDF representation and cosine similarity
# RELSIMPATH = join(HOME, 'relsim/coo_mat_sym_2016-10-24_log-tf_tfidf.npy')
# assert exists(RELSIMPATH)
##############################################################
RELSIMPATH = join(HOME, 'relsim/coo_mat_sym_2016-10-24_log-tf_tfidf.npy')
assert exists(RELSIMPATH)
##############################################################
# relational similarity using TF-IDF representation and cosine similarity

# relsim = np.load(RELSIMPATH)

# Date
DATE = '{}'.format(date.today())

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

inf = float('inf')
#######################################################################
#######################################################################

# ███████ ████████ ██████  ███████  █████  ███    ███     ███    ███ ██ ███    ██ ███████ ██████
# ██         ██    ██   ██ ██      ██   ██ ████  ████     ████  ████ ██ ████   ██ ██      ██   ██
# ███████    ██    ██████  █████   ███████ ██ ████ ██     ██ ████ ██ ██ ██ ██  ██ █████   ██████
#      ██    ██    ██   ██ ██      ██   ██ ██  ██  ██     ██  ██  ██ ██ ██  ██ ██ ██      ██   ██
# ███████    ██    ██   ██ ███████ ██   ██ ██      ██     ██      ██ ██ ██   ████ ███████ ██   ██


def predpath_train_model_sm(G, triples, relsim, use_interpretable_features=False, cv=10):
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
	# defining the total path weights
	weight = 20.0

	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].to_dict(orient='records')

	pid = triples[0]['pid']
	log.info('PID is: {}, with type: {}'.format(pid, pid.dtype))
	#print 'PID is: {}, with type: {}'.format(pid, pid.dtype)

	# G.targets = G.csr.indices % G.N
	G_bak = {'data': G.csr.data.copy(),
	'indices': G.csr.indices.copy(),
	'indptr': G.csr.indptr.copy()
	}
	# cost_vec = cost_vec_bak.copy()
	# indegsim = weighted_degree(G.indeg_vec, weight=WTFN)
	# specificity_wt = indegsim[G.targets] # specificity

	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	relations = (G.csr.indices - targets) / G.N
	#####################################################
	# Setting paths using relsim (from K-Stream)
	# set weights
	relsimvec = np.array(relsim[int(pid), :]) # specific to predicate p
	relsim_wt = relsimvec[relations] # with the size of relations as the number of relations
	# G.csr.data = np.multiply(relsim_wt, specificity_wt) # it is the capacity (U) of each edge, under p
	######################################################
	specificity_wt = indegsim[targets] # specificity


	## Removing all the edges with the predicte p in between any nodes.
	log.info('=> Removing predicate {} from KG.\n'.format(pid))
	#print '=> Removing predicate {} from KG.'.format(pid)
	eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
	specificity_wt[eraseedges_mask] = 0
	G.csr.data = specificity_wt.copy()
	#print ''

	# Path extraction
	log.info('=> Path extraction..(this can take a while)')
	#print '=> Path extraction..(this can take a while)'
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths_sm(G, relsim_wt, triples, y, weight)
	log.info('P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features)))
	#print 'P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features))
	vec = DictVectorizer()
	X = vec.fit_transform(measurements)
	n, m = X.shape
	log.info('Time taken: {:.2f}s\n'.format(time() - t1))
	#print 'Time taken: {:.2f}s'.format(time() - t1)
	#print ''

	# Path selection
	log.info('=> Path selection..')
	#print '=> Path selection..'
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
	#print 'D: +:{}, -:{}, tot:{}'.format(len(select_pos_features), len(select_neg_features), X_select.shape[1])
	#print 'Time taken: {:.2f}s'.format(time() - t1)
	#print ''

	# Fact interpretation
	if use_interpretable_features and len(select_neg_features) > 0:
		log.info('=> Fact interpretation..')
		#print '=> Fact interpretation..'
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
		#print 'D*: +:{}, -:{}, tot:{}'.format(len(select_pos_features), len(select_neg_features), X_select.shape[1])
		#print 'Time taken: {:.2f}s'.format(time() - t1)
		#print ''

	# Model creation
	log.info('=> Model building..')
	#print '=> Model building..'
	t1 = time()
	model = find_best_model(X_select, y, cv=cv)
	log.info('#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score']))
	#print '#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score'])
	log.info('Time taken: {:.2f}s\n'.format(time() - t1))
	#print 'Time taken: {:.2f}s'.format(time() - t1)
	#print ''
	############################################
	## From KStream
	np.copyto(G.csr.data, G_bak['data'])
	np.copyto(G.csr.indices, G_bak['indices'])
	np.copyto(G.csr.indptr, G_bak['indptr'])
	np.copyto(cost_vec, cost_vec_bak)
	############################################

	return vec, model

###########################################################

# ███████ ██   ██ ████████ ██████   █████   ██████ ████████     ██████   █████  ████████ ██   ██ ███████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██        ██   ██ ██   ██    ██    ██   ██ ██
# █████     ███      ██    ██████  ███████ ██         ██        ██████  ███████    ██    ███████ ███████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██        ██      ██   ██    ██    ██   ██      ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██        ██      ██   ██    ██    ██   ██ ███████

#############################################################

def extract_paths_sm(G, relsim_wt, triples, y, weight = 10.0, features=None):
	"""
	Extracts anchored predicate paths for a given sequence of triples.

	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: sequence
		A list of triples (sid, pid, oid).
	y: array
		A sequence of class labels.
	length: int
		Maximum length of any path.
	features: dict
		Features extracted earlier. A set of (feature_id, path) pairs.
		If None, it is assumed feature set and feature matrix are desired.
		If not None, only X (feature matrix) is returned.

	Returns:
	--------
	features: dict
		A set of (feature_id, path) pairs.
	X: dict
		A dictionary representation of feature matrix.
	"""

	# print "Shape of the data array: {}".format(G.csr.data.shape)
	return_features = False

	if features is None:
		return_features = True
		features, pos_features, neg_features = set(), set(), set()
	measurements = []
	# Make backup here
		## Create graph backup
	G_bak = {
		'data': G.csr.data.copy(),
		'indices': G.csr.indices.copy(),
		'indptr': G.csr.indptr.copy()
	}

	for idx, triple in enumerate(triples):
		sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
		label = y[idx]
		triple_feature = dict()

		targets = G.csr.indices % G.N
		G.csr.data[targets == oid] = 1 # no cost for target t => max. specificity.
		G.csr.data = np.multiply(relsim_wt, G.csr.data)
		# Alex 1/9/18 - added for debugging purposes
		log.debug("(extract_paths_sm, Before YenKSP4) indices: {}, data: {}".format(G.csr.data, G.csr.indices))
		log.debug("(extract_paths_sm, Before YenKSP4) The masked edges for {}: {}".format(pid, ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid))

		# paths = get_paths_sm(G, sid, pid, oid, relsim_wt, \
								# weight = weight, maxpaths=20)
		# paths = get_paths_sm_limited(G, sid, pid, oid, relsim_wt, \
		# 				weight = weight, maxpaths=20, top_n_neighbors=5)
		paths = yenKSP4(G, sid, pid, oid)
		log.debug("(extract_paths_sm, After YenKSP4) indices: {}, data: {}".format(G.csr.data, G.csr.indices))
		log.debug("(extract_paths_sm, After YenKSP4) The masked edges for {}: {}".format(pid, ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid))
		trus = [elem == True for elem in ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid]
		logger = log.getLogger()
		for i in range(len(logger.handlers)):
			logger.handlers[i].flush()
		for pth in paths:
			ff =  tuple(pth.relational_path)
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
		sys.stdout.flush()
		np.copyto(G.csr.data, G_bak['data'])
		np.copyto(G.csr.indices, G_bak['indices'])
		np.copyto(G.csr.indptr, G_bak['indptr'])
	print ''
	if return_features:
		return features, pos_features, neg_features, measurements
	#else
	return measurements

def get_paths_sm(G, s, p, o, relsim_wt, weight = 10.0, maxpaths=-1):
	# "Returns all paths of length `length` starting at s and ending in o."
	path_stack = [[s]]
	weight_path_stack = [[0.0]]
	relpath_stack = [[-1]]
	discoverd_paths = []
	print 'We\'re checking the path: ({}, {}, {})'.format(s, p, o)
	while len(path_stack) > 0:
		curr_path = path_stack.pop()
		curr_relpath = relpath_stack.pop()
		node = curr_path[-1]
		curr_path_weight = weight_path_stack.pop()
		if np.sum(curr_path_weight) <= weight:
			if int(node) == int(o):
				print 'We have found a path!'
				print 'The total weight of the path is: {}'.format(np.sum(curr_path_weight))
				path = RelationalPathSM(
					s, p, o, 0., len(curr_path)-1, curr_path, curr_relpath, curr_path_weight
				)
				discoverd_paths.append(path)
				if maxpaths != -1 and len(discoverd_paths) >= maxpaths:
					break
				continue
		elif np.sum(curr_path_weight) >= weight:
			continue
		# print "Node is: {}, o is: {}, s is: {}, p is: {}".format(node, o, s, p)
		relnbrs, data = G.get_neighbors_sm(int(node))
		# print 'Data vector is: {}'.format(data)
		for i in xrange(relnbrs.shape[1]):
			rel, nbr = relnbrs[:, i]
			# print "rel is: {}, nbr is: {}".format(rel, nbr)
			path_stack.append(curr_path + [nbr])
			weight_path_stack.append(curr_path_weight + [data[i]])
			relpath_stack.append(curr_relpath + [rel])
	return discoverd_paths

def get_paths_sm_limited(G, s, p, o, relsim_wt, weight = 10.0, maxpaths=-1, top_n_neighbors=5):
	# "Returns all paths of length `length` starting at s and ending in o."
	path_stack = [[s]]
	weight_path_stack = [[0.0]]
	relpath_stack = [[-1]]
	discoverd_paths = []
	print 'We\'re checking the path: ({}, {}, {})'.format(s, p, o)
	while len(path_stack) > 0:
		# print 'Stack: {} {}'.format(path_stack, relpath_stack)
		curr_path = path_stack.pop()
		curr_relpath = relpath_stack.pop()
		node = curr_path[-1]
		curr_path_weight = weight_path_stack.pop()
		# print 'Node: {}'.format(node)
		total_path_weight = np.sum(curr_path_weight)
		if total_path_weight <= weight:
			if int(node) == int(o):
				print 'We have found a path!'
				print 'The total weight of the path is: {}'.format(np.sum(curr_path_weight))
				path = RelationalPathSM(
					s, p, o, 0., length, curr_path, curr_relpath, curr_path_weight
				)
				discoverd_paths.append(path)
				if maxpaths != -1 and len(discoverd_paths) >= maxpaths:
					print "Exceeded number of paths!"
					break
				continue
		elif total_path_weight > weight:
			print 'Discarded path with weight'
			continue
		# print "Node is: {}, o is: {}, s is: {}, p is: {}".format(node, o, s, p)
		relnbrs, data = G.get_neighbors_sm(int(node))
		ordering = np.argsort(data)
		relnbrs = relnbrs[:, ordering]
		# print 'Data vector is: {}'.format(data)
		# print(data)
		range = relnbrs.shape[1] if relnbrs.shape[1] < top_n_neighbors else top_n_neighbors
		for i in xrange(range):
			rel, nbr = relnbrs[:, i]
			# print "rel is: {}, nbr is: {}".format(rel, nbr)
			path_stack.append(curr_path + [nbr])
			weight_path_stack.append(curr_path_weight + [data[i]])
			relpath_stack.append(curr_relpath + [rel])
	return discoverd_paths

###################################################

# ██    ██ ████████ ██ ██      ██ ████████ ███████ ███████
# ██    ██    ██    ██ ██      ██    ██    ██      ██
# ██    ██    ██    ██ ██      ██    ██    █████   ███████
# ██    ██    ██    ██ ██      ██    ██    ██           ██
#  ██████     ██    ██ ███████ ██    ██    ███████ ███████

###################################################
def in_lists(list1, list2):
    result = False
    node_result = -1
    if len(list1) < len(list2):
        toIter = list1
        toRefer = list2
    else:
        toIter = list2
        toRefer = list1
    for element in toIter:
        result = element in toRefer
        if result:
            node_result = element
            break
    return result, node_result

def relax(weight, u, v, r, Dist, prev):
	d = Dist.get(u, inf) + weight
	if d < Dist.get(v, inf):
		Dist[v] = d
		prev[v] = (-weight, u, r)

def delete_edge(G, s, p, o, removed_edges):

	flag = 0
	s, p, o = int(s), int(p), int(o)

	try:
		start = G.csr.indptr[s]
		end = G.csr.indptr[s+1]
		neighbors = G.csr.indices[start:end] % G.N# nbrs in wide-CSR
		rels = (G.csr.indices[start:end] - neighbors) / G.N
		foo = np.zeros(rels.shape[0])
		bar = np.zeros(neighbors.shape[0])
		foo[np.where(rels == p)[0][0]] = 1
		bar[np.where(neighbors == o)[0][0]] = 1
		check = foo.astype(int) & bar.astype(int)

		if np.sum(check) == 1:
			edge = G.csr.data[ start + np.where(check)[0][0] ]
			if edge == 0:
				flag = 0
			else:
				flag = 1
				removed_edges.append((s, o, p, edge))
				G.csr.data[ start + np.where(check)[0][0] ] = 0

		start = G.csr.indptr[o]
		end = G.csr.indptr[o+1]
		neighbors = G.csr.indices[start:end] % G.N # nbrs in wide-CSR
		rels = (G.csr.indices[start:end] - neighbors) / G.N
		foo = np.zeros(rels.shape[0])
		bar = np.zeros(neighbors.shape[0])
		foo[np.where(rels == p)[0][0]] = 1
		bar[np.where(neighbors == s)[0][0]] = 1
		check = foo.astype(int) & bar.astype(int)
		if np.sum(check) == 1:
			edge = G.csr.data[ start + np.where(check)[0][0] ]
			if edge == 0:
				flag = 0
			else:
				flag = 1
				removed_edges.append((o, s, p, edge))
				G.csr.data[ start + np.where(check)[0][0] ] = 0
	except:
		flag = 0
	return removed_edges, flag

def add_edge(G, removed_edges):
	for removed_edge in removed_edges:
		s, o, p, cost = removed_edge
		start = G.csr.indptr[s]
		end = G.csr.indptr[s+1]
		neighbors = G.csr.indices[start:end] % G.N# nbrs in wide-CSR
		rels = (G.csr.indices[start:end] - neighbors) / G.N
		foo = np.zeros(rels.shape[0])
		bar = np.zeros(neighbors.shape[0])
		foo[np.where(rels == p)[0][0]] = 1
		bar[np.where(neighbors == o)[0][0]] = 1
		check = foo.astype(int) & bar.astype(int)
		G.csr.data[ start + np.where(check)[0][0] ] = cost
###########################################################

# ██████       ██ ██ ██   ██ ███████ ████████ ██████   █████  ███████
# ██   ██      ██ ██ ██  ██  ██         ██    ██   ██ ██   ██ ██
# ██   ██      ██ ██ █████   ███████    ██    ██████  ███████ ███████
# ██   ██ ██   ██ ██ ██  ██       ██    ██    ██   ██ ██   ██      ██
# ██████   █████  ██ ██   ██ ███████    ██    ██   ██ ██   ██ ███████

###########################################################

def get_shortest_path(G, sid, pid, oid):
	#making sure that nodes are integers:
	# discovered_path = []
	sid = int(sid)
	oid = int(oid)
	#prev is of the type: [weight, node, relation]

	Dist, visited, priority_q, prev = {sid:0}, set(), [(0,sid)], {sid:(0, -1, -1)}
	path_stack, rel_stack, weight_stack = [], [], []
	while priority_q:
		_, u = heapq.heappop(priority_q)
		if u == oid:
			k = u
			path_stack = [oid]
			while prev[k][1] != -1:
				path_stack.insert(0, prev[k][1])
				rel_stack.insert(0, prev[k][2])
				weight_stack.insert(0, prev[k][0])
				k = prev[k][1]
			break
		if u in visited:
			continue
		visited.add(u)
		# get the neighbours and cost of the node u
		# returns [relations, neighbors, cost]
		rels, nbrs, costs = G.get_neighbors_sm_unpacked(int(u))
		for rel, nbr, cost in zip(rels, nbrs, costs): # for the iteration through keys
			if cost != 0:
				relax(-cost, u, nbr, rel, Dist, prev)
				heapq.heappush(priority_q, (-cost, nbr))
				# discovered_path = RelationalPathSM(sid, pid, oid, 0., len(path_stack)-1, ..)  								  path_stack, rel_stack, weight_stack)
	return path_stack, rel_stack, weight_stack

#######################################################################

# ██    ██ ███████ ███    ██     ██   ██ ███████ ██████
#  ██  ██  ██      ████   ██     ██  ██  ██      ██   ██
#   ████   █████   ██ ██  ██     █████   ███████ ██████
#    ██    ██      ██  ██ ██     ██  ██       ██ ██
#    ██    ███████ ██   ████     ██   ██ ███████ ██

#######################################################################
def yenKSP(G, sid, pid, oid, K = 20):
	discovered_paths = []
	path_stack, rel_stack, weight_stack = get_shortest_path(G, sid, pid, oid)
	if not path_stack:
		return discovered_paths
	A = [{'path_total_cost': np.sum(weight_stack),
		'path': path_stack,
		'path_rel': rel_stack,
		'path_weights': weight_stack}]
	B = []
	for k in xrange(1, K):
		for i in xrange(0, len(A[-1]['path'])-1):
			spurNode = A[-1]['path'][i]
			rootPath = A[-1]['path'][:i+1]
			rootPathRel = A[-1]['path_rel'][:i]
			rootPathWeights = A[-1]['path_weights'][:i]
			#print "rp: {}, rpr: {}, rpw: {}".format(len(rootPath), len(rootPathRel), len(rootPathWeights))
			removed_edges = []
			for path_dict in A:
				if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
					#find the edge between ith and i+1th node
					edge = G.csr[path_dict['path'][i], path_dict['path_rel'][i]*G.N+path_dict['path'][i+1]]
					if edge == 0:
						continue
					removed_edges.append((path_dict['path'][i], path_dict['path'][i+1], path_dict['path_rel'][i], edge))
					edge = 0 #delete the edge
			spurPath, spurPathRel, spurPathWeights = get_shortest_path(G, spurNode, pid, oid)
			if spurPath:
				totalPath = rootPath[:-1] + spurPath
				totalDist = np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights)
				totalWeights = rootPathWeights[:-1] + spurPathWeights
				totalPathRel = rootPathRel[:-1] + spurPathRel
				potential_k = {'path_total_cost': totalDist,
							'path': totalPath,
							'path_rel': totalPathRel,
							'path_weights': totalWeights}
				if not (potential_k in B):
					B.append(potential_k)
			for removed_edge in removed_edges:
				G.csr[removed_edge[0], removed_edge[2]*G.N + removed_edge[1]] = removed_edge[3]
		if len(B):
			B = sorted(B, key=lambda k: k['path_total_cost'])
			A.append(B[0])
			B.pop(0)
		else:
			break
	for path_dict in A:
		discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
	return discovered_paths
#######################################################################

# ██    ██ ███████ ███    ██     ██   ██ ███████ ██████      ██████
#  ██  ██  ██      ████   ██     ██  ██  ██      ██   ██          ██
#   ████   █████   ██ ██  ██     █████   ███████ ██████       █████
#    ██    ██      ██  ██ ██     ██  ██       ██ ██          ██
#    ██    ███████ ██   ████     ██   ██ ███████ ██          ███████

#######################################################################

def yenKSP2(G, sid, pid, oid, K = 20):
	discovered_paths = []
	path_stack, rel_stack, weight_stack = get_shortest_path(G, sid, pid, oid)
	print "Shortest path for s:{}, p:{}, o:{} is: {}".format(sid, pid, oid, path_stack)
	if not path_stack:
		return discovered_paths
	A = [{'path_total_cost': np.sum(weight_stack),
		'path': path_stack,
		'path_rel': rel_stack,
		'path_weights': weight_stack}]
	B = []
	for k in xrange(1, K):
		for i in xrange(0, len(A[-1]['path'])-1):
			spurNode = A[-1]['path'][i]
			rootPath = A[-1]['path'][:i+1]
			rootPathRel = A[-1]['path_rel'][:i]
			rootPathWeights = A[-1]['path_weights'][:i]
			#print "rp: {}, rpr: {}, rpw: {}".format(len(rootPath), len(rootPathRel), len(rootPathWeights))
			removed_edges = []
			for path_dict in A:
				if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
					#find the edge between ith and i+1th node
					edge1 = G.csr[path_dict['path'][i], path_dict['path_rel'][i]*G.N+path_dict['path'][i+1]]
					edge2 = G.csr[path_dict['path'][i+1], path_dict['path_rel'][i]*G.N+path_dict['path'][i]]
					if edge1 == 0:
						continue
					if edge2 == 0:
						continue
					removed_edges.append((path_dict['path'][i], path_dict['path'][i+1], path_dict['path_rel'][i], edge1))
					removed_edges.append((path_dict['path'][i+1], path_dict['path'][i], path_dict['path_rel'][i], edge2))
					edge1 = 0 #delete the edge
					edge2 = 0
			while True:
				spurPath, spurPathRel, spurPathWeights = get_shortest_path(G, spurNode, pid, oid)

				[is_loop, loop_element] = in_lists(spurPath, rootPath[:-1]) ## Check indexing
				if not is_loop:
					break
				else:
					loop_index = spurPath.index(loop_element) # Check indexing
					edge1 = G.csr[spurPath[loop_index], spurPathRel[loop_index-1]*G.N + spurPath[loop_index-1]]
					edge2 = G.csr[spurPath[loop_index-1], spurPathRel[loop_index-1]*G.N + spurPath[loop_index]]
					if edge1 == 0:
						continue
					if edge2 == 0:
						continue
					removed_edges.append((spurPath[loop_index], spurPath[loop_index-1], spurPathRel[loop_index-1], edge1))
					removed_edges.append((spurPath[loop_index-1], spurPath[loop_index], spurPathRel[loop_index-1], edge2))
					edge1 = 0
					edge2 = 0

				if spurPath:
					totalPath = rootPath[:-1] + spurPath
					totalDist = np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights)
					totalWeights = rootPathWeights[:-1] + spurPathWeights
					totalPathRel = rootPathRel[:-1] + spurPathRel
					potential_k = {'path_total_cost': totalDist,
								'path': totalPath,
								'path_rel': totalPathRel,
								'path_weights': totalWeights}
					if not (potential_k in B):
						B.append(potential_k)
				# Add back the removed edges
				for removed_edge in removed_edges:
					G.csr[removed_edge[0], removed_edge[2]*G.N + removed_edge[1]] = removed_edge[3]
		if len(B):
			B = sorted(B, key=lambda k: k['path_total_cost'])
			A.append(B[0])
			B.pop(0)
		else:
			break
	for path_dict in A:
		discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
	return discovered_paths

##################################################################

# ██    ██ ███████ ███    ██     ██   ██ ███████ ██████      ██████
#  ██  ██  ██      ████   ██     ██  ██  ██      ██   ██          ██
#   ████   █████   ██ ██  ██     █████   ███████ ██████       █████
#    ██    ██      ██  ██ ██     ██  ██       ██ ██               ██
#    ██    ███████ ██   ████     ██   ██ ███████ ██          ██████

####################################################################

def yenKSP3(G, sid, pid, oid, K = 20):

	discovered_paths = []
	#create graph backup

	weight_stack, path_stack, rel_stack = relclosure_sm(G, int(sid), int(pid), int(oid), kind='metric', linkpred=False)
	print "Shortest path for s:{}, p:{}, o:{} is: {}".format(sid, pid, oid, path_stack)
	if not path_stack:
		return discovered_paths
	A = [{'path_total_cost': np.sum(weight_stack),
		'path': path_stack,
		'path_rel': rel_stack,
		'path_weights': weight_stack}]
	B = []
	for k in xrange(1, K):
		for i in xrange(0, len(A[-1]['path'])-1):
			spurNode = A[-1]['path'][i]
			rootPath = A[-1]['path'][:i+1]
			rootPathRel = A[-1]['path_rel'][:i+1]
			rootPathWeights = A[-1]['path_weights'][:i+1]
			#print "rp: {}, rpr: {}, rpw: {}".format(len(rootPath), len(rootPathRel), len(rootPathWeights))
			removed_edges = []
			for path_dict in A:
				if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
					G, removed_edges, flag = delete_edge(G, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1], removed_edges)
					if flag == 0: continue

			while True:
				spurPathWeights, spurPath, spurPathRel = relclosure_sm(G, int(spurNode), int(pid), int(oid), kind='metric', linkpred = True)

				[is_loop, loop_element] = in_lists(spurPath, rootPath[:-1]) ## Check indexing
				if not is_loop:
					break
				else:
					loop_index = spurPath.index(loop_element) # Check indexing
					G, removed_edges, flag = delete_edge(spurPath[loop_index], spurPathRel[loop_index], spurPath[loop_index-1])
					if flag == 0: continue
				if spurPath:
					print("Supplementary path was found!")
					totalPath = rootPath[:-1] + spurPath
					totalDist = np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights)
					totalWeights = rootPathWeights[:-1] + spurPathWeights
					totalPathRel = rootPathRel[:-1] + spurPathRel
					potential_k = {'path_total_cost': totalDist,
								'path': totalPath,
								'path_rel': totalPathRel,
								'path_weights': totalWeights}
					print(totalPath)
					if not (potential_k in B):
						B.append(potential_k)
				# Add back the removed edges
				G = add_edge(G, removed_edges)
				# for removed_edge in removed_edges:
				# 	G.csr[removed_edge[0], removed_edge[2]*G.N + removed_edge[1]] = removed_edge[3]
		if len(B):
			B = sorted(B, key=lambda k: k['path_total_cost'])
			A.append(B[0])
			B.pop(0)
		else:
			break
	# Restore the graph backup
	np.copyto(G.csr.data, G_bak['data'])
	np.copyto(G.csr.indices, G_bak['indices'])
	np.copyto(G.csr.indptr, G_bak['indptr'])

	for path_dict in A:
		discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
	return discovered_paths
##################################################################

# ██    ██ ███████ ███    ██ ██   ██ ███████ ██████  ██   ██
#  ██  ██  ██      ████   ██ ██  ██  ██      ██   ██ ██   ██
#   ████   █████   ██ ██  ██ █████   ███████ ██████  ███████
#    ██    ██      ██  ██ ██ ██  ██       ██ ██           ██
#    ██    ███████ ██   ████ ██   ██ ███████ ██           ██

###################################################################
def yenKSP4(G, sid, pid, oid, K = 20):

	discovered_paths = []
	#create graph backup
	weight_stack, path_stack, rel_stack = relclosure_sm(G, int(sid), int(pid), int(oid), kind='metric', linkpred=True)
	log.info("Shortest path for s:{}, p:{}, o:{} is: {}".format(sid, pid, oid, path_stack))
	#print "Shortest path for s:{}, p:{}, o:{} is: {}".format(sid, pid, oid, path_stack)
	if not path_stack:
		## if the first shortest path is empty, retuen empty discoverd_paths
		return discovered_paths
	A = [{'path_total_cost': np.sum(weight_stack),
		'path': path_stack,
		'path_rel': rel_stack,
		'path_weights': weight_stack}]
	B = []
	for k in xrange(1, K): #for the k-th path, it assumes all paths 1..k-1 are available

		for i in xrange(0, len(A[-1]['path'])-1):
			spurNode = A[-1]['path'][i]
			rootPath = A[-1]['path'][:i+1]
			rootPathRel = A[-1]['path_rel'][:i+1]
			rootPathWeights = A[-1]['path_weights'][:i+1]
			#print "rp: {}, rpr: {}, rpw: {}".format(len(rootPath), len(rootPathRel), len(rootPathWeights))
			removed_edges = []
			for path_dict in A:
				if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
					removed_edges, flag = delete_edge(G, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1], removed_edges)
					if flag == 0:
						continue

			spurPathWeights, spurPath, spurPathRel = relclosure_sm(G, int(spurNode), int(pid), int(oid), kind='metric', linkpred = True)

			if spurPath:
				# print("Supplementary path was found!")
				totalPath = rootPath[:-1] + spurPath
				totalDist = np.sum(rootPathWeights[:]) + np.sum(spurPathWeights[1:])
				totalWeights = rootPathWeights[:] + spurPathWeights[1:]
				totalPathRel = rootPathRel[:] + spurPathRel[1:]
				potential_k = {'path_total_cost': totalDist,
							'path': totalPath,
							'path_rel': totalPathRel,
							'path_weights': totalWeights}
				log.info("totalPath: {}, totalPathRel: {}".format(totalPath, totalPathRel))
				if not (potential_k in B):
					B.append(potential_k)
			# Add back the removed edges
			add_edge(G, removed_edges)
				# for removed_edge in removed_edges:
				# 	G.csr[removed_edge[0], removed_edge[2]*G.N + removed_edge[1]] = removed_edge[3]
			sys.stdout.flush()
		if len(B):
			B = sorted(B, key=lambda k: k['path_total_cost'])
			A.append(B[0])
			B.pop(0)
		else:
			break

	for path_dict in A:
		discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
	return discovered_paths

def predpath_train_model(G, triples, use_interpretable_features=False, cv=10):
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

	# Remove all edges in G corresponding to predicate p.
	pid = triples[0]['pid']

	print '=> Removing predicate {} from KG.'.format(pid)
	eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
	G.csr.data[eraseedges_mask] = 0
	print ''

	# Path extraction
	print '=> Path extraction..(this can take a while)'
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths(G, triples, y)
	print 'P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features))
	vec = DictVectorizer()
	X = vec.fit_transform(measurements)
	n, m = X.shape
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''

	# Path selection
	print '=> Path selection..'
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
	print 'D: +:{}, -:{}, tot:{}'.format(
		len(select_pos_features), len(select_neg_features), X_select.shape[1]
	)
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''

	# Fact interpretation
	if use_interpretable_features and len(select_neg_features) > 0:
		print '=> Fact interpretation..'
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
		print 'D*: +:{}, -:{}, tot:{}'.format(
			len(select_pos_features), len(select_neg_features), X_select.shape[1]
		)
		print 'Time taken: {:.2f}s'.format(time() - t1)
		print ''

	# Model creation
	print '=> Model building..'
	t1 = time()
	model = find_best_model(X_select, y, cv=cv)
	print '#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score'])
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''

	return vec, model

def predict(G, triples, vec, model):
	"""
	Predicts unseen triples using previously built PredPath (PP) model.

	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	vec: DictVectorizer
		For preprocessing triples.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.

	Returns:
	--------
	pred: array
		An array of predicttions, 1 or 0.
	"""
	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].to_dict(orient='records')

	# Path extraction
	print '=> Path extraction.. (this can take a while)'
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths(G, triples, y)
	print 'P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features))
	X = vec.fit_transform(measurements)
	pred = model['clf'].predict(X) # array
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	return pred

def extract_paths(G, triples, y, length=3, features=None):
	"""
	Extracts anchored predicate paths for a given sequence of triples.

	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: sequence
		A list of triples (sid, pid, oid).
	y: array
		A sequence of class labels.
	length: int
		Maximum length of any path.
	features: dict
		Features extracted earlier. A set of (feature_id, path) pairs.
		If None, it is assumed feature set and feature matrix are desired.
		If not None, only X (feature matrix) is returned.

	Returns:
	--------
	features: dict
		A set of (feature_id, path) pairs.
	X: dict
		A dictionary representation of feature matrix.
	"""
	return_features = False
	if features is None:
		return_features = True
		features, pos_features, neg_features = set(), set(), set()
	measurements = []
	for idx, triple in enumerate(triples):
		sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
		label = y[idx]

		# extract paths for a triple
		triple_feature = dict()
		for m in xrange(length + 1):
			if m in [0, 1]: # paths of length 0 and 1 mean nothing
				continue
			paths = c_get_paths(G, sid, pid, oid, length=m, maxpaths=200) # cythonized
			for pth in paths:
				ff = tuple(pth.relational_path) # feature
				# print 'FF was this: {}'.format(ff)
				if ff not in features:
					features.add(ff)
					if label == 1:
						pos_features.add(ff)
					elif label == 0:
						neg_features.add(ff)
					else:
						raise Exception('Unknown class label: {}'.format(label))
				triple_feature[ff] = triple_feature.get(ff, 0) + 1
		measurements.append(triple_feature)
		# print '(T:{}, F:{})'.format(idx+1, len(triple_feature))
		sys.stdout.flush()
	print ''
	if return_features:
		return features, pos_features, neg_features, measurements
	return measurements

def get_paths(G, s, p, o, length=3):
	"Returns all paths of length `length` starting at s and ending in o."
	path_stack = [[s]]
	relpath_stack = [[-1]]
	discoverd_paths = []
	while len(path_stack) > 0:
		# print 'Stack: {} {}'.format(path_stack, relpath_stack)
		curr_path = path_stack.pop()
		curr_relpath = relpath_stack.pop()
		node = curr_path[-1]
		# print 'Node: {}'.format(node)
		if len(curr_path) == length + 1:
			if node == o:
				# create a path
				path = RelationalPath(
					s, p, o, 0., length, curr_path, curr_relpath, np.ones(length+1)
				)
				discoverd_paths.append(path)
			continue
		relnbrs = G.get_neighbors(node)
		for i in xrange(relnbrs.shape[1]):
			rel, nbr = relnbrs[:, i]
			path_stack.append(curr_path + [nbr])
			relpath_stack.append(curr_relpath + [rel])
	return discoverd_paths

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

# =============== MIN-COST FLOW ALGORITHM =================== #
def compute_mincostflow(G, relsim, subs, preds, objs, flowfile):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	relsim: ndarray
		A square matrix containing relational similarity scores.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of
		input triples.
	flowfile: str
		Absolute path of the file where flow will be stored as JSON,
		one line per triple.

	Returns:
	--------
	mincostflows: sequence
		A sequence containing total flow for each triple.
	times: sequence
		Times taken to compute stream of each triple.
	"""
	# take graph backup
	G_bak = {
		'data': G.csr.data.copy(),
		'indices': G.csr.indices.copy(),
		'indptr': G.csr.indptr.copy()
	}
	# Uses the log of indegree to calculate the costs of the successive shortest paths
	## Change this to another metric
	cost_vec_bak = np.log(G.indeg_vec).copy()
	# print "Shape of cost_vec_bak: {}".format(cost_vec_bak.shape)
	# print "cost_vec_bak: {}".format(cost_vec_bak)
	# print "cost_vec_bak, non-zero: {}".format(cost_vec_bak.nonzero())

	# some set up
	G.sources = np.repeat(np.arange(G.N), np.diff(G.csr.indptr))
	G.targets = G.csr.indices % G.N
	cost_vec = cost_vec_bak.copy()
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN)
	specificity_wt = indegsim[G.targets] # specificity
	relations = (G.csr.indices - G.targets) / G.N
	mincostflows, times = [], []
	with open(flowfile, 'w', 0) as ff:
		for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
			s, p, o = [int(x) for x in (s, p, o)]
			ts = time()
			print '{}. Working on {} .. '.format(idx+1, (s, p, o)),
			sys.stdout.flush()

			# set weights
			relsimvec = np.array(relsim[p, :]) # specific to predicate p
			relsim_wt = relsimvec[relations]
			G.csr.data = np.multiply(relsim_wt, specificity_wt)

			# compute
			mcflow = succ_shortest_path(
				G, cost_vec, s, p, o, return_flow=False, npaths=5
			)
			mincostflows.append(mcflow.flow)
			ff.write(json.dumps(mcflow.stream) + '\n')
			tend = time()
			times.append(tend - ts)
			print 'mincostflow: {:.5f}, #paths: {}, time: {:.2f}s.'.format(
				mcflow.flow, len(mcflow.stream['paths']), tend - ts
			)

			# reset state of the graph
			np.copyto(G.csr.data, G_bak['data'])
			np.copyto(G.csr.indices, G_bak['indices'])
			np.copyto(G.csr.indptr, G_bak['indptr'])
			np.copyto(cost_vec, cost_vec_bak)
	return mincostflows, times


def compute_relklinker(G, relsim, subs, preds, objs):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	relsim: ndarray
		A square matrix containing relational similarity scores.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of
		input triples.

	Returns:
	--------
	scores, paths, rpaths, times: sequence
		One sequence each for the proximity scores, shortest path in terms of
		nodes, shortest path in terms of relation sequence, and times taken.
	"""
	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	print 'G.N is : {}'.format(G.N)
	targets = G.csr.indices % G.N
	print 'targets is: {}, size of targets is: {}'.format(targets, targets.shape)
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()
	print 'Shape of CSR.data is: {}'.format(G.csr.data.shape)

	# relation vector
	###########################################
	# THIS IS DIFFERENT THAN USUAL KL
	relations = (G.csr.indices - targets) / G.N
	print 'G.csr.indices has a size: {}'.format(G.csr.indices.shape)
	print '{}'.format(relations)
	###########################################
	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	scores, paths, rpaths, times = [], [], [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
		print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		ts = time()
		# set relational weight

		######################################
		# THIS IS DIFFERENT THAN USUAL KL
		G.csr.data[targets == o] = 1 # no cost for target t => max. specificity.
		relsimvec = relsim[p, :] # specific to predicate p
		relsim_wt = relsimvec[relations] # graph weight
		G.csr.data = np.multiply(relsim_wt, G.csr.data)
		######################################

		rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
		tend = time()
		print 'time: {:.2f}s'.format(tend - ts)
		times.append(tend - ts)
		scores.append(rp.score)
		paths.append(rp.path)
		rpaths.append(rp.relational_path)
		print '{}. Score: {}, path: {}, rpath: {}'.format(idx, rp.score, rp.path, rp.relational_path)
		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	log.info('')
	return scores, paths, rpaths, times

# ================= KNOWLEDGE LINKER ALGORITHM ============

def compute_klinker(G, subs, preds, objs):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of
		input triples.

	Returns:
	--------
	scores, paths, rpaths, times: sequence
		One sequence each for the proximity scores, shortest path in terms of
		nodes, shortest path in terms of relation sequence, and times taken.
	"""
	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	# compute closure
	scores, paths, rpaths, times = [], [], [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
		print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		ts = time()
		rp = closure(G, s, p, o, kind='metric', linkpred=True)
		tend = time()
		print 'time: {:.2f}s'.format(tend - ts)
		times.append(tend - ts)
		scores.append(rp.score)
		paths.append(rp.path)
		rpaths.append(rp.relational_path)
		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	log.info('')
	return scores, paths, rpaths, times
# ================== MAIN CALLING FUNCTION ==================== #

def normalize(df):
	softmax = lambda x: np.exp(x) / float(np.exp(x).sum())
	df['softmaxscore'] = df[['sid','score']].groupby(by=['sid'], as_index=False).transform(softmax)
	return df

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

	# logging
	# Alex 1/9/18 - Commented out to add more logging stuff for debugging
	#disable_logging(log.DEBUG)


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

	if args.method == 'stream': # KNOWLEDGE STREAM (KS)
		# compute min. cost flow
		log.info('Computing KS for {} triples..'.format(spo_df.shape[0]))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			outjson = join(args.outdir, 'out_kstream_{}_{}.json'.format(base, DATE))
			outcsv = join(args.outdir, 'out_kstream_{}_{}.csv'.format(base, DATE))
			mincostflows, times = compute_mincostflow(G, relsim, subs, preds, objs, outjson)
			# save the results
			spo_df['score'] = mincostflows
			spo_df['time'] = times
			spo_df = normalize(spo_df)
			spo_df.to_csv(outcsv, sep=',', header=True, index=False)
			log.info('* Saved results: %s' % outcsv)
		log.info('Mincostflow computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'relklinker': # RELATIONAL KNOWLEDGE LINKER (KL-REL)
		log.info('Computing KL-REL for {} triples..'.format(spo_df.shape[0]))
		scores, paths, rpaths, times = compute_relklinker(G, relsim, subs, preds, objs)
		# save the results
		spo_df['score'] = scores
		spo_df['path'] = paths
		spo_df['rpath'] = rpaths
		spo_df['time'] = times
		spo_df = normalize(spo_df)
		outcsv = join(args.outdir, 'out_relklinker_{}_{}.csv'.format(base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		log.info('* Saved results: %s' % outcsv)
		log.info('Relatioanal KL computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'klinker':
		log.info('Computing KL for {} triples..'.format(spo_df.shape[0]))
		scores, paths, rpaths, times = compute_klinker(G, subs, preds, objs)
		# save the results
		spo_df['score'] = scores
		spo_df['path'] = paths
		spo_df['rpath'] = rpaths
		spo_df['time'] = times
		spo_df = normalize(spo_df)
		outcsv = join(args.outdir, 'out_klinker_{}_{}.csv'.format(base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		log.info('* Saved results: %s' % outcsv)
		log.info('KL computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'predpath': # PREDPATH
		vec, model = predpath_train_model(G, spo_df) # train
		# vec, model = predpath_train_model(G, spo_df, relsim)
		print 'Time taken: {:.2f}s\n'.format(time() - t1)
		# save model
		predictor = { 'dictvectorizer': vec, 'model': model }
		try:
			outpkl = join(args.outdir, 'out_predpath_{}_{}.pkl'.format(base, DATE))
			with open(outpkl, 'wb') as g:
				pkl.dump(predictor, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved: {}'.format(outpkl)
		except IOError, e:
			raise e
	elif args.method == 'sm':
		vec, model = predpath_train_model_sm(G, spo_df, relsim) # train
		print 'Time taken: {:.2f}s\n'.format(time() - t1)
		log.info('Time taken: {:.2f}s\n'.format(time() - t1))
		# save model
		predictor = { 'dictvectorizer': vec, 'model': model }
		try:
			outpkl = join(args.outdir, 'out_streamminer_{}_{}.pkl'.format(base, DATE))
			with open(outpkl, 'wb') as g:
				pkl.dump(predictor, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved: {}'.format(outpkl)
		except IOError, e:
			raise e
	print '\nDone!\n'

if __name__ == '__main__':
	main()
