# coding: utf-8
import argparse, gc, sys, os
import numpy as np
import pandas as pd
import warnings
import ujson as json
import logging as log
from tqdm import tqdm
###### Cython benign warning ignore ##########################
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
###############################################################
from os.path import expanduser, abspath, basename, splitext, join, exists
from time import time, strftime, localtime
import cPickle as pkl
#####################################
from datastructures.rgraph import Graph, weighted_degree
#####################################
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from scipy.sparse import save_npz, load_npz, csr_matrix
##############################################
from algorithms.pra.pra_mining import find_best_model
from algorithms.klinker.closure import closure
from algorithms.mincostflow.ssp import succ_shortest_path
from algorithms.pathenum import get_paths as c_get_paths
from algorithms.sm.extract_paths import extract_paths_sm
from algorithms.sm.extract_paths_par import extract_paths_sm_par
from algorithms.relklinker.rel_closure import relational_closure as relclosure
##############################################
from memory_profiler import profile

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float
inf = float('inf')

# Date
DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))

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

# ██████   █████  ███████ ███████     ███████ ███    ███
# ██   ██ ██   ██ ██      ██          ██      ████  ████
# ██████  ███████ ███████ █████       ███████ ██ ████ ██
# ██   ██ ██   ██      ██ ██               ██ ██  ██  ██
# ██████  ██   ██ ███████ ███████     ███████ ██      ██

@profile
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
        log.info('=> Removing predicate {} from KG.\n\n'.format(pid))
        eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
        specificity_wt[eraseedges_mask] = 0
        relsim_wt[eraseedges_mask] = 0
        G.csr.data = specificity_wt.copy()

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
    features, pos_features, neg_features, measurements = extract_paths_sm_par(Gv, Gr, triples, y)
    gc.collect()
    log.info('P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features)))
    vec = DictVectorizer()
    X = vec.fit_transform(measurements)
    n, m = X.shape
    log.info('Time taken: {:.2f}s\n\n'.format(time() - t1))

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
    t1 = time()
    model = find_best_model(X_select, y, cv=cv)
    log.info('#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score']))
    log.info('Time taken: {:.2f}s\n'.format(time() - t1))

    return vec, model

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
