import gc
import logging as log
from tqdm import tqdm
from algorithms.sm.yenKSP import yenKSP
import multiprocessing as mp

cpdef extract_paths_sm_par(Gv, Gr, triples, y, features=None):
    return_features = False
    if features is None:
        return_features = True
        features, pos_features, neg_features = set(), set(), set()
    measurements = []
    
    cdef int sid, pid, oid, label
    data = []
    for label, triple in zip(y, triples):
        data.append((triple, label, Gv, Gr))
    input_queue = mp.Queue()
    pool = mp.Pool(mp.cpu_count()-1)
    _max = len(data)
    with tqdm(total=_max) as pbar:
        for i, result in tqdm(enumerate(pool.imap_unordered(get_features, data))):
            pbar.update()
            measurements.append(result[1])
            for i in result[2]:
                features.add(i)
            for i in result[3]:
                pos_features.add(i)
            for i in result[4]:
                neg_features.add(i)
    pbar.close()
    pool.close()
    if return_features:
        return features, pos_features, neg_features, measurements
    return measurements

def listener(q):
    pbar = tqdm(total=q.size())
    for item in iter(q.get, None):
        pbar.update()

def get_features(data, features=None):
    triple, label, Gv, Gr = data[0],data[1],data[2],data[3]
    if features is None:
        features, pos_features, neg_features = set(), set(), set()
    sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
    triple_feature = dict()
    discovered_paths = yenKSP(Gv, Gr, sid, pid, oid, K = 5)
    for path in discovered_paths:
        log.info("{}\n".format(path))
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
    return (label, triple_feature, features, pos_features, neg_features)
