import gc
import logging as log
from tqdm import tqdm
from algorithms.sm.yenKSP import yenKSP

cpdef extract_paths_sm(Gv, Gr, triples, y, features=None):
    return_features = False
    if features is None:
        return_features = True
        features, pos_features, neg_features = set(), set(), set()
    measurements = []
    
    cdef int sid, pid, oid, label
    for label, triple in tqdm(zip(y, triples), total=len(triples)):
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
        measurements.append(triple_feature)
        gc.collect()
    if return_features:
        return features, pos_features, neg_features, measurements
    return measurements
