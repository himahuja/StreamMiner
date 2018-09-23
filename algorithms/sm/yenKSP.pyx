import numpy as np
from algorithms.sm.rel_closure_2 import relational_closure_sm as relclosure_sm
from datastructures.relationalpath_sm import RelationalPathSM

def yenKSP(Gv, Gr, int sid, int pid, int oid, K = 5):
    discovered_paths = []
    weight_stack, path_stack, rel_stack = relclosure_sm(Gv, Gr, sid, pid, oid, kind='metric', linkpred = False)
    if rel_stack == [-1]:
        ## if the first shortest path is empty, return empty discovered_paths
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
            # the spurnode ranges from first node of the previous (k-1) shortest path to its next to last node.
            spurNode = A[-1]['path'][i]
            rootPath = A[-1]['path'][:i+1]
            rootPathRel = A[-1]['path_rel'][:i+1]
            rootPathWeights = A[-1]['path_weights'][:i+1]
            removed_edges[:] = []
            removed_nodes[:] = []
            for path_dict in A:
                if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
                    removed_edges.extend( delete_edge(Gv, Gr, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1]) )
            for rootPathNode in rootPath[:-1]:
                removed_nodes.extend( delete_node(Gv, Gr, rootPathNode) )
            spurPathWeights, spurPath, spurPathRel = relclosure_sm(Gv, Gr, int(spurNode), pid, oid, kind='metric', linkpred = False)
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
            mergeSort(B,0,len(B)-1)
            A.append(B[0])
            B.pop(0)
        else:
            break
    for path_dict in A:
        discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
    return discovered_paths

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
        
cpdef merge(arr, int l, int m, int r): 
    cdef int n1 = m - l + 1
    cdef int n2 = r- m
    cdef L = np.empty(n1, dtype=object)
    cdef R = np.empty(n2, dtype=object)
    cdef int i, j
    cdef int k = l

    for i in xrange(0 , n1): 
        L[i] = arr[l + i] 
    for j in xrange(0 , n2): 
        R[j] = arr[m + 1 + j]
    
    i = 0
    j = 0
  
    while i < n1 and j < n2 : 
        if L[i]['path_total_cost'] <= R[j]['path_total_cost']: 
            arr[k] = L[i] 
            i += 1
        else: 
            arr[k] = R[j] 
            j += 1
        k += 1
    while i < n1: 
        arr[k] = L[i] 
        i += 1
        k += 1
    while j < n2: 
        arr[k] = R[j] 
        j += 1
        k += 1
  
cpdef mergeSort(arr, int l, int r):
    cdef int m = (l+(r-1))/2
    if l < r:
        mergeSort(arr, l, m) 
        mergeSort(arr, m+1, r) 
        merge(arr, l, m, r)
