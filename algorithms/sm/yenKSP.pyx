import numpy as np
from algorithms.sm.rel_closure_2 import relational_closure_sm as relclosure_sm
from datastructures.relationalpath_sm import RelationalPathSM

def yenKSP(Gv, Gr, int sid, int pid, int oid, K = 5):
    cdef int s, o, p
    cdef float cost
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
            rootPath = A[-1]['path'][:i+1]
            rootPathWeights = A[-1]['path_weights'][:i+1]
            removed_edges[:] = []
            removed_nodes[:] = []
            for path_dict in A:
                if len(path_dict['path']) > i and np.array_equal(rootPath, path_dict['path'][:i+1]):
                    removed_edges.extend( delete_edge(Gv, Gr, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1]) )
            for rootPathNode in rootPath[:-1]:
                start = Gr.indptr[s]
                end = Gr.indptr[s+1]
                Gv.data[start:end] = np.inf
                removed_nodes.append( (s, np.copy(Gv.data[start:end])) )
            spurPathWeights, spurPath, spurPathRel = relclosure_sm(Gv, Gr, A[-1]['path'][i], pid, oid, kind='metric', linkpred = False)
            if spurPath and spurPathRel != [-1]:
                potential_k = {'path_total_cost': np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights[:-1]),
                                'path': list(rootPath[:-1] + spurPath),
                                'path_rel': list(A[-1]['path_rel'][:i+1] + spurPathRel[1:]),
                                'path_weights': list(rootPathWeights[:-1] + spurPathWeights[:])}
                if not (potential_k in B or potential_k in A):
                    # removes repititive paths in A & B
                    B.append(potential_k)
            removed_nodes.reverse()
#            add_node(Gv, Gr, removed_nodes)
            for removedNode in removed_nodes:
                start = Gr.indptr[removedNode[0]]
                end = Gr.indptr[removedNode[0]+1]
                Gv.data[start:end] = removedNode[1]
            removed_edges.reverse()
#            add_edge(Gv, Gr, removed_edges)
            for removed_edge in removed_edges:
                s, o, p, cost = removed_edge
                start = Gr.indptr[s]
                end = Gr.indptr[s+1]
                Gv.data[start + np.where(np.logical_and(Gr.indices[start:end] == o, Gr.data[start:end] == p))] = cost
        if len(B):
            mergeSort(B,0,len(B)-1)
            A.append(B[0])
            B.pop(0)
        else:
            break
    for path_dict in A:
        discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
    return discovered_paths

cpdef delete_edge(Gv, Gr, int s, int p, int o):
    cdef int pos
    deletedEdges = []

    # deleting the edge: s --> o
    start = Gr.indptr[s]
    end = Gr.indptr[s+1]
    pos = start + np.where(np.logical_and(Gr.indices[start:end] == o, Gr.data[start:end] == p))
    deletedEdges.append((s, o, p, Gv.data[pos]))
    Gv.data[pos] = np.inf

    # deleting the edge: o --> s
    start = Gr.indptr[o]
    end = Gr.indptr[o+1]
    pos = start + np.where(np.logical_and(Gr.indices[start:end] == s, Gr.data[start:end] == p))
    deletedEdges.append((o, s, p, Gv.data[pos]))
    Gv.data[pos] = np.inf

    return deletedEdges
        
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
