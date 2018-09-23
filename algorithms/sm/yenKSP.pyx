# cython: profile=True
import sys
import numpy as np
cimport numpy as np
from datastructures.relationalpath_sm import RelationalPathSM
from algorithms.sm.rel_closure_2 import relational_closure_sm as relclosure_sm
from streamminer2 import delete_edge, delete_node

cpdef yenKSP(Gv, Gr, int sid, int pid, int oid, int K = 5):
    cdef int start
    cdef int end
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
    for k in xrange(1, K): #for the k-th path, it assumes all paths 1..k-1 are available
        for i in xrange(0, len(A[-1]['path'])-1):
            # the spurnode ranges from first node of the previous (k-1) shortest path to its next to last node.
            #spurNode = A[-1]['path'][i]
            rootPath = A[-1]['path'][:i+1]
            #rootPathRel = A[-1]['path_rel'][:i+1]
            #rootPathWeights = A[-1]['path_weights'][:i+1]
            # print "SpurNode: {}, Rootpath: {}".format(spurNode, rootPath)
            removed_edges = []
            removed_nodes = []
            for path_dict in A:
                if len(path_dict['path']) > i and rootPath == path_dict['path'][:i+1]:
                    removed_edges.extend( delete_edge(Gv, Gr, path_dict['path'][i], path_dict['path_rel'][i+1], path_dict['path'][i+1]) )
                    for rootPathNode in rootPath[:-1]:
                        removed_nodes.extend( delete_node(Gv, Gr, rootPathNode) )
                spurPathWeights, spurPath, spurPathRel = relclosure_sm(Gv, Gr, A[-1]['path'][i], pid, oid, kind='metric', linkpred = False)
                if spurPath and spurPathRel != [-1]:
                    #totalPath = rootPath[:-1] + spurPath
                    #totalDist = np.sum(rootPathWeights[:-1]) + np.sum(spurPathWeights[:-1])
                    #totalWeights = rootPathWeights[:-1] + spurPathWeights[:]
                    #totalPathRel = rootPathRel[:] + spurPathRel[1:]
                    potential_k = {'path_total_cost': np.sum(A[-1]['path_weights'][:i+1][:-1]) + np.sum(spurPathWeights[:-1]), #totalDist
                               'path': list(rootPath[:-1]+ spurPath), #totalPath
                               'path_rel': list(A[-1]['path_rel'][:i+1] + spurPathRel[1:]), #totalPathRel && roothPathRel-->np.array(A[-1]['path_rel'][:i+1]
                               'path_weights': list(A[-1]['path_weights'][:i+1][:-1] + spurPathWeights[:])} #totalWeights
                if not (potential_k in B or potential_k in A):
                    # removes repetitive paths in A & B
                    B.append(potential_k)
            removed_nodes.reverse()
            #add_node code
            for removedNode in removed_nodes:
                start = Gr.indptr[removedNode[0]]
                end = Gr.indptr[removedNode[0]+1]
                Gv.data[start:end] = removedNode[1]
            removed_edges.reverse()
            #add_edge code
            for removed_edge in removed_edges:
                s, o, p, cost = removed_edge
                start = Gr.indptr[s]
                end = Gr.indptr[s+1]
                neighbors = Gr.indices[start:end]
                rels = Gr.data[start:end]
                Gv.data[start + np.where(np.logical_and(neighbors == o, rels == p))] = cost
            del removed_edges
            del removed_nodes
        if len(B):
            mergeSort(B,0,len(B)-1)
            A.append(B[0])
            del B[0]
        else:
            break
    for path_dict in A:
        discovered_paths.append(RelationalPathSM(sid, pid, oid, path_dict['path_total_cost'], len(path_dict['path'])-1, path_dict['path'], path_dict['path_rel'], path_dict['path_weights']))
    return discovered_paths
        
cpdef merge(arr, int l, int m, int r): 
    cdef int n1 = m - l + 1
    cdef int n2 = r- m
    L = np.zeros(n1)
    R = np.zeros(n2)
    cdef int i = 0     # Initial index of first subarray 
    cdef int j = 0     # Initial index of second subarray 
    cdef int k = l     # Initial index of merged subarray 

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