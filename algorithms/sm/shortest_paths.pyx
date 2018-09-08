"""
Routines for performing shortest-path graph searches
The main interface is in the function :func:`shortest_path`.  This
calls cython routines that compute the shortest path using
the Floyd-Warshall algorithm, Dijkstra's algorithm with Fibonacci Heaps,
the Bellman-Ford algorithm, or Johnson's Algorithm.
"""

# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2011
from __future__ import absolute_import

import warnings

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csr, isspmatrix_csc
from scipy.sparse.csgraph._validation import validate_graph

cimport cython

from libc.stdlib cimport malloc, free
from numpy.math cimport INFINITY

include 'parameters.pxi'


class NegativeCycleError(Exception):
    def __init__(self, message=''):
        Exception.__init__(self, message)

def dijkstra(csgraph, directed=True, indices=None,
             return_predecessors=False,
             unweighted=False, limit=np.inf, target=None):
    """
    dijkstra(csgraph, directed=True, indices=None, return_predecessors=False,
             unweighted=False, limit=np.inf)
    Dijkstra algorithm using Fibonacci Heaps
    .. versionadded:: 0.11.0
    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        The N x N array of non-negative distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j] and from
        point j to i along paths csgraph[j, i].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j or j to i along either
        csgraph[i, j] or csgraph[j, i].
    indices : array_like or int, optional
        if specified, only compute the paths for the points at the given
        indices.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecesor matrix
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    limit : float, optional
        The maximum distance to calculate, must be >= 0. Using a smaller limit
        will decrease computation time by aborting calculations between pairs
        that are separated by a distance > limit. For such pairs, the distance
        will be equal to np.inf (i.e., not connected).
        .. versionadded:: 0.14.0
    Returns
    -------
    dist_matrix : ndarray
        The matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.
    predecessors : ndarray
        Returned only if return_predecessors == True.
        The matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999
    Notes
    -----
    As currently implemented, Dijkstra's algorithm does not work for
    graphs with direction-dependent distances when directed == False.
    i.e., if csgraph[i,j] and csgraph[j,i] are not equal and
    both are nonzero, setting directed=False will not yield the correct
    result.
    Also, this routine does not work for graphs with negative
    distances.  Negative distances can lead to infinite cycles that must
    be handled by specialized algorithms such as Bellman-Ford's algorithm
    or Johnson's algorithm.
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import dijkstra
    >>> graph = [
    ... [0, 1 , 2, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (0, 1)	1
      (0, 2)	2
      (1, 3)	1
      (2, 3)	3
    >>> dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
    >>> dist_matrix
    array([ 0.,  1.,  2.,  2.])
    >>> predecessors
    array([-9999,     0,     0,     1], dtype=int32)
    """
    #------------------------------
    # validate csgraph and convert to csr matrix
    csgraph = validate_graph(csgraph, directed, DTYPE,
                             dense_output=False)

    if np.any(csgraph.data < 0):
        warnings.warn("Graph has negative weights: dijkstra will give "
                      "inaccurate results if the graph contains negative "
                      "cycles. Consider johnson or bellman_ford.")

    N = csgraph.shape[0]

    #------------------------------
    # initialize/validate indices
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
        return_shape = indices.shape + (N,)
    else:
        indices = np.array(indices, order='C', dtype=ITYPE, copy=True)
        return_shape = indices.shape + (N,)
        indices = np.atleast_1d(indices).reshape(-1)
        indices[indices < 0] += N
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")

    cdef DTYPE_t limitf = limit
    if limitf < 0:
        raise ValueError('limit must be >= 0')

    #------------------------------
    # initialize dist_matrix for output
    dist_matrix = np.zeros((len(indices), N), dtype=DTYPE)
    dist_matrix.fill(np.inf)
    dist_matrix[np.arange(len(indices)), indices] = 0

    #------------------------------
    # initialize predecessors for output
    if return_predecessors:
        predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
        predecessor_matrix.fill(NULL_IDX)
    else:
        predecessor_matrix = np.empty((0, N), dtype=ITYPE)

    if unweighted:
        csr_data = np.ones(csgraph.data.shape)
    else:
        csr_data = csgraph.data

    if directed:
        _dijkstra_directed(indices,
                           csr_data, csgraph.indices, csgraph.indptr,
                           dist_matrix, predecessor_matrix, limitf, target)
    else:
        csgraphT = csgraph.T.tocsr()
        if unweighted:
            csrT_data = csr_data
        else:
            csrT_data = csgraphT.data
        _dijkstra_undirected(indices,
                             csr_data, csgraph.indices, csgraph.indptr,
                             csrT_data, csgraphT.indices, csgraphT.indptr,
                             dist_matrix, predecessor_matrix, limitf)

    if return_predecessors:
        return (dist_matrix.reshape(return_shape),
                predecessor_matrix.reshape(return_shape))
    else:
        return dist_matrix.reshape(return_shape)


cdef _dijkstra_directed(
            np.ndarray[ITYPE_t, ndim=1, mode='c'] source_indices,
            np.ndarray[DTYPE_t, ndim=1, mode='c'] csr_weights,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indices,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indptr,
            np.ndarray[DTYPE_t, ndim=2, mode='c'] dist_matrix,
            np.ndarray[ITYPE_t, ndim=2, mode='c'] pred,
            DTYPE_t limit,
            ITYPE_t target):
    cdef unsigned int Nind = dist_matrix.shape[0]
    cdef unsigned int N = dist_matrix.shape[1]
    cdef unsigned int i, k, j_source, j_current
    cdef ITYPE_t j

    cdef DTYPE_t next_val

    cdef int return_pred = (pred.size > 0)

    cdef FibonacciHeap heap
    cdef FibonacciNode *v
    cdef FibonacciNode *current_node
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(N *
                                                        sizeof(FibonacciNode))

    for i in range(Nind):
        j_source = source_indices[i]

        for k in range(N):
            initialize_node(&nodes[k], k)

        dist_matrix[i, j_source] = 0
        heap.min_node = NULL
        insert_node(&heap, &nodes[j_source])

        while heap.min_node:
            v = remove_min(&heap)
            v.state = SCANNED
            # Early stop on reaching target
            if v.index == target:
                dist_matrix[i, v.index] = v.val
                break
            for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
                j_current = csr_indices[j]
                current_node = &nodes[j_current]
                if current_node.state != SCANNED:
                    next_val = v.val + csr_weights[j]
                    if next_val <= limit:
                        if current_node.state == NOT_IN_HEAP:
                            current_node.state = IN_HEAP
                            current_node.val = next_val
                            insert_node(&heap, current_node)
                            if return_pred:
                                pred[i, j_current] = v.index
                        elif current_node.val > next_val:
                            decrease_val(&heap, current_node,
                                         next_val)
                            if return_pred:
                                pred[i, j_current] = v.index

            #v has now been scanned: add the distance to the results
            dist_matrix[i, v.index] = v.val

    free(nodes)


cdef _dijkstra_undirected(
            np.ndarray[ITYPE_t, ndim=1, mode='c'] source_indices,
            np.ndarray[DTYPE_t, ndim=1, mode='c'] csr_weights,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indices,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indptr,
            np.ndarray[DTYPE_t, ndim=1, mode='c'] csrT_weights,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csrT_indices,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csrT_indptr,
            np.ndarray[DTYPE_t, ndim=2, mode='c'] dist_matrix,
            np.ndarray[ITYPE_t, ndim=2, mode='c'] pred,
            DTYPE_t limit):
    cdef unsigned int Nind = dist_matrix.shape[0]
    cdef unsigned int N = dist_matrix.shape[1]
    cdef unsigned int i, k, j_source, j_current
    cdef ITYPE_t j

    cdef DTYPE_t next_val

    cdef int return_pred = (pred.size > 0)

    cdef FibonacciHeap heap
    cdef FibonacciNode *v
    cdef FibonacciNode *current_node
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(N *
                                                        sizeof(FibonacciNode))

    for i in range(Nind):
        j_source = source_indices[i]

        for k in range(N):
            initialize_node(&nodes[k], k)

        dist_matrix[i, j_source] = 0
        heap.min_node = NULL
        insert_node(&heap, &nodes[j_source])

        while heap.min_node:
            v = remove_min(&heap)
            v.state = SCANNED
            ## Stopping here
            for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
                j_current = csr_indices[j]
                current_node = &nodes[j_current]
                if current_node.state != SCANNED:
                    next_val = v.val + csr_weights[j]
                    if next_val <= limit:
                        if current_node.state == NOT_IN_HEAP:
                            current_node.state = IN_HEAP
                            current_node.val = next_val
                            insert_node(&heap, current_node)
                            if return_pred:
                                pred[i, j_current] = v.index
                        elif current_node.val > next_val:
                            decrease_val(&heap, current_node,
                                         next_val)
                            if return_pred:
                                pred[i, j_current] = v.index

            for j in range(csrT_indptr[v.index], csrT_indptr[v.index + 1]):
                j_current = csrT_indices[j]
                current_node = &nodes[j_current]
                if current_node.state != SCANNED:
                    next_val = v.val + csrT_weights[j]
                    if next_val <= limit:
                        if current_node.state == NOT_IN_HEAP:
                            current_node.state = IN_HEAP
                            current_node.val = next_val
                            insert_node(&heap, current_node)
                            if return_pred:
                                pred[i, j_current] = v.index
                        elif current_node.val > next_val:
                            decrease_val(&heap, current_node, next_val)
                            if return_pred:
                                pred[i, j_current] = v.index

            #v has now been scanned: add the distance to the results
            dist_matrix[i, v.index] = v.val

    free(nodes)


######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#
cdef enum FibonacciState:
    SCANNED
    NOT_IN_HEAP
    IN_HEAP


cdef struct FibonacciNode:
    unsigned int index
    unsigned int rank
    FibonacciState state
    DTYPE_t val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children


cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0):
    # Assumptions: - node is a valid pointer
    #              - node is not currently part of a heap
    node.index = index
    node.val = val
    node.rank = 0
    node.state = NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL


cdef FibonacciNode* rightmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.right_sibling):
        temp = temp.right_sibling
    return temp


cdef FibonacciNode* leftmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp


cdef void add_child(FibonacciNode* node, FibonacciNode* new_child):
    # Assumptions: - node is a valid pointer
    #              - new_child is a valid pointer
    #              - new_child is not the sibling or child of another node
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:
        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1


cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling):
    # Assumptions: - node is a valid pointer
    #              - new_sibling is a valid pointer
    #              - new_sibling is not the child or sibling of another node
    cdef FibonacciNode* temp = rightmost_sibling(node)
    temp.right_sibling = new_sibling
    new_sibling.left_sibling = temp
    new_sibling.right_sibling = NULL
    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1


cdef void remove(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    if node.parent:
        node.parent.rank -= 1
        if node.left_sibling:
            node.parent.children = node.left_sibling
        elif node.right_sibling:
            node.parent.children = node.right_sibling
        else:
            node.parent.children = NULL

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.


cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    if heap.min_node:
        add_sibling(heap.min_node, node)
        if node.val < heap.min_node.val:
            heap.min_node = node
    else:
        heap.min_node = node


cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval):
    # Assumptions: - heap is a valid pointer
    #              - newval <= node.val
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    #              - node is in the heap
    node.val = newval
    if node.parent and (node.parent.val >= newval):
        remove(node)
        insert_node(heap, node)
    elif heap.min_node.val > node.val:
        heap.min_node = node


cdef void link(FibonacciHeap* heap, FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is already within heap

    cdef FibonacciNode *linknode
    cdef FibonacciNode *parent
    cdef FibonacciNode *child

    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)


cdef FibonacciNode* remove_min(FibonacciHeap* heap):
    # Assumptions: - heap is a valid pointer
    #              - heap.min_node is a valid pointer
    cdef FibonacciNode *temp
    cdef FibonacciNode *temp_right
    cdef FibonacciNode *out
    cdef unsigned int i

    # make all min_node children into root nodes
    if heap.min_node.children:
        temp = leftmost_sibling(heap.min_node.children)
        temp_right = NULL

        while temp:
            temp_right = temp.right_sibling
            remove(temp)
            add_sibling(heap.min_node, temp)
            temp = temp_right

        heap.min_node.children = NULL

    # choose a root node other than min_node
    temp = leftmost_sibling(heap.min_node)
    if temp == heap.min_node:
        if heap.min_node.right_sibling:
            temp = heap.min_node.right_sibling
        else:
            out = heap.min_node
            heap.min_node = NULL
            return out

    # remove min_node, and point heap to the new min
    out = heap.min_node
    remove(heap.min_node)
    heap.min_node = temp

    # re-link the heap
    for i in range(100):
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    return out


######################################################################
# Debugging: Functions for printing the Fibonacci heap
#
#cdef void print_node(FibonacciNode* node, int level=0):
#    print '%s(%i,%i) %i' % (level*'   ', node.index, node.val, node.rank)
#    if node.children:
#        print_node(leftmost_sibling(node.children), level+1)
#    if node.right_sibling:
#        print_node(node.right_sibling, level)
#
#
#cdef void print_heap(FibonacciHeap* heap):
#    print "---------------------------------"
#    print "min node: (%i, %i)" % (heap.min_node.index, heap.min_node.val)
#    if heap.min_node:
#        print_node(leftmost_sibling(heap.min_node))
#    else:
#        print "[empty heap]"
