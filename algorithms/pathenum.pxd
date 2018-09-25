cimport cython

cpdef get_paths(G, s, p, o, length=*, maxpaths=*)
cpdef get_paths_sm(G, s, p, o, relsim_wt, weights=*, maxpaths=*)

cdef object enumerate_paths(
		double[:] data, long[:] indices, int[:] indptr,
		int s, int p, int o, int length=*, int maxpaths=*
	)

cdef object enumerate_paths_sm(
		double[:] data, long[:] indices, int[:] indptr,
		int s, int p, int o, double weight=*, int maxpaths=*
	)
