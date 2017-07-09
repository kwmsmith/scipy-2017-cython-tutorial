cdef extern from "aligned_malloc.h":
    cdef void *PyArray_realloc_aligned(void *p, size_t n);
    cdef void *PyArray_malloc_aligned(size_t n);
    cdef void *PyArray_calloc_aligned(size_t n, size_t s);
    cdef void PyArray_free_aligned(void *p);
