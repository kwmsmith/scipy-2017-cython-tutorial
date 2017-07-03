ctypedef void (* random_float_fill)(aug_state* state, np.npy_intp count, float *out) nogil
ctypedef void (* random_double_fill)(aug_state* state, np.npy_intp count, double *out) nogil




cdef object double_fill(aug_state* state, void *func, object size, object lock, object out):
    cdef random_double_fill f = <random_double_fill>func
    cdef double fout
    cdef double *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp n

    if size is None and out is None:
        with lock:
            f(state, 1, &fout)
        return fout

    if out is not None:
        check_output(out, np.float64, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float64)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <double *>np.PyArray_DATA(out_array)
    with lock, nogil:
        f(state, n, out_array_data)
    return out_array


cdef object float_fill(aug_state* state, void *func, object size, object lock, object out):
    cdef random_float_fill f = <random_float_fill>func
    cdef float fout
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp n

    if size is None and out is None:
        with lock:
            f(state, 1, &fout)
        return fout

    if out is not None:
        check_output(out, np.float32, size)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <float *>np.PyArray_DATA(out_array)
    with lock, nogil:
        f(state, n, out_array_data)
    return out_array


cdef object float_fill_from_double(aug_state* state, void *func, object size, object lock):
    cdef random_double_fill f = <random_double_fill>func
    cdef double out
    cdef double* buffer
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n, buffer_size, remaining, total, fetch

    if size is None:
        with lock:
            f(state, 1, &out)
        return <float>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)
        n = np.PyArray_SIZE(out_array)
        buffer_size = 1000 if n > 1000 else n
        total = 0
        remaining = n
        buffer = <double *> malloc(buffer_size * sizeof(double))

        out_array_data = <float *>np.PyArray_DATA(out_array)
        while remaining > 0:
            fetch = 1000 if remaining > 1000 else remaining
            with lock, nogil:
                f(state, fetch, buffer)
            for i in range(fetch):
                out_array_data[i + total] = <float> buffer[i]
            total += fetch
            remaining -= fetch

        free(buffer)

        return out_array
