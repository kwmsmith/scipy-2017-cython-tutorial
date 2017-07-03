_randint_type = {'bool': (0, 2),
                 'int8': (-2**7, 2**7),
                 'int16': (-2**15, 2**15),
                 'int32': (-2**31, 2**31),
                 'int64': (-2**63, 2**63),
                 'uint8': (0, 2**8),
                 'uint16': (0, 2**16),
                 'uint32': (0, 2**32),
                 'uint64': (0, 2**64)
                 }

ctypedef np.npy_bool bool_t





cdef object _rand_int64(low, high, size, aug_state *state, lock):
    """
    _rand_int64(self, low, high, size, rngstate)

    Return random np.int64 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int64 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.int64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint64_t off, rng, buf
    cdef uint64_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint64_t>(high - low)
    off = <uint64_t>(<int64_t>low)
    if size is None:
        with lock:
            random_bounded_uint64_fill(state, off, rng, 1, &buf)
        return np.int64(<int64_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.int64)
        cnt = np.PyArray_SIZE(array)
        out = <uint64_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint64_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_int32(low, high, size, aug_state *state, lock):
    """
    _rand_int32(self, low, high, size, rngstate)

    Return random np.int32 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int32 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.int32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint32_t off, rng, buf
    cdef uint32_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint32_t>(high - low)
    off = <uint32_t>(<int32_t>low)
    if size is None:
        with lock:
            random_bounded_uint32_fill(state, off, rng, 1, &buf)
        return np.int32(<int32_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.int32)
        cnt = np.PyArray_SIZE(array)
        out = <uint32_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint32_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_int16(low, high, size, aug_state *state, lock):
    """
    _rand_int16(self, low, high, size, rngstate)

    Return random np.int16 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int16 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.int16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint16_t off, rng, buf
    cdef uint16_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint16_t>(high - low)
    off = <uint16_t>(<int16_t>low)
    if size is None:
        with lock:
            random_bounded_uint16_fill(state, off, rng, 1, &buf)
        return np.int16(<int16_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.int16)
        cnt = np.PyArray_SIZE(array)
        out = <uint16_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint16_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_int8(low, high, size, aug_state *state, lock):
    """
    _rand_int8(self, low, high, size, rngstate)

    Return random np.int8 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int8 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.int8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint8_t off, rng, buf
    cdef uint8_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint8_t>(high - low)
    off = <uint8_t>(<int8_t>low)
    if size is None:
        with lock:
            random_bounded_uint8_fill(state, off, rng, 1, &buf)
        return np.int8(<int8_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.int8)
        cnt = np.PyArray_SIZE(array)
        out = <uint8_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint8_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_uint64(low, high, size, aug_state *state, lock):
    """
    _rand_uint64(self, low, high, size, rngstate)

    Return random np.uint64 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint64 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.uint64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint64_t off, rng, buf
    cdef uint64_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint64_t>(high - low)
    off = <uint64_t>(<uint64_t>low)
    if size is None:
        with lock:
            random_bounded_uint64_fill(state, off, rng, 1, &buf)
        return np.uint64(<uint64_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.uint64)
        cnt = np.PyArray_SIZE(array)
        out = <uint64_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint64_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_uint32(low, high, size, aug_state *state, lock):
    """
    _rand_uint32(self, low, high, size, rngstate)

    Return random np.uint32 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint32 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.uint32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint32_t off, rng, buf
    cdef uint32_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint32_t>(high - low)
    off = <uint32_t>(<uint32_t>low)
    if size is None:
        with lock:
            random_bounded_uint32_fill(state, off, rng, 1, &buf)
        return np.uint32(<uint32_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.uint32)
        cnt = np.PyArray_SIZE(array)
        out = <uint32_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint32_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_uint16(low, high, size, aug_state *state, lock):
    """
    _rand_uint16(self, low, high, size, rngstate)

    Return random np.uint16 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint16 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.uint16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint16_t off, rng, buf
    cdef uint16_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint16_t>(high - low)
    off = <uint16_t>(<uint16_t>low)
    if size is None:
        with lock:
            random_bounded_uint16_fill(state, off, rng, 1, &buf)
        return np.uint16(<uint16_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.uint16)
        cnt = np.PyArray_SIZE(array)
        out = <uint16_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint16_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_uint8(low, high, size, aug_state *state, lock):
    """
    _rand_uint8(self, low, high, size, rngstate)

    Return random np.uint8 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint8 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.uint8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef uint8_t off, rng, buf
    cdef uint8_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <uint8_t>(high - low)
    off = <uint8_t>(<uint8_t>low)
    if size is None:
        with lock:
            random_bounded_uint8_fill(state, off, rng, 1, &buf)
        return np.uint8(<uint8_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.uint8)
        cnt = np.PyArray_SIZE(array)
        out = <uint8_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_uint8_fill(state, off, rng, cnt, out)
        return array



cdef object _rand_bool(low, high, size, aug_state *state, lock):
    """
    _rand_bool(self, low, high, size, rngstate)

    Return random np.bool integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.bool type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python scalar or ndarray of np.bool
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef bool_t off, rng, buf
    cdef bool_t *out
    cdef np.ndarray array
    cdef np.npy_intp cnt

    rng = <bool_t>(high - low)
    off = <bool_t>(<bool_t>low)
    if size is None:
        with lock:
            random_bounded_bool_fill(state, off, rng, 1, &buf)
        return np.bool_(<bool_t>buf)
    else:
        array = <np.ndarray>np.empty(size, np.bool)
        cnt = np.PyArray_SIZE(array)
        out = <bool_t *>np.PyArray_DATA(array)
        with lock, nogil:
            random_bounded_bool_fill(state, off, rng, cnt, out)
        return array
