#!/bin/sh

python setup_test_xtension.py build_ext -if
echo "***********************************************************"
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import cython; print('cython version:', cython.__version__)"
python -c "from xtension import foo; print('xtension module test (31.415926):', foo.multiply_by_pi(10))"
echo "***********************************************************"
