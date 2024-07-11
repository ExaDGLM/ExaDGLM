import numpy as np
from mpmath import mp, mpf, matrix
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from io_mp_mats import *


def test_truncate_to_zero():
    eps = np.finfo('f8').eps
    arr = np.array([10, eps*11, eps*9])
    arr2 = truncate_to_zero(arr)
    assert_ae(arr, [10, eps*11, 0])
    
    
def test_write_mp_matrix():
    N = 1
    mat_name = "mat_mp"
    
    mat_mp = matrix(2, 3)
    mat_mp[0,0] = mpf("1.234567890123456789012")
    mat_mp[0,1] = mpf("2.345678901234567890123")
    mat_mp[0,2] = mpf("3.456789012345678901234")
    mat_mp[1,0] = mpf("4.567890123456789012345")
    mat_mp[1,1] = mpf("5.678901234567890123456")
    mat_mp[1,2] = mpf("6.789012345678901234567")
    
    write_mp_matrix(N, mat_name, mat_mp, path.cwd)
    expect = '''1.23456789012345678901
2.34567890123456789012
3.45678901234567890123
4.56789012345678901234
5.67890123456789012346
6.78901234567890123457
'''
    with open(f"{path.cwd}/mp_mats/N{N}/{mat_name}.txt", "r") as f:
        assert f.read() == expect
    
    
def test_read_mp_matrix():
    N, nrow, ncol = 1, 2, 3
    mat_mp = read_mp_matrix(N, nrow, ncol, "mat_mp", path.cwd)
    
    assert mat_mp[0,0] == mpf("1.23456789012345678901")
    assert mat_mp[0,1] == mpf("2.34567890123456789012")
    assert mat_mp[0,2] == mpf("3.45678901234567890123")
    assert mat_mp[1,0] == mpf("4.56789012345678901234")
    assert mat_mp[1,1] == mpf("5.67890123456789012346")
    assert mat_mp[1,2] == mpf("6.78901234567890123457")
    
    
def test_convert_mp_matrix_to_numpy():
    N, nrow, ncol = 1, 2, 3
    mat_mp = read_mp_matrix(N, nrow, ncol, "mat_mp", path.cwd)
    mat = convert_mp_matrix_to_numpy(mat_mp, truncate=True)
    expect = np.array([
        1.2345678901234568,
        2.3456789012345679,
        3.4567890123456790, 
        4.5678901234567890,
        5.6789012345678901, 
        6.7890123456789012], 'f8').reshape(2,3)
    assert_ae(mat, expect)
    
    
def test_write_read_bin_matrix():
    N = 1
    mat_name = "mat_np"
    
    mat_np = np.array([
        1.2345678901234568,
        2.3456789012345679,
        3.4567890123456790, 
        4.5678901234567890,
        5.6789012345678901, 
        6.7890123456789012], 'f8').reshape(2,3)
    
    write_bin_matrix(N, mat_name, mat_np, path.cwd)
    
    mat2_np = read_bin_matrix(N, 2, 3, mat_name, path.cwd)
    assert_ae(mat_np, mat2_np)
