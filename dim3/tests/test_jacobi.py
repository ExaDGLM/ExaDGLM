import numpy as np
from mpmath import mpf
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from jacobi import jacobi_gll, jacobi, grad_jacobi


def test_jacobi_gll():    
    assert_ae(jacobi_gll(1), [-1, 1])
    assert_ae(jacobi_gll(2), [-1, 0, 1])
    assert_aae(jacobi_gll(3), [-1, -0.447213595499958, 0.4472135954999580, 1], 15)
    assert_aae(jacobi_gll(4), [-1, -0.654653670707977, 0, 0.654653670707977, 1], 15)
    assert_aae(jacobi_gll(5), [-1, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1], 15)

    
def test_jacobi_2():
    N = 2
    r = jacobi_gll(N)
    j = jacobi(N, 0, 0, r)
    dP = grad_jacobi(N, 0, 0, r)
    assert_aae(j, [1.581138830084190, -0.790569415042095, 1.581138830084190], 15)
    assert_aae(dP, [-4.743416490252569, 0, 4.743416490252569], 15)

    
def test_jacobi_3():
    N = 3
    r = jacobi_gll(N)
    j = jacobi(N, 0, 0, r)
    dP = grad_jacobi(N, 0, 0, r)
    assert_aae(j, [-1.87082869338697, 0.836660026534076, -0.836660026534076, 1.87082869338697], 15)
    assert_aae(dP, [11.224972160321824, 0, 0, 11.224972160321824], 15)

    
def test_jacobi_4():
    N = 4
    r = jacobi_gll(N)
    j = jacobi(N, 0, 0, r)
    dP = grad_jacobi(N, 0, 0, r)
    assert_aae(j, [2.121320343559642, -0.909137290096990, 0.795495128834866, -0.909137290096990, 2.121320343559642], 15)
    assert_aae(dP, [-21.213203435596430, 0, 0, 0, 21.213203435596430], 14)


def test_jacobi_5():
    N = 5
    r = jacobi_gll(N)
    j = jacobi(N, 0, 0, r)
    dP = grad_jacobi(N, 0, 0, r)
    assert_aae(j, [-2.345207879911714, 0.984276557099483, -0.812914072195837, 0.812914072195837, -0.984276557099483, 2.345207879911714], 14)
    assert_aae(dP, [35.178118198675726, 0, 0, 0, 0, 35.178118198675726], 14)
