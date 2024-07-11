import pytest
import numpy as np
from mpmath import matrix
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from tetra import TetraRef
from operator_3d_ref import Operator3DRef


@pytest.fixture(scope="module")
def OP3DN8():
    N = 8
    tet = TetraRef(N, interpN=2, use_mp=True)
    OP3DRef = Operator3DRef(tet)
    
    Np = tet.Np
    Dr = np.zeros((Np, Np), 'f8')
    Ds = np.zeros((Np, Np), 'f8')
    Dt = np.zeros((Np, Np), 'f8')
    for i in range(Np):
        for j in range(Np):        
            Dr[i,j]  = OP3DRef.Dr_mp[i,j]
            Ds[i,j]  = OP3DRef.Ds_mp[i,j]
            Dt[i,j]  = OP3DRef.Dt_mp[i,j]
    
    return tet, OP3DRef, Dr, Ds, Dt


@pytest.fixture(scope="module")
def OP3DN12():
    N = 12
    tet = TetraRef(N, interpN=2, use_mp=True)
    OP3DRef = Operator3DRef(tet)
    
    return tet, OP3DRef


def test_make_mass_matrix(OP3DN8):
    tet, OP, Dr, Ds, Dt = OP3DN8
    #print(f"M 3D {OP.M.shape}\n", OP.M)

    # mass matrix 적분
    # standard tetrahedron 부피이므로 값은 (1/6)*(2*2*2)
    a = np.sum(OP.M)    
    assert_aae(a, 8/6, 15)

    # mass matrix 함수 적분
    f = lambda x, y, z: np.cos(x)*np.sin(y)*np.sin(0.5*z)
    a = np.sum(OP.M@f(tet.rst[:,0], tet.rst[:,1], tet.rst[:,2]))

    from scipy import integrate
    b = integrate.tplquad(f, -1, 1, -1, lambda x:-x, -1, lambda x,y:-x-y-1)[0]    
    assert_aae(a, b, 8)


def test_make_deriv_matrix(OP3DN8):
    tet, OP, Dr, Ds, Dt = OP3DN8
    
    # row-sum is zero
    assert_aae(np.sum(Dr, axis=1), 0, 10)
    assert_aae(np.sum(Ds, axis=1), 0, 10)
    assert_aae(np.sum(Dt, axis=1), 0, 10)
    
    # 함수 미분
    r, s, t = tet.rst[:,0], tet.rst[:,1], tet.rst[:,2]
    f    =  np.cos(r)*np.sin(s)*np.sin(0.5*t)
    dfdr = -np.sin(r)*np.sin(s)*np.sin(0.5*t)
    dfds =  np.cos(r)*np.cos(s)*np.sin(0.5*t)
    dfdt =  np.cos(r)*np.sin(s)*np.cos(0.5*t)*0.5
    assert_aae(Dr@f, dfdr, 3)
    assert_aae(Ds@f, dfds, 3)
    assert_aae(Dt@f, dfdt, 3)
    
    
def test_make_deriv_matrix(OP3DN8):
    tet, OP, Dr, Ds, Dt = OP3DN8
            
    # 다항함수 미분
    f  = lambda x: 1 + x +   x**2 +   x**3 +   x**4 +   x**5 +   x**6 +   x**7 +   x**8
    df = lambda x:     1 + 2*x    + 3*x**2 + 4*x**3 + 5*x**4 + 6*x**5 + 7*x**6 + 8*x**7
    
    r = tet.rst[:,0]
    s = tet.rst[:,1]
    t = tet.rst[:,2]
    assert_aae(Dr@f(r), df(r), 13)
    assert_aae(Ds@f(s), df(s), 13)
    assert_aae(Dt@f(t), df(t), 13)
    

def test_make_deriv_matrix_N12(OP3DN12):
    tet, OP = OP3DN12
    
    Np = tet.Np
    Dr = np.zeros((Np, Np), 'f8')
    Ds = np.zeros((Np, Np), 'f8')
    Dt = np.zeros((Np, Np), 'f8')
    for i in range(Np):
        for j in range(Np):        
            Dr[i,j]  = OP.Dr_mp[i,j]
            Ds[i,j]  = OP.Ds_mp[i,j]
            Dt[i,j]  = OP.Dt_mp[i,j]
        
    # 다항함수 미분
    f  = lambda x: 1 + x +   x**2 +   x**3 +   x**4 +   x**5 +   x**6 +   x**7 +   x**8 +   x**9 +    x**10 +    x**11 +    x**12
    df = lambda x:     1 + 2*x    + 3*x**2 + 4*x**3 + 5*x**4 + 6*x**5 + 7*x**6 + 8*x**7 + 9*x**8 + 10*x**9  + 11*x**10 + 12*x**11
    
    r = tet.rst[:,0]
    s = tet.rst[:,1]
    t = tet.rst[:,2]
    assert_aae(Dr@f(r), df(r), 11)
    assert_aae(Ds@f(s), df(s), 11)
    assert_aae(Dt@f(t), df(t), 11)



def test_make_deriv_matrix_N12_mp(OP3DN12):
    tet, OP = OP3DN12
    
    # 다항함수 미분
    def f_df(xx):
        f_mp  = matrix(tet.Np, 1)
        df_mp = matrix(tet.Np, 1)
    
        for i, x in enumerate(xx):
            f_mp[i]  = 1 + x +   x**2 +   x**3 +   x**4 +   x**5 +   x**6 +   x**7 +   x**8 +   x**9 +    x**10 +    x**11 +    x**12
            df_mp[i] =     1 + 2*x    + 3*x**2 + 4*x**3 + 5*x**4 + 6*x**5 + 7*x**6 + 8*x**7 + 9*x**8 + 10*x**9  + 11*x**10 + 12*x**11
            
        return f_mp, df_mp
    
    r_mp = tet.rst_mp[:,0]
    s_mp = tet.rst_mp[:,1]
    t_mp = tet.rst_mp[:,2]
    
    fr_mp, dfr_mp = f_df(r_mp)
    fs_mp, dfs_mp = f_df(s_mp)
    ft_mp, dft_mp = f_df(t_mp)
    
    assert_aae(OP.Dr_mp*fr_mp, dfr_mp, 13)
    assert_aae(OP.Ds_mp*fs_mp, dfs_mp, 13)
    assert_aae(OP.Dt_mp*ft_mp, dft_mp, 13)
