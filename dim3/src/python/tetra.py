import numpy as np
from mpmath import mp, mpf, matrix, pi, sin, cos, sqrt, fabs, norm, lu_solve

from jacobi import DPS, jacobi_gll

from os.path import abspath, dirname, join
cwd = dirname(abspath(__file__))  # current working directory, DGLM/x.x.x/dim3/src/python
dglm_dir = dirname(dirname(dirname(cwd)))
data_dir = dglm_dir + "/dim3/data"

import sys
sys.path.append(dglm_dir + "/utils/src/python")
from io_mp_mats import write_mp_matrix, read_mp_matrix
from io_mp_mats import write_bin_matrix, read_bin_matrix
from io_mp_mats import convert_mp_matrix_to_numpy, print_elapsed_time

mp.dps = DPS
TOL = 1e-15


def eval_warp(N, xnodes, xout):
    Np = len(xout)
    warp = matrix(Np,1)
    
    d = matrix(Np,1)
    xeq = matrix([-1 + 2*(N+1-i)/N for i in range(1, N+2)])

    for i in range(1, N+2):
        d[:] = xnodes[i-1] - xeq[i-1]
        
        for j in range(2, N+1):
            if i != j:
                for k in range(Np):
                    d[k] = d[k]*(xout[k] - xeq[j-1])/(xeq[i-1] - xeq[j-1])

        if i != 1:
            d[:] = -d[:]/(xeq[i-1] - xeq[0])

        if i != (N+1):
            d[:] = d[:]/(xeq[i-1] - xeq[N])

        warp[:] += d[:]

    return warp


def warp_shift(N, pval, L1, L2, L3):
    '''
    Compute two-dimensional Warp & Blend transform
    '''
    Np = len(L1)
    
    # 1) compute Gauss-Lobatto-Legendre node distribution
    gaussX = -jacobi_gll(N, 0, 0)    
    
    # 2) compute blending function at each node for each edge
    blend1 = matrix(Np,1)
    blend2 = matrix(Np,1)
    blend3 = matrix(Np,1)
    
    for k in range(Np):
        blend1[k] = L2[k]*L3[k]
        blend2[k] = L1[k]*L3[k]
        blend3[k] = L1[k]*L2[k]
    
    # 3) amount of warp for each node, for each edge
    warpfactor1 = matrix(Np,1)
    warpfactor2 = matrix(Np,1)
    warpfactor3 = matrix(Np,1)
    
    warpfactor1[:] = 4*eval_warp(N, gaussX, L3 - L2)
    warpfactor2[:] = 4*eval_warp(N, gaussX, L1 - L3)
    warpfactor3[:] = 4*eval_warp(N, gaussX, L2 - L1)
    
    # 4) combine blend & warp
    warp1 = matrix(Np,1)
    warp2 = matrix(Np,1)
    warp3 = matrix(Np,1)
    
    for k in range(Np):
        warp1[k] = blend1[k]*warpfactor1[k]*(1 + (pval*L1[k])**2)
        warp2[k] = blend2[k]*warpfactor2[k]*(1 + (pval*L2[k])**2)
        warp3[k] = blend3[k]*warpfactor3[k]*(1 + (pval*L3[k])**2)
    
    # 5) evaluate shift in equilateral triangle
    dx = matrix(Np,1)
    dy = matrix(Np,1)
    
    for k in range(Np):
        dx[k] = 1*warp1[k] + cos(2*pi/3)*warp2[k] + cos(4*pi/3)*warp3[k]
        dy[k] = 0*warp1[k] + sin(2*pi/3)*warp2[k] + sin(4*pi/3)*warp3[k]
    
    return dx, dy


def equi_nodes_3d(N):
    Np = (N+1)*(N+2)*(N+3)//6
    r = matrix(Np,1)
    s = matrix(Np,1)
    t = matrix(Np,1)
    
    sk = 0
    for n in range(1, N+2):
        for m in range(1, N+3-n):
            for q in range(1, N+4-n-m):
                r[sk] = -1 + (q-1)*2/N
                s[sk] = -1 + (m-1)*2/N
                t[sk] = -1 + (n-1)*2/N
                sk += 1
                
    return r, s, t


def make_nodes_equilateral(N):
    '''
    Compute Warp & Blend tetrahedron nodes
    Input N: polynomial order of interpolant
    Output X,Y,Z: vectors of node coordinates in equilateral tetrahedron
    '''
    
    # choose optimized blending parameter
    alpha_store = [     0,      0,      0, 0.1002, 1.1332,
                   1.5608, 1.3413, 1.2577, 1.1603, 1.10153,
                   0.6080, 0.4523, 0.8856, 0.8717, 0.9655]
    
    alpha = alpha_store[N-1] if N <= 15 else 1.0
    
    # total number of nodes and tolerance
    Np = (N+1)*(N+2)*(N+3)//6
    
    L1 = matrix(Np,1)
    L2 = matrix(Np,1)
    L3 = matrix(Np,1)
    L4 = matrix(Np,1)
    
    r, s, t = equi_nodes_3d(N)  # create equi-distributed nodes
    L1[:] = (1 + t[:])/2
    L2[:] = (1 + s[:])/2
    L3[:] = -(1 + r[:] + s[:] + t[:])/2
    L4[:] = (1 + r[:])/2
    
    # set vertices of tetrahedron
    v1 = matrix([-1, -1/sqrt(3), -1/sqrt(6)])
    v2 = matrix([ 1, -1/sqrt(3), -1/sqrt(6)])
    v3 = matrix([ 0,  2/sqrt(3), -1/sqrt(6)])
    v4 = matrix([ 0,          0,  3/sqrt(6)])
    
    # orthogonal axis tangents on faces 1-4
    t1 = matrix(4, 3)
    t2 = matrix(4, 3)
    
    for k in range(3):
        t1[0,k] = v2[k] - v1[k]
        t1[1,k] = v2[k] - v1[k]
        t1[2,k] = v3[k] - v2[k]
        t1[3,k] = v3[k] - v1[k]
        t2[0,k] = v3[k] - 0.5*(v1[k] + v2[k])
        t2[1,k] = v4[k] - 0.5*(v1[k] + v2[k])
        t2[2,k] = v4[k] - 0.5*(v2[k] + v3[k])
        t2[3,k] = v4[k] - 0.5*(v1[k] + v3[k])
    
    for n in range(4):        
        norm1 = norm(t1[n,:])
        norm2 = norm(t2[n,:])
        
        for k in range(3):
            t1[n,k] = t1[n,k]/norm1
            t2[n,k] = t2[n,k]/norm2
    
    # Warp and blend for each face (accumulated in shiftXYZ)
    # form undeformed coordinates
    xyz = matrix(Np, 3)
    shift = matrix(Np, 3)
    blend = matrix(Np, 1)
    
    for k in range(Np):
        for j in range(3):
            xyz[k,j] = L3[k]*v1[j] + L4[k]*v2[j] + L2[k]*v3[j] + L1[k]*v4[j]
    
    for face in range(4):
        if face == 0:
            La = L1
            Lb = L2
            Lc = L3
            Ld = L4
        elif face == 1:
            La = L2
            Lb = L1
            Lc = L3
            Ld = L4
        elif face == 2:
            La = L3
            Lb = L1
            Lc = L4
            Ld = L2
        else:
            La = L4
            Lb = L1
            Lc = L3
            Ld = L2
        
        # compute warp tangential to face
        warp1, warp2 = warp_shift(N, alpha, Lb, Lc, Ld)
                
        for k in range(Np):
            # compute volume blending
            blend[k] = Lb[k]*Lc[k]*Ld[k]
        
            # modify linear blend
            denom = (Lb[k] + 0.5*La[k])*(Lc[k] + 0.5*La[k])*(Ld[k] + 0.5*La[k])
            
            if denom > TOL:
                blend[k] = (1 + (alpha*La[k])**2)*blend[k]/denom
        
            for j in range(3):
                if La[k] < TOL and (Lb[k] > TOL) + (Lc[k] > TOL) + (Ld[k] > TOL) < 3:
                    # fix face warp
                    shift[k,j] = warp1[k]*t1[face,j] + warp2[k]*t2[face,j]
                else:
                    # compute warp & blend
                    shift[k,j] += blend[k]*warp1[k]*t1[face,j] + blend[k]*warp2[k]*t2[face,j]
    
    xyz = xyz + shift
    
    return xyz


def xyz2rst(xyz):
    Np = len(xyz)    
    rst = matrix(Np, 3)
    rhs = matrix(3, 1)
    A = matrix(3, 3)
    
    v1 = matrix([-1, -1/sqrt(3), -1/sqrt(6)])
    v2 = matrix([ 1, -1/sqrt(3), -1/sqrt(6)])
    v3 = matrix([ 0,  2/sqrt(3), -1/sqrt(6)])
    v4 = matrix([ 0,          0,  3/sqrt(6)])
    
    for i in range(3):
        A[i,:] = matrix([[0.5*(v2[i]-v1[i]), 0.5*(v3[i]-v1[i]), 0.5*(v4[i]-v1[i])]])
    
    for k in range(Np):
        for j in range(3):
            rhs[j] = xyz[k,j] - 0.5*(v2[j] + v3[j] + v4[j] - v1[j])  # (Np,3) - (3,)
         
        rst[k,:] = lu_solve(A, rhs).transpose()
    
    return rst
    
    
def rst2abc(rst):
    '''
    Transfer from (r,s,t) -> (a,b,c) coordinates in triangle
    (r, s, t) : coordinates of standard tetrahedron
    (a, b, c) : coordinates of Legendre polynomial
    '''
    Np = len(rst)
    abc = matrix(Np, 3)
    
    for k in range(Np):
        if fabs(rst[k,1] + rst[k,2]) > TOL:
            abc[k,0] = -2*(1 + rst[k,0])/(rst[k,1] + rst[k,2]) - 1
        else:
            abc[k,0] = -1

        if fabs(rst[k,2] - 1) > TOL:
            abc[k,1] = 2*(1 + rst[k,1])/(1 - rst[k,2]) - 1
        else:
            abc[k,1] = -1

        abc[k,2] = rst[k,2]
    
    return abc


@print_elapsed_time
def make_tetra_coords(N):
    Np = (N+1)*(N+2)*(N+3)//6
    
    xyz_mp = read_mp_matrix(N, Np, 3, 'xyz', data_dir)
    if not isinstance(xyz_mp, matrix):
        xyz_mp = make_nodes_equilateral(N)
        write_mp_matrix(N, 'xyz', xyz_mp, data_dir)

    rst_mp = read_mp_matrix(N, Np, 3, 'rst', data_dir)
    if not isinstance(rst_mp, matrix):
        rst_mp = xyz2rst(xyz_mp)
        write_mp_matrix(N, 'rst', rst_mp, data_dir)

    abc_mp = read_mp_matrix(N, Np, 3, 'abc', data_dir)  # for Vandermonde matrix
    if not isinstance(abc_mp, matrix):
        abc_mp = rst2abc(rst_mp)
        write_mp_matrix(N, 'abc', abc_mp, data_dir)
    
    xyz = convert_mp_matrix_to_numpy(xyz_mp)
    rst = convert_mp_matrix_to_numpy(rst_mp)
    
    write_bin_matrix(N, 'xyz', xyz, data_dir)
    write_bin_matrix(N, 'rst', rst, data_dir)
        
    return xyz_mp, rst_mp, abc_mp, xyz, rst



class TetraRef:
    def __init__(self, N, interpN=2, use_mp=False):
        self.N     = N                     # polynomial order        
        self.Np    = (N+1)*(N+2)*(N+3)//6  # number of points in an element
        self.Nface = 4                     # number of faces in an element
        self.Nfp   = (N+1)*(N+2)//2        # number of points on a face
        self.Nrot  = 3                     # number of rotation in a face        
        self.interpN = interpN
        self.high_N  = high_N = N + interpN  # for L2 error
        self.high_Np = (high_N+1)*(high_N+2)*(high_N+3)//6        
        self.use_mp  = use_mp
        
        #
        # reference tetrahedron
        #
        if use_mp:
            print("\nTetraRef")            
            self.xyz_mp, self.rst_mp, self.abc_mp, self.xyz, self.rst = \
                make_tetra_coords(N)
            self.high_xyz_mp, self.high_rst_mp, self.high_abc_mp, self.high_xyz, self.high_rst = \
                make_tetra_coords(high_N)
            
        else:
            self.xyz = read_bin_matrix(N, self.Np, 3, 'xyz', data_dir)
            self.rst = read_bin_matrix(N, self.Np, 3, 'rst', data_dir)
            self.high_xyz = read_bin_matrix(high_N, self.high_Np, 3, 'xyz', data_dir)
            self.high_rst = read_bin_matrix(high_N, self.high_Np, 3, 'rst', data_dir)
