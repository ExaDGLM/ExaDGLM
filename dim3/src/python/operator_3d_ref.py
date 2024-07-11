import numpy as np
from mpmath import mp, mpf, matrix, sqrt, inverse

from jacobi import DPS, jacobi, grad_jacobi

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


def simplex_3d(i, j, k, abc_mp):
    '''
    Evalute 3D orthonormal polynomial of the modal basis (i,j,k)
    on simplex at (a,b,c) of order N (i+j+k <= N)
        
    return size : (N+1)*(N+2)*(N+3)/6
    '''
    a = abc_mp[:,0]
    b = abc_mp[:,1]
    c = abc_mp[:,2]
    
    Np = len(abc_mp)
    p = matrix(Np, 1)
    
    h1 = jacobi(i,         0, 0, a)
    h2 = jacobi(j,     2*i+1, 0, b)
    h3 = jacobi(k, 2*(i+j)+2, 0, c)
    
    for k in range(Np):
        p[k] = 2*sqrt(2)*h1[k]*h2[k]*((1-b[k])**i)*h3[k]*((1-c[k])**(i+j))
        
    return p
    
    
def deriv_simplex_3d(i, j, k, abc_mp):
    '''
    derivatives of the modal basis (i,j,k)
    on the 3D simplex at (a,b,c)
    '''
    a = abc_mp[:,0]
    b = abc_mp[:,1]
    c = abc_mp[:,2]
    
    Np = len(abc_mp)
    dmdr = matrix(Np, 1) 
    dmds = matrix(Np, 1)
    dmdt = matrix(Np, 1)
    
    fa  =      jacobi(i,         0, 0, a)
    dfa = grad_jacobi(i,         0, 0, a)
    gb  =      jacobi(j,     2*i+1, 0, b)
    dgb = grad_jacobi(j,     2*i+1, 0, b)
    hc  =      jacobi(k, 2*(i+j)+2, 0, c)
    dhc = grad_jacobi(k, 2*(i+j)+2, 0, c)
        
    for k in range(Np):
        #
        # r-derivative
        #
        dmdr[k] = dfa[k]*gb[k]*hc[k]
        
        if i > 0:
            dmdr[k] *= (0.5*(1-b[k]))**(i-1)        

        if i + j > 0:        
            dmdr[k] *= (0.5*(1-c[k]))**(i+j-1)
    
        #
        # s-derivative
        #    
        dmds[k] = 0.5*(1+a[k])*dmdr[k]

        tmp = dgb[k]*(0.5*(1-b[k]))**i
        if i > 0:
            tmp = tmp + (-0.5*i)*(gb[k]*(0.5*(1-b[k]))**(i-1))    

        if i + j > 0:
            tmp = tmp*(0.5*(1-c[k]))**(i+j-1)

        tmp = fa[k]*tmp*hc[k]
        dmds[k] += tmp
    
        #
        # t-derivative
        #
        dmdt[k] = 0.5*(1+a[k])*dmdr[k] + 0.5*(1+b[k])*tmp

        tmp = dhc[k]*(0.5*(1-c[k]))**(i+j)
        if i + j > 0:
            tmp -= 0.5*(i+j)*hc[k]*(0.5*(1-c[k]))**(i+j-1)

        tmp = fa[k]*gb[k]*tmp*(0.5*(1-b[k]))**i
        dmdt[k] += tmp

        #
        # normalize
        #
        dmdr[k] = 2**(2*i + j + 1.5)*dmdr[k]
        dmds[k] = 2**(2*i + j + 1.5)*dmds[k]
        dmdt[k] = 2**(2*i + j + 1.5)*dmdt[k]
    
    return dmdr, dmds, dmdt


@print_elapsed_time
def make_vandermonde(N, Np, abc_mp):
    '''
    Make a 3D Vandermonde Matrix
    V_{ij} = phi_j(r_i, s_i, t_i)

    a, b, c: coordinates for the Legendre polynomial
    '''
    V_mp = read_mp_matrix(N, Np, Np, 'V', data_dir)

    if not isinstance(V_mp, matrix):            
        V_mp = matrix(Np, Np)

        sk = 0
        for i in range(N+1):
            for j in range(N-i+1):
                for k in range(N-i-j+1):
                    V_mp[:,sk] = simplex_3d(i, j, k, abc_mp)
                    sk += 1

        write_mp_matrix(N, 'V', V_mp, data_dir)

    #
    # inverse of V
    #
    invV_mp = read_mp_matrix(N, Np, Np, 'invV', data_dir)

    if not isinstance(invV_mp, matrix):
        invV_mp = inverse(V_mp)
        write_mp_matrix(N, 'invV', invV_mp, data_dir)

    return V_mp, invV_mp


@print_elapsed_time
def make_mass_matrix(N, Np, V_mp, invV_mp):
    M_mp    = read_mp_matrix(N, Np, Np, 'M', data_dir)
    invM_mp = read_mp_matrix(N, Np, Np, 'invM', data_dir)

    if not isinstance(M_mp, matrix):        
        M_mp    = invV_mp.transpose()*invV_mp
        write_mp_matrix(N, 'M', M_mp, data_dir)

    if not isinstance(invM_mp, matrix):
        invM_mp = V_mp*V_mp.transpose()
        write_mp_matrix(N, 'invM', invM_mp, data_dir)

    M = convert_mp_matrix_to_numpy(M_mp)
    write_bin_matrix(N, 'M', M, data_dir)

    return M_mp, invM_mp, M    


@print_elapsed_time
def make_deriv_vandermonde(N, Np, abc_mp):
    '''
    initialize the gradient of the modal basis (i,j,k)
    at (r,s,t) at order N

    a, b, c: coordinates for the Legendre polynomial
    '''
    Vr_mp = read_mp_matrix(N, Np, Np, 'Vr', data_dir)
    Vs_mp = read_mp_matrix(N, Np, Np, 'Vs', data_dir)
    Vt_mp = read_mp_matrix(N, Np, Np, 'Vt', data_dir)

    if not (isinstance(Vr_mp, matrix) and isinstance(Vs_mp, matrix) and isinstance(Vt_mp, matrix)):
        Vr_mp = matrix(Np, Np)
        Vs_mp = matrix(Np, Np)
        Vt_mp = matrix(Np, Np)

        sk = 0
        for i in range(N+1):
            for j in range(N-i+1):
                for k in range(N-i-j+1):
                    Vr_mp[:,sk], Vs_mp[:,sk], Vt_mp[:,sk] = deriv_simplex_3d(i, j, k, abc_mp)
                    sk += 1

        write_mp_matrix(N, 'Vr', Vr_mp, data_dir)
        write_mp_matrix(N, 'Vs', Vs_mp, data_dir)
        write_mp_matrix(N, 'Vt', Vt_mp, data_dir)

    return Vr_mp, Vs_mp, Vt_mp


@print_elapsed_time
def make_deriv_matrix(N, Np, invV_mp, Vr_mp, Vs_mp, Vt_mp):
    '''
    MATLAB code
    matrix right division operator /
    solve Dr V = Vr
    Dr = Vr/V
    '''
    Dr_mp = read_mp_matrix(N, Np, Np, 'Dr', data_dir)
    Ds_mp = read_mp_matrix(N, Np, Np, 'Ds', data_dir)
    Dt_mp = read_mp_matrix(N, Np, Np, 'Dt', data_dir)

    if not isinstance(Dr_mp, matrix):
        Dr_mp = Vr_mp*invV_mp
        write_mp_matrix(N, 'Dr', Dr_mp, data_dir)

    if not isinstance(Ds_mp, matrix):
        Ds_mp = Vs_mp*invV_mp
        write_mp_matrix(N, 'Ds', Ds_mp, data_dir)

    if not isinstance(Dt_mp, matrix):
        Dt_mp = Vt_mp*invV_mp
        write_mp_matrix(N, 'Dt', Dt_mp, data_dir)

    Dr = convert_mp_matrix_to_numpy(Dr_mp)
    Ds = convert_mp_matrix_to_numpy(Ds_mp)
    Dt = convert_mp_matrix_to_numpy(Dt_mp)
    
    write_bin_matrix(N, 'Dr', Dr, data_dir)
    write_bin_matrix(N, 'Ds', Ds, data_dir)
    write_bin_matrix(N, 'Dt', Dt, data_dir)

    return Dr_mp, Ds_mp, Dt_mp, Dr, Ds, Dt


@print_elapsed_time
def make_weak_matrix(N, Np, M_mp, invM_mp, Dr_mp, Ds_mp, Dt_mp):
    '''
    derivative matrices with weak formulation
    '''
    WDr_mp = read_mp_matrix(N, Np, Np, 'WDr', data_dir)
    WDs_mp = read_mp_matrix(N, Np, Np, 'WDs', data_dir)
    WDt_mp = read_mp_matrix(N, Np, Np, 'WDt', data_dir)

    if not isinstance(WDr_mp, matrix):
        WDr_mp = invM_mp*Dr_mp.transpose()*M_mp
        write_mp_matrix(N, 'WDr', WDr_mp, data_dir)

    if not isinstance(WDs_mp, matrix):
        WDs_mp = invM_mp*Ds_mp.transpose()*M_mp
        write_mp_matrix(N, 'WDs', WDs_mp, data_dir)

    if not isinstance(WDt_mp, matrix):
        WDt_mp = invM_mp*Dt_mp.transpose()*M_mp
        write_mp_matrix(N, 'WDt', WDt_mp, data_dir)

    WDr = convert_mp_matrix_to_numpy(WDr_mp)
    WDs = convert_mp_matrix_to_numpy(WDs_mp)
    WDt = convert_mp_matrix_to_numpy(WDt_mp)
    
    write_bin_matrix(N, 'WDr', WDr, data_dir)
    write_bin_matrix(N, 'WDs', WDs, data_dir)
    write_bin_matrix(N, 'WDt', WDt, data_dir)

    return WDr_mp, WDs_mp, WDt_mp, WDr, WDs, WDt


@print_elapsed_time
def make_interp_matrix(N, Np, high_N, high_Np, high_abc_mp, invV_mp):
    '''
    interpolation matrix
    '''
    #
    # Vandermonde matrixfor interpolation
    #
    interpV_mp = read_mp_matrix(N, high_Np, Np, f"interpV{high_N-N}", data_dir)

    if not isinstance(interpV_mp, matrix):            
        interpV_mp = matrix(high_Np, Np)

        sk = 0
        for i in range(N+1):
            for j in range(N-i+1):
                for k in range(N-i-j+1):
                    interpV_mp[:,sk] = simplex_3d(i, j, k, high_abc_mp)
                    sk += 1

        write_mp_matrix(N, f"interpV{high_N-N}", interpV_mp, data_dir)

    #
    # interpolation matrix
    #
    interpM_mp = read_mp_matrix(N, high_Np, Np, f"interpM{high_N-N}", data_dir)
    if not isinstance(interpM_mp, matrix):
        interpM_mp = interpV_mp*invV_mp
        write_mp_matrix(N, f"interpM{high_N-N}", interpM_mp, data_dir)

    interpM = convert_mp_matrix_to_numpy(interpM_mp)
    write_bin_matrix(N, f"interpM{high_N-N}", interpM, data_dir)
    
    return interpM_mp, interpM
    
    

class Operator3DRef:
    def __init__(self, tet):
        '''
        [input]
        tet: TetraRef object
        
        [output]        
        V          (Np,Np): Vandermonde matrix
        invV       (Np,Np): inverse matrix of V
        Vr, Vs, Vt (Np,Np): Derivative of V
        M          (Np,Np): Mass matrix
        Dr, Ds, Dt (Np,Np): Derivative of M
        '''        
        self.tet = tet
        self.N  = N = tet.N
        self.Np = Np = tet.Np
        self.high_N  = high_N = tet.high_N
        self.high_Np = high_Np = tet.high_Np
        
        if tet.use_mp:
            print("\nOperator3DRef")
            self.V_mp, self.invV_mp = make_vandermonde(N, Np, tet.abc_mp)
            self.M_mp, self.invM_mp, self.M = make_mass_matrix(N, Np, self.V_mp, self.invV_mp)
            self.Vr_mp, self.Vs_mp, self.Vt_mp = make_deriv_vandermonde(N, Np, tet.abc_mp)
            self.Dr_mp, self.Ds_mp, self.Dt_mp, self.Dr, self.Ds, self.Dt = \
                    make_deriv_matrix(N, Np, self.invV_mp, self.Vr_mp, self.Vs_mp, self.Vt_mp)
            self.WDr_mp, self.WDs_mp, self.WDt_mp, self.WDr, self.WDs, self.WDt = \
                    make_weak_matrix(N, Np, self.M_mp, self.invM_mp, self.Dr_mp, self.Ds_mp, self.Dt_mp)

            #
            # high-order operators for L2 error calculations
            #
            self.high_V_mp, self.high_invV_mp = make_vandermonde(high_N, high_Np, tet.high_abc_mp)
            self.high_M_mp, self.high_invM_mp, self.high_M = \
                    make_mass_matrix(high_N, high_Np, self.high_V_mp, self.high_invV_mp)
            self.interpM_mp, self.interpM = \
                    make_interp_matrix(N, Np, high_N, high_Np, tet.high_abc_mp, self.invV_mp)
            
        else:
            self.M   = read_bin_matrix(N, Np, Np, 'M', data_dir)
            self.Dr  = read_bin_matrix(N, Np, Np, 'Dr', data_dir)
            self.Ds  = read_bin_matrix(N, Np, Np, 'Ds', data_dir)
            self.Dt  = read_bin_matrix(N, Np, Np, 'Dt', data_dir)
            self.WDr = read_bin_matrix(N, Np, Np, 'WDr', data_dir)
            self.WDs = read_bin_matrix(N, Np, Np, 'WDs', data_dir)
            self.WDt = read_bin_matrix(N, Np, Np, 'WDt', data_dir)
            self.high_M  = read_bin_matrix(high_N, high_Np, high_Np, 'M', data_dir)
            self.interpM = read_bin_matrix(N, high_Np, Np, f"interpM{high_N-N}", data_dir)
