import numpy as np
from mpmath import mp, mpf, matrix, sqrt, inverse
from copy import deepcopy

from jacobi import DPS, jacobi

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
NODETOL = 1e-12


class FaceMask:
    def __init__(self, tet):
        self.tet = tet
        self.N   = tet.N
        
        
    def arr_to_nested_list(self, src_arr):
        nested_list = []

        k = 0
        for n in range(self.N+1, 0, -1):
            sub_list = []
            for _ in range(n):
                sub_list.append(src_arr[k])
                k += 1
            nested_list.append(sub_list)

        return nested_list


    def nested_list_to_arr(self, nested_list):
        ret_list = []    
        for sub_list in nested_list:
            ret_list.extend(sub_list)

        return ret_list


    def rotate(self, nested_list):
        nested_list = deepcopy(nested_list)
        ret_list = []

        while len(nested_list[0]) > 0:
            ret_sub_list = []

            for sub_list in nested_list[::-1]:
                if len(sub_list) == 0: continue
                ret_sub_list.append( sub_list.pop(0) )

            ret_list.append(ret_sub_list)

        return ret_list


    def invert(self, nested_list):
        ret_list = []    
        for sub_list in nested_list:
            ret_list.append(sub_list[::-1])

        return ret_list
    
    
    def make_face_mask(self):
        '''
        [output]
        Fmask (Nface,Nrot,Nfp)
        '''
        Nface, Nrot, Nfp = self.tet.Nface, self.tet.Nrot, self.tet.Nfp
        
        Fmask  = np.zeros((Nface, Nfp), 'i4')
        FmaskP = np.zeros((Nface, Nrot, Nfp), 'i4')
        
        # vertices: v0, v1, v2, v3
        # face0: (v1, v0, v2)
        # face1: (v0, v1, v3)
        # face2: (v1, v2, v3)
        # face3: (v2, v0, v3)

        r = self.tet.rst[:,0]
        s = self.tet.rst[:,1]
        t = self.tet.rst[:,2]        
        face0 = np.where(np.abs(t+1) < NODETOL)[0]
        face1 = np.where(np.abs(s+1) < NODETOL)[0]
        face2 = np.where(np.abs(r+s+t+1) < NODETOL)[0]
        face3 = np.where(np.abs(r+1) < NODETOL)[0]
            
        for fi, face in enumerate([face0, face1, face2, face3]):
            f_list = self.arr_to_nested_list(face)
            if fi in [0,3]:
                f_list = self.invert(f_list)
            
            r1_list = self.rotate(f_list)    
            r2_list = self.rotate(r1_list)
            
            inv_r0_list = self.invert(f_list)
            inv_r1_list = self.invert(r1_list)
            inv_r2_list = self.invert(r2_list)

            r0 = self.nested_list_to_arr(f_list)            
            inv_r0 = self.nested_list_to_arr(inv_r0_list)
            inv_r1 = self.nested_list_to_arr(inv_r1_list)
            inv_r2 = self.nested_list_to_arr(inv_r2_list)
                        
            # for vmapM
            Fmask[fi,:] = r0[:]
            
            # for vmapP
            FmaskP[fi,0,:] = inv_r0[:]
            FmaskP[fi,1,:] = inv_r1[:]
            FmaskP[fi,2,:] = inv_r2[:]
            
        return Fmask, FmaskP
    

def simplex_2d(i, j, a_mp, b_mp):
    '''
    Evalute 2D orthonormal polynomial of the modal basis (i,j)
    on simplex at (a,b) of order N (i+j <= N)

    return size : (N+1)*(N+2)/2
    '''

    Nfp = len(a_mp)
    p = matrix(Nfp, 1)

    h1 = jacobi(i,     0, 0, a_mp)
    h2 = jacobi(j, 2*i+1, 0, b_mp)

    for k in range(Nfp):
        p[k] = sqrt(2)*h1[k]*h2[k]*(1-b_mp[k])**i

    return p


def make_vandermonde_2d(N, a_mp, b_mp):
    '''
    Make a 2D Vandermonde Matrix
    V_{ij} = phi_j(r_i, s_i)

    a, b: coordinates for the Legendre polynomial
    '''
    Nfp = (N+1)*(N+2)//2        
    V2D_mp = matrix(Nfp, Nfp)

    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2D_mp[:,sk] = simplex_2d(i, j, a_mp, b_mp)
            sk += 1

    invV2D_mp = inverse(V2D_mp)

    return V2D_mp, invV2D_mp


@print_elapsed_time
def make_mass_surface(tet, op_ref, Fmask):
    '''        
    Compute surface to volume lift operator for DG formulation

    shape : (Np, Nface*Nfp)
    '''
    N, Np = tet.N, tet.Np
    Nface, Nfp = tet.Nface, tet.Nfp

    Emat_mp = read_mp_matrix(N, Np, Nface*Nfp, "Emat", data_dir)
    LIFT_mp = read_mp_matrix(N, Np, Nface*Nfp, "LIFT", data_dir)

    if not (isinstance(Emat_mp, matrix) and isinstance(LIFT_mp, matrix)):        
        V3D_mp = op_ref.V_mp
        abc_mp = tet.abc_mp  

        Emat_mp = matrix(Np, Nface*Nfp)
        E_mp = matrix(Nfp, Nfp)
        a_mp = matrix(Nfp, 1)
        b_mp = matrix(Nfp, 1)

        #
        # face 0
        #
        print()
        for i in range(Nfp):
            fidx = Fmask[0,i]
            a_mp[i] = abc_mp[fidx,0]  # a
            b_mp[i] = abc_mp[fidx,1]  # b                

        V2D_mp, invV2D_mp = make_vandermonde_2d(N, a_mp, b_mp)
        E_mp[:,:] = invV2D_mp.transpose()*invV2D_mp  # Mass matrix in the face

        for i in range(Nfp):
            fidx = Fmask[0,i]                
            for j in range(Nfp):
                Emat_mp[fidx,j] = E_mp[i,j]

        #
        # face 1
        #
        for i in range(Nfp):
            fidx = Fmask[1,i]
            a_mp[i] = abc_mp[fidx,0]  # a
            b_mp[i] = abc_mp[fidx,2]  # c

        V2D_mp, invV2D_mp = make_vandermonde_2d(N, a_mp, b_mp)
        E_mp[:,:] = invV2D_mp.transpose()*invV2D_mp

        for i in range(Nfp):
            fidx = Fmask[1,i]                
            for j in range(Nfp):
                Emat_mp[fidx,j+Nfp] = E_mp[i,j]

        #
        # face 2
        #
        for i in range(Nfp):
            fidx = Fmask[2,i]
            a_mp[i] = abc_mp[fidx,1]  # b
            b_mp[i] = abc_mp[fidx,2]  # c

        V2D_mp, invV2D_mp = make_vandermonde_2d(N, a_mp, b_mp)
        E_mp[:,:] = invV2D_mp.transpose()*invV2D_mp

        for i in range(Nfp):
            fidx = Fmask[2,i]                
            for j in range(Nfp):
                Emat_mp[fidx,j+2*Nfp] = E_mp[i,j]

        #
        # face 3
        #
        for i in range(Nfp):
            fidx = Fmask[3,i]
            a_mp[i] = abc_mp[fidx,1]  # b
            b_mp[i] = abc_mp[fidx,2]  # c

        V2D_mp, invV2D_mp = make_vandermonde_2d(N, a_mp, b_mp)
        E_mp[:,:] = invV2D_mp.transpose()*invV2D_mp

        for i in range(Nfp):
            fidx = Fmask[3,i]
            for j in range(Nfp):
                Emat_mp[fidx,j+3*Nfp] = E_mp[i,j]

        #
        # LIFT: inv(mass matrix)*\I_n (L_i,L_j)_{edge_n}
        #
        LIFT_mp = V3D_mp*(V3D_mp.transpose()*Emat_mp)  # shape (Np, Nface*Nfp)

        #
        # write the mpmath matrix to txt file
        #
        write_mp_matrix(N, 'Emat', Emat_mp, data_dir)
        write_mp_matrix(N, 'LIFT', LIFT_mp, data_dir)

    Emat = convert_mp_matrix_to_numpy(Emat_mp)
    LIFT = convert_mp_matrix_to_numpy(LIFT_mp)
    
    write_bin_matrix(N, 'Emat', Emat, data_dir)
    write_bin_matrix(N, 'LIFT', LIFT, data_dir)

    return Emat_mp, LIFT_mp, Emat, LIFT



class Operator3DFace:
    def __init__(self, op_ref):
        '''
        [input]
        op_ref: Operator3DRef object
        
        [output]
        Fmask (Nface, Nfp)       : mask indices of each face
        FmaskP(Nface, Nrot, Nfp) : mask indices of each face for vmapP
        LIFT  (Np,Nface*Nfp)     : inverse of mass matrix for surface integral
        Fscale(Nelem, Nface)     : geometric factors
        vmapM (Nelem, Nface, Nfp): indices of interior node, u^-
        vmapP (Nelem, Nface, Nfp): indices of exterior node, u^+
        '''
        
        self.op_ref = op_ref
        self.tet = tet = op_ref.tet
        
        self.Fmask, self.FmaskP = FaceMask(tet).make_face_mask()
        
        if tet.use_mp:
            self.Emat_mp, self.LIFT_mp, self.Emat, self.LIFT = \
                    make_mass_surface(tet, op_ref, self.Fmask)  # Emat used for unittest
        
        else:
            self.Emat = read_bin_matrix(tet.N, tet.Np, tet.Nface*tet.Nfp, 'Emat', data_dir)
            self.LIFT = read_bin_matrix(tet.N, tet.Np, tet.Nface*tet.Nfp, 'LIFT', data_dir)    
