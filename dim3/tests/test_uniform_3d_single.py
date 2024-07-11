import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from uniform_3d_single import Uniform3DSingle


x1, x2, nx = 0, 2, 3
y1, y2, ny = 1, 3, 3
z1, z2, nz = 2, 5, 4 
bc = {'x':'pbc', 'y':'pbc', 'z':'pbc'}
mod_suffix = 'py'  # 'cpp', 'py'

msh = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix)
    

def test_make_VXYZ():
    assert_ae(msh.VXYZ[0], [0,1,2])
    assert_ae(msh.VXYZ[1], [0,1,3])
    assert_ae(msh.VXYZ[2], [0,1,4])
    assert_ae(msh.VXYZ[3], [0,1,5])
    assert_ae(msh.VXYZ[4], [0,2,2])
    assert_ae(msh.VXYZ[5], [0,2,3])
    assert_ae(msh.VXYZ[6], [0,2,4])
    assert_ae(msh.VXYZ[7], [0,2,5])
    assert_ae(msh.VXYZ[8], [0,3,2])
    assert_ae(msh.VXYZ[9], [0,3,3])
    assert_ae(msh.VXYZ[22], [1,3,4])
    assert_ae(msh.VXYZ[23], [1,3,5])
    

def test_make_EtoV():
    # hexahedron nodeTags = [v0, v1, v2, v3, v4, v5, v6, v7]
    # elem0 nodeTags = [v0, v7, v3, v1]
    # elem1 nodeTags = [v0, v7, v2, v3]
    # elem2 nodeTags = [v0, v7, v6, v2]
    # elem3 nodeTags = [v0, v7, v4, v6]
    # elem4 nodeTags = [v0, v7, v5, v4]
    # elem5 nodeTags = [v0, v7, v1, v5]
    
    # 6 tetrahedrons in the 1st hexahedron
    assert_ae(msh.EtoV[0,:], [0,17, 5, 1])  # 0 1 4 5 12 13 16 17
    assert_ae(msh.EtoV[1,:], [0,17, 4, 5])
    assert_ae(msh.EtoV[2,:], [0,17,16, 4])
    assert_ae(msh.EtoV[3,:], [0,17,12,16])
    assert_ae(msh.EtoV[4,:], [0,17,13,12])
    assert_ae(msh.EtoV[5,:], [0,17, 1,13])
    
    # 6 tetrahedrons in the 2nd hexahedron
    assert_ae(msh.EtoV[6,:],  [1,18, 6, 2])  # 1 2 5 6 13 14 17 18
    assert_ae(msh.EtoV[7,:],  [1,18, 5, 6])
    assert_ae(msh.EtoV[8,:],  [1,18,17, 5])
    assert_ae(msh.EtoV[9,:],  [1,18,13,17])
    assert_ae(msh.EtoV[10,:], [1,18,14,13])
    assert_ae(msh.EtoV[11,:], [1,18, 2,14])
    
    # 6 tetrahedrons in the 6th hexahedron
    assert_ae(msh.EtoV[30,:], [6,23,11, 7])  # 6 7 10 11 18 19 22 23
    assert_ae(msh.EtoV[31,:], [6,23,10,11])
    assert_ae(msh.EtoV[32,:], [6,23,22,10])
    assert_ae(msh.EtoV[33,:], [6,23,18,22])
    assert_ae(msh.EtoV[34,:], [6,23,19,18])
    assert_ae(msh.EtoV[35,:], [6,23, 7,19])
    
    # 6 tetrahedrons in the 9th hexahedron
    assert_ae(msh.EtoV[48,:], [14,31,19,15])  # 14 15 18 19 26 27 30 31
    assert_ae(msh.EtoV[49,:], [14,31,18,19])
    assert_ae(msh.EtoV[50,:], [14,31,30,18])
    assert_ae(msh.EtoV[51,:], [14,31,26,30])
    assert_ae(msh.EtoV[52,:], [14,31,27,26])
    assert_ae(msh.EtoV[53,:], [14,31,15,27])
    
    # 6 tetrahedrons in the last (12th) hexahedron
    assert_ae(msh.EtoV[66,:], [18,35,23,19])  # 18 19 22 23 30 31 34 35
    assert_ae(msh.EtoV[67,:], [18,35,22,23])
    assert_ae(msh.EtoV[68,:], [18,35,34,22])
    assert_ae(msh.EtoV[69,:], [18,35,30,34])
    assert_ae(msh.EtoV[70,:], [18,35,31,30])
    assert_ae(msh.EtoV[71,:], [18,35,19,31])
    
    
def test_make_EtoEFRB_pbc():
    # check neighbor consistency
    for ei in range(msh.Nelem):
        for fi in range(msh.Nface):            
            nbr_ei = msh.EtoE[ei,fi]
            nbr_fi = msh.EtoF[ei,fi]
            nbr_ri = msh.EtoR[ei,fi]
            
            assert_ae(msh.EtoE[nbr_ei,nbr_fi], ei)
            assert_ae(msh.EtoF[nbr_ei,nbr_fi], fi)
            assert_ae(msh.EtoR[nbr_ei,nbr_fi], nbr_ri)
            
            
def test_make_EtoEFRB_bc1():
    # check neighbor consistency and bc1
    bc={'x':'bc1', 'y':'pbc', 'z':'pbc'}
    msh = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix)
    
    for ei in range(msh.Nelem):
        for fi in range(msh.Nface):            
            nbr_ei = msh.EtoE[ei,fi]
            nbr_fi = msh.EtoF[ei,fi]
            nbr_ri = msh.EtoR[ei,fi]
            #print(f"ei={ei}, fi={fi}, nbr_ei={nbr_ei}, nbr_fi={nbr_fi}, rot={nbr_ri}")
            
            if nbr_ei == -99:                
                face_vts = [(1,0,2), (0,1,3), (1,2,3), (2,0,3)][fi]
                for v in msh.EtoV[ei,face_vts]:
                    x, y, z = msh.VXYZ[v]
                    assert x == x1 or x == x2
                    assert_ae(msh.EtoB[ei,fi], 1)
            else:
                assert_ae(msh.EtoE[nbr_ei,nbr_fi], ei)
                assert_ae(msh.EtoF[nbr_ei,nbr_fi], fi)
                assert_ae(msh.EtoR[nbr_ei,nbr_fi], nbr_ri)
                

def test_make_EtoEFRB_bc12():
    # check neighbor consistency and bc1, bc2
    bc={'x':'bc1', 'y':'bc2', 'z':'pbc'}
    msh = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix)
    
    for ei in range(msh.Nelem):
        for fi in range(msh.Nface):            
            nbr_ei = msh.EtoE[ei,fi]
            nbr_fi = msh.EtoF[ei,fi]
            nbr_ri = msh.EtoR[ei,fi]
            #print(f"ei={ei}, fi={fi}, nbr_ei={nbr_ei}, nbr_fi={nbr_fi}, rot={nbr_ri}")
            
            if nbr_ei == -99:                
                face_vts = [(1,0,2), (0,1,3), (1,2,3), (2,0,3)][fi]
                v1, v2, v3 = msh.EtoV[ei,face_vts]
                vx1, vy1, vz1 = msh.VXYZ[v1]
                vx2, vy2, vz2 = msh.VXYZ[v2]
                vx3, vy3, vz3 = msh.VXYZ[v3]
                
                if vx1 == x1 and vx2 == x1 and vx3 == x1:
                    assert_ae(msh.EtoB[ei,fi], 1)
                elif vx1 == x2 and vx2 == x2 and vx3 == x2:
                    assert_ae(msh.EtoB[ei,fi], 1)
                elif vy1 == y1 and vy2 == y1 and vy3 == y1:
                    assert_ae(msh.EtoB[ei,fi], 2)
                elif vy1 == y2 and vy2 == y2 and vy3 == y2:
                    assert_ae(msh.EtoB[ei,fi], 2)
                    
            else:
                assert_ae(msh.EtoE[nbr_ei,nbr_fi], ei)
                assert_ae(msh.EtoF[nbr_ei,nbr_fi], fi)
                assert_ae(msh.EtoR[nbr_ei,nbr_fi], nbr_ri)
                
                
def test_make_EtoEFRB_bc123():
    # check neighbor consistency and bc1, bc2, bc3
    bc={'x':'bc1', 'y':'bc2', 'z':'bc3'}
    msh = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix)
    
    for ei in range(msh.Nelem):
        for fi in range(msh.Nface):            
            nbr_ei = msh.EtoE[ei,fi]
            nbr_fi = msh.EtoF[ei,fi]
            nbr_ri = msh.EtoR[ei,fi]
            #print(f"ei={ei}, fi={fi}, nbr_ei={nbr_ei}, nbr_fi={nbr_fi}, rot={nbr_ri}")
            
            if nbr_ei == -99:                
                face_vts = [(1,0,2), (0,1,3), (1,2,3), (2,0,3)][fi]
                v1, v2, v3 = msh.EtoV[ei,face_vts]
                vx1, vy1, vz1 = msh.VXYZ[v1]
                vx2, vy2, vz2 = msh.VXYZ[v2]
                vx3, vy3, vz3 = msh.VXYZ[v3]
                
                if vx1 == x1 and vx2 == x1 and vx3 == x1:
                    assert_ae(msh.EtoB[ei,fi], 1)
                elif vx1 == x2 and vx2 == x2 and vx3 == x2:
                    assert_ae(msh.EtoB[ei,fi], 1)
                elif vy1 == y1 and vy2 == y1 and vy3 == y1:
                    assert_ae(msh.EtoB[ei,fi], 2)
                elif vy1 == y2 and vy2 == y2 and vy3 == y2:
                    assert_ae(msh.EtoB[ei,fi], 2)
                elif vz1 == z1 and vz2 == z1 and vz3 == z1:
                    assert_ae(msh.EtoB[ei,fi], 3)
                elif vz1 == z2 and vz2 == z2 and vz3 == z2:
                    assert_ae(msh.EtoB[ei,fi], 3)
                    
            else:
                assert_ae(msh.EtoE[nbr_ei,nbr_fi], ei)
                assert_ae(msh.EtoF[nbr_ei,nbr_fi], fi)
                assert_ae(msh.EtoR[nbr_ei,nbr_fi], nbr_ri)
                
                
def test_compare_mod_py_cpp():
    '''
    compare two core functions (python vs c++)
    '''     
    bc={'x':'bc1', 'y':'pbc', 'z':'pbc'}
    msh1 = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix='py')
    msh2 = Uniform3DSingle(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc, mod_suffix='cpp')
    
    assert_ae(msh1.EtoV, msh2.EtoV)
    #assert_ae(msh1.EtoE, msh2.EtoE)
    #assert_ae(msh1.EtoF, msh2.EtoF)
    #assert_ae(msh1.EtoR, msh2.EtoR)
    assert_ae(msh1.EtoB, msh2.EtoB)
