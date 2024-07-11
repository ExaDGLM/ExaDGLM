import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from tetra import TetraRef
from operator_3d_ref import Operator3DRef
from operator_3d_face import Operator3DFace, FaceMask
#from mesh_3d_gmsh import Mesh3DGmsh


interpN = 2
use_mp = False


def test_FaceMask():
    N = 3
    tet = TetraRef(N, interpN, use_mp)
    obj = FaceMask(tet)
    
    src_arr = [3, 6, 8, 9, 12, 14, 15, 17, 18, 19]
    
    nested_list = obj.arr_to_nested_list(src_arr)
    assert nested_list == [[3,6,8,9], [12,14,15], [17,18], [19]]
    
    rotate_nested_list = obj.rotate(nested_list)
    assert rotate_nested_list == [[19,17,12,3], [18,14,6], [15,8], [9]]
    
    invert_rotate_nested_list = obj.invert(rotate_nested_list)
    assert invert_rotate_nested_list == [[3,12,17,19], [6,14,18], [8,15], [9]]
    
    dst_arr = obj.nested_list_to_arr(invert_rotate_nested_list)
    assert_ae(dst_arr, [3,12,17,19,6,14,18,8,15,9])
    
    
def test_make_face_mask():
    N = 3
    tet = TetraRef(N, interpN, use_mp)
    Fmask, FmaskP = FaceMask(tet).make_face_mask()
    
    # face 0
    assert_ae(Fmask[0,:], [3, 2, 1, 0, 6, 5, 4, 8, 7, 9])
    assert_ae(FmaskP[0,0,:], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])    
    assert_ae(FmaskP[0,1,:], [3, 6, 8, 9, 2, 5, 7, 1, 4, 0])
    assert_ae(FmaskP[0,2,:], [9, 7, 4, 0, 8, 5, 1, 6, 2, 3])
    
    # face 1
    assert_ae(Fmask[1,:], [0, 1, 2, 3, 10, 11, 12, 16, 17, 19])
    assert_ae(FmaskP[1,0,:], [3, 2, 1, 0, 12, 11, 10, 17, 16, 19])
    assert_ae(FmaskP[1,1,:], [0, 10, 16, 19, 1, 11, 17, 2, 12, 3])
    assert_ae(FmaskP[1,2,:], [19, 17, 12, 3, 16, 11, 2, 10, 1, 0])
    
    # face 2
    assert_ae(Fmask[2,:], [3, 6, 8, 9, 12, 14, 15, 17, 18, 19])
    assert_ae(FmaskP[2,0,:], [9, 8, 6, 3, 15, 14, 12, 18, 17, 19])
    assert_ae(FmaskP[2,1,:], [3, 12, 17, 19, 6, 14, 18, 8, 15, 9])
    assert_ae(FmaskP[2,2,:], [19, 18, 15, 9, 17, 14, 8, 12, 6, 3])
    
    # face 3
    assert_ae(Fmask[3,:], [9, 7, 4, 0, 15, 13, 10, 18, 16, 19])
    assert_ae(FmaskP[3,0,:], [0, 4, 7, 9, 10, 13, 15, 16, 18, 19])    
    assert_ae(FmaskP[3,1,:], [9, 15, 18, 19, 7, 13, 16, 4, 10, 0])
    assert_ae(FmaskP[3,2,:], [19, 16, 10, 0, 18, 13, 4, 15, 7, 9])
    

def test_Emat():
    N = 6
    tet = TetraRef(N, interpN, use_mp)
    op_ref = Operator3DRef(tet)
    op_face = Operator3DFace(op_ref)
    
    # LIFT mass matrix 적분
    # standard triangle의 넓이가 2이므로, 2*4=8    
    a = np.sum(op_face.Emat)
    assert_ae(a, 8)
