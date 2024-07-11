import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from uniform_3d_partition import Uniform3DPartition
from mesh_3d import Mesh3D

from test_mesh_3d_common import *
from test_mesh_3d_parallel import *


@pytest.fixture(scope="module")
def mesh_elem_uniform_np3(op_face_N3):
    npart = 3
    x1, x2, nx = 0, 1.4, 4
    y1, y2, ny = 0, 1.2, 3
    z1, z2, nz = 0, 1.0, 3
    bc = {'x':'pbc', 'y':'pbc', 'z':'pbc'}
    
    mesh_list = []
    for mypart in range(1, npart+1):
        mesh_list.append( Uniform3DPartition(mypart, npart, nx, ny, nz, x1, x2, y1, y2, z1, z2, bc) )
    
    return mesh_list


@pytest.fixture(scope="module")
def mesh_py_list(op_face_N3, mesh_elem_uniform_np3):
    return [Mesh3D(op_face_N3, m_e, mod_suffix='py') for m_e in mesh_elem_uniform_np3]


@pytest.fixture(scope="module")
def mesh_cpp_list(op_face_N3, mesh_elem_uniform_np3):
    return [Mesh3D(op_face_N3, m_e) for m_e in mesh_elem_uniform_np3]  # mod_suffix='cpp', default


def test_make_jacobian(mesh_py_list):
    for mesh in mesh_py_list:
        check_make_jacobian(mesh)

        
def test_jacobian(mesh_py_list):
    '''
    3D Jacobian metric은 physcial과 standard tetrahedron의 부피비율.
    standard의 부피는 8/6이고, 예제 격자 box의 모든 element 부피를 더하면 1.0*1.2*1.4=1.68이다.
    모든 J를 더하면 1.68/(8/6)=1.26 이 되어야 한다.
    '''
    jac_list = [np.sum(mesh.J) for mesh in mesh_py_list]    
    assert_aae(np.sum(jac_list), 1.26, 14)
    

def test_surface_jacobian(mesh_py_list):
    for mesh in mesh_py_list:
        check_surface_jacobian(mesh)
            

def test_normal_vectors(mesh_py_list):
    for mesh in mesh_py_list:
        check_normal_vectors(mesh)
        
                
def test_high_phy_coord(mesh_py_list):
    for mesh in mesh_py_list:
        check_high_phy_coord(mesh)
    
    
def test_compare_mod_cpp_py(mesh_cpp_list, mesh_py_list):
    for mesh_cpp, mesh_py in zip(mesh_cpp_list, mesh_py_list):
        check_compare_mod_cpp_py(mesh_cpp, mesh_py)


def test_sendbuf_nfp_idxs(mesh_py_list):
    '''
    for mesh in mesh_py_list:
        print(f"\nmypart={mesh.mesh_elem.mypart}")
        
        print("part2faces")
        for nbr_partTag, idx_list in mesh.part2faces.items():
            for elemIdx, fidx, ridx in idx_list:
                print(f"  {nbr_partTag}, ({elemIdx}, {fidx}, {ridx})")
        
        print("nbrFace2bufIdxs")
        for (nbr_partTag, nbr_elemTag, nbr_fidx), buf_idx in mesh.nbrFace2bufIdxs.items():
            print(f"  buf[{buf_idx}] = ({nbr_partTag}, {nbr_elemTag}, {nbr_fidx})")
                    
        print(f"sendbuf_face_idxs {mesh.sendbuf_face_idxs.shape}")
        print(f"{mesh.sendbuf_face_idxs}")
        
        print(f"sendbuf_nfp_idxs {mesh.sendbuf_nfp_idxs.shape}")
        print(f"{mesh.sendbuf_nfp_idxs}")
        
        print("vmapF")
        print(f"{mesh.vmapF}")
        
        #print("vmapP")
        #print(f"{mesh.vmapP}")
    '''            
    check_sendbuf_nfp_idxs(mesh_py_list)
                
                
def test_comm_nfp_idxs(mesh_py_list):
    check_comm_nfp_idxs(mesh_py_list)
                    
                    
def test_comm_px(mesh_py_list):
    check_comm_px(mesh_py_list)
    
    
def test_comm_pxyz(mesh_py_list):
    check_comm_pxyz(mesh_py_list)
