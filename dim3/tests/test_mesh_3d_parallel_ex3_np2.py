import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from gmsh_3d_partition import Gmsh3DPartition
from mesh_3d import Mesh3D

from test_mesh_3d_common import *
from test_mesh_3d_parallel import *


@pytest.fixture(scope="module")
def mesh_elem_ex3_np2(op_face_N3):
    gmsh_fpath = path.join(path.cwd, "gmsh", "ex3.tet.metis.dx0.4.np2.msh")
        
    mesh_elem1 = Gmsh3DPartition(gmsh_fpath, mypart=1)
    mesh_elem2 = Gmsh3DPartition(gmsh_fpath, mypart=2)
    
    return [mesh_elem1, mesh_elem2]


@pytest.fixture(scope="module")
def mesh_py_list(op_face_N3, mesh_elem_ex3_np2):
    return [Mesh3D(op_face_N3, m_e, mod_suffix='py') for m_e in mesh_elem_ex3_np2]


@pytest.fixture(scope="module")
def mesh_cpp_list(op_face_N3, mesh_elem_ex3_np2):
    return [Mesh3D(op_face_N3, m_e) for m_e in mesh_elem_ex3_np2]  # mod_suffix='cpp', default


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
    check_sendbuf_nfp_idxs(mesh_py_list)
                
                
def test_comm_nfp_idxs(mesh_py_list):
    check_comm_nfp_idxs(mesh_py_list)
                    
                    
def test_comm_px(mesh_py_list):
    check_comm_px(mesh_py_list)
    
    
def test_comm_pxyz(mesh_py_list):
    check_comm_pxyz(mesh_py_list)
