import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from gmsh_3d_single import Gmsh3DSingle
from mesh_3d import Mesh3D

from test_mesh_3d_common import *


@pytest.fixture(scope="module")
def mesh_elem_ex2(op_face_N3):
    gmsh_fpath = path.join(path.cwd, "gmsh", "ex2.tet.phy.dx0.4.msh")
    mesh_elem = Gmsh3DSingle(gmsh_fpath)    
    return mesh_elem


@pytest.fixture(scope="module")
def mesh_py(op_face_N3, mesh_elem_ex2):
    return Mesh3D(op_face_N3, mesh_elem_ex2, mod_suffix='py')


@pytest.fixture(scope="module")
def mesh_cpp(op_face_N3, mesh_elem_ex2):
    return Mesh3D(op_face_N3, mesh_elem_ex2)  # mod_suffix='cpp', default


def test_make_jacobian(mesh_py):
    check_make_jacobian(mesh_py)

        
def test_jacobian(mesh_py):
    check_jacobian(mesh_py)
    
    
def test_surface_jacobian(mesh_py):
    check_surface_jacobian(mesh_py)
            
            
def test_normal_vectors(mesh_py):
    check_normal_vectors(mesh_py)
                
                
def test_high_phy_coord(mesh_py):
    check_high_phy_coord(mesh_py)
    
    
def test_compare_mod_cpp_py(mesh_cpp, mesh_py):
    check_compare_mod_cpp_py(mesh_cpp, mesh_py)
    
    
def test_vmapM_vmapP_serial(mesh_py):
    '''
    check two (px,py,pz) coordinates of vmapM and vmapP are mached
    '''
    mesh = mesh_py    
    PX = mesh.PX.ravel()
    PY = mesh.PY.ravel()
    PZ = mesh.PZ.ravel()
    
    for ei in range(mesh.Nelem):
        for fi in range(mesh.tet.Nface):
            idxM = mesh.vmapM[ei,fi,:]
            idxP = mesh.vmapP[ei,fi,:]
            
            try:
                assert_aae(PX[idxM], PX[idxP], 14)
            except:
                assert_aae(np.abs(PX[idxM] - PX[idxP]), 1.4, 14)
            
            assert_aae(PY[idxM], PY[idxP], 14)
            assert_aae(PZ[idxM], PZ[idxP], 14)
