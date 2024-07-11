import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from tetra import TetraRef
from operator_3d_ref import Operator3DRef
from operator_3d_face import Operator3DFace
from gmsh_3d_single import Gmsh3DSingle
from mesh_3d import Mesh3D


@pytest.fixture(scope="module")
def op_face_N3():
    N = 3
    tet = TetraRef(N)
    op_ref = Operator3DRef(tet)
    op_face = Operator3DFace(op_ref)    
    return op_face
    

def check_make_jacobian(mesh):
    J, rx, ry, rz, sx, sy, sz, tx, ty, tz = mesh.make_jacobian_elem()
    J2, rx2, ry2, rz2, sx2, sy2, sz2, tx2, ty2, tz2 = mesh.make_jacobian_np()
    
    precision = 11
    for ei in range(mesh.Nelem):
        assert_aae( J[ei],  J2[ei,0], precision)
        assert_aae(rx[ei], rx2[ei,0], precision)
        assert_aae(ry[ei], ry2[ei,0], precision)
        assert_aae(rz[ei], rz2[ei,0], precision)
        assert_aae(sx[ei], sx2[ei,0], precision)
        assert_aae(sy[ei], sy2[ei,0], precision)
        assert_aae(sz[ei], sz2[ei,0], precision)
        assert_aae(tx[ei], tx2[ei,0], precision)
        assert_aae(ty[ei], ty2[ei,0], precision)
        assert_aae(tz[ei], tz2[ei,0], precision)        

        
def check_jacobian(mesh):
    '''
    3D Jacobian metric은 physcial과 standard tetrahedron의 부피비율.
    standard의 부피는 8/6이고, 예제 격자 box의 모든 element 부피를 더하면 1.0*1.2*1.4=1.68이다.
    모든 J를 더하면 1.68/(8/6)=1.26 이 되어야 한다.
    '''
    assert_aae(np.sum(mesh.J), 1.26, 14)
    
    
def check_surface_jacobian(mesh):
    '''
    surface jacobian은 physical과 standard triangle의 면적비율
    standard triangle 면적은 2
    sJ = (physical 면적)/(standard 면적)
    '''    
    def triangle_area_3d(xyz0, xyz1, xyz2):        
        vec1 = xyz1 - xyz0
        vec2 = xyz2 - xyz0
        return 0.5*np.linalg.norm(np.cross(vec1, vec2))
    
    for ei, vts in enumerate(mesh.EtoV):
        vxyz = np.column_stack([mesh.VX[vts], mesh.VY[vts], mesh.VZ[vts]])
        
        face_vts = [(1,0,2), (0,1,3), (1,2,3), (2,0,3)]
        for fi, (v0,v1,v2) in enumerate(face_vts):
            phy_area = triangle_area_3d(vxyz[v0], vxyz[v1], vxyz[v2])
            ref_area = 2
            assert_aae(mesh.sJ[ei,fi], phy_area/ref_area, 14)
            

def check_normal_vectors(mesh):
    '''
    1. The dot product of the normal vector and the side vector should be zero.
    2. The cross product of the normal vector and the side vector should be positive.
    3. The normal vectors should be normalized.
    '''
    for ei, vts in enumerate(mesh.EtoV):
        vxyz = np.column_stack([mesh.VX[vts], mesh.VY[vts], mesh.VZ[vts]])
        
        face_vts = [(1,0,2), (0,1,3), (1,2,3), (2,0,3)]
        for fi, (v0,v1,v2) in enumerate(face_vts):
            nx, ny, nz = mesh.NX[ei,fi], mesh.NY[ei,fi], mesh.NZ[ei,fi]
            nvec = (nx, ny, nz)
            
            vec1 = vxyz[v1] - vxyz[v0]
            vec2 = vxyz[v2] - vxyz[v0]
            assert_aae(np.dot(nvec, (vec1 + vec2)), 0, 14)
            
            assert np.dot(nvec, np.cross(vec1, vec2)) > 0
            
            assert_aae(np.sqrt(nx*nx + ny*ny + nz*nz), 1, 14)

                
def check_high_phy_coord(mesh):
    '''
    (Nelem, high_Np).T = (high_Np,Np)@(Nelem,Np).T
    (high_Np, Nelem)   = (high_Np,Np)@(Np,Nelem)
    
    high_PX.T = interM@PX.TS
    '''
    assert_aae(mesh.high_PX.T, mesh.op_ref.interpM@mesh.PX.T, 14)
    assert_aae(mesh.high_PY.T, mesh.op_ref.interpM@mesh.PY.T, 14)
    assert_aae(mesh.high_PZ.T, mesh.op_ref.interpM@mesh.PZ.T, 14)
    
    
def check_compare_mod_cpp_py(mesh1, mesh2):
    '''
    compare two core functions (python vs c++)
    '''
    assert_ae(mesh1.EtoE, mesh2.EtoE)
    assert_ae(mesh1.EtoF, mesh2.EtoF)
    assert_ae(mesh1.EtoR, mesh2.EtoR)
    
    assert_aae(mesh1.PX, mesh2.PX, 14)
    assert_aae(mesh1.PY, mesh2.PY, 14)
    assert_aae(mesh1.PZ, mesh2.PZ, 14)
    
    assert_aae(mesh1.J, mesh2.J, 14)
    assert_aae(mesh1.rx, mesh2.rx, 14)
    assert_aae(mesh1.ry, mesh2.ry, 14)
    assert_aae(mesh1.rz, mesh2.rz, 14)    
    assert_aae(mesh1.sx, mesh2.sx, 14)
    assert_aae(mesh1.sy, mesh2.sy, 14)
    assert_aae(mesh1.sz, mesh2.sz, 14)
    assert_aae(mesh1.tx, mesh2.tx, 14)
    assert_aae(mesh1.ty, mesh2.ty, 14)
    assert_aae(mesh1.tz, mesh2.tz, 14)
    
    assert_aae(mesh1.NX, mesh2.NX, 14)
    assert_aae(mesh1.NY, mesh2.NY, 14)
    assert_aae(mesh1.NZ, mesh2.NZ, 14)
    
    assert_aae(mesh1.dt_scale, mesh2.dt_scale, 14)
    
    assert_ae(mesh1.vmapM, mesh2.vmapM)
    assert_ae(mesh1.vmapP, mesh2.vmapP)
    
    assert_aae(mesh1.high_PX, mesh2.high_PX, 14)
    assert_aae(mesh1.high_PY, mesh2.high_PY, 14)
    assert_aae(mesh1.high_PZ, mesh2.high_PZ, 14)
