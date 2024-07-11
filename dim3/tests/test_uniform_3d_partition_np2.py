import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path
from uniform_3d_partition import Uniform3DPartition


npart = 2
x1, x2, nx = 0, 2, 3
y1, y2, ny = 1, 3, 3
z1, z2, nz = 2, 5, 4 
bc={'x':'pbc', 'y':'pbc', 'z':'pbc'}

msh1 = Uniform3DPartition(1, npart, nx, ny, nz, x1, x2, y1, y2, z1, z2, bc)
msh2 = Uniform3DPartition(2, npart, nx, ny, nz, x1, x2, y1, y2, z1, z2, bc)


def test_VXYZ():
    assert_ae(msh1.VXYZ[  : 4], [[0,1,2], [0,1,3], [0,1,4], [0,1,5]])
    assert_ae(msh1.VXYZ[ 8:12], [[0,3,2], [0,3,3], [0,3,4], [0,3,5]])
    assert_ae(msh1.VXYZ[12:16], [[1,1,2], [1,1,3], [1,1,4], [1,1,5]])
    assert_ae(msh1.VXYZ[20:  ], [[1,3,2], [1,3,3], [1,3,4], [1,3,5]])
    
    assert_ae(msh2.VXYZ[  : 4], [[1,1,2], [1,1,3], [1,1,4], [1,1,5]])
    assert_ae(msh2.VXYZ[ 8:12], [[1,3,2], [1,3,3], [1,3,4], [1,3,5]])
    assert_ae(msh2.VXYZ[12:16], [[2,1,2], [2,1,3], [2,1,4], [2,1,5]])
    assert_ae(msh2.VXYZ[20:  ], [[2,3,2], [2,3,3], [2,3,4], [2,3,5]])
        

def test_EtoV():
    # hexahedron nodeTags = [v0, v1, v2, v3, v4, v5, v6, v7]
    # elem0 nodeTags = [v0, v7, v3, v1]
    # elem1 nodeTags = [v0, v7, v2, v3]
    # elem2 nodeTags = [v0, v7, v6, v2]
    # elem3 nodeTags = [v0, v7, v4, v6]
    # elem4 nodeTags = [v0, v7, v5, v4]
    # elem5 nodeTags = [v0, v7, v1, v5]    
    
    for msh in [msh1, msh2]:
        # check Nelem size
        assert_ae(msh.Nelem, 6*(nz-1)*(ny-1)*(nx-1)//npart)
        assert_ae(msh.EtoV.shape, (msh1.Nelem, 4))
        
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

        # 6 tetrahedrons in the last (6th) hexahedron
        assert_ae(msh.EtoV[30,:], [6,23,11, 7])  # 6 7 10 11 18 19 22 23
        assert_ae(msh.EtoV[31,:], [6,23,10,11])
        assert_ae(msh.EtoV[32,:], [6,23,22,10])
        assert_ae(msh.EtoV[33,:], [6,23,18,22])
        assert_ae(msh.EtoV[34,:], [6,23,19,18])
        assert_ae(msh.EtoV[35,:], [6,23, 7,19])
    
    
def test_consistency_EtoEFRP_pbc():
    '''
    check neighbor consistency across partitions
    '''
    msh_list = [msh1, msh2]
    
    for msh in msh_list:
        for ei in range(msh.Nelem):
            for fi in range(msh.Nface):            
                nbr_ei = msh.EtoE[ei,fi]
                nbr_fi = msh.EtoF[ei,fi]
                nbr_ri = msh.EtoR[ei,fi]
                nbr_part = msh.EtoP[ei,fi]

                if nbr_part == msh.mypart:
                    assert_ae(msh.EtoE[nbr_ei,nbr_fi], ei)
                    assert_ae(msh.EtoF[nbr_ei,nbr_fi], fi)
                    assert_ae(msh.EtoR[nbr_ei,nbr_fi], nbr_ri)
                
                else:
                    nbr_msh = msh_list[nbr_part-1]
                    nbr_elemTag = nbr_ei
                    nbr_elemIdx = nbr_msh.elemTag2Idxs[nbr_elemTag]
                    elemTag = nbr_msh.EtoE[nbr_elemIdx,nbr_fi]
                    elemIdx = msh.elemTag2Idxs[elemTag]
                            
                    assert_ae(elemIdx, ei)
                    assert_ae(nbr_msh.EtoF[nbr_elemIdx,nbr_fi], fi)
                    assert_ae(nbr_msh.EtoR[nbr_elemIdx,nbr_fi], nbr_ri)
                    
                    
def test_consistency_buffer():
    '''
    part2faces = {}       # {partTag: [(eidx,fidx,ridx), ...], ...}
    nbrFace2bufIdxs = {}  # {(partTag, elemTag, fidx):buf_idx, ...}
    '''    
    # check the number of list
    buf_size = 6*2*2  # 6(cube faces)*2(tet_faces/cube_face)*2(PBC)
    assert_ae(len(msh1.part2faces[2]), buf_size)
    assert_ae(len(msh2.part2faces[1]), buf_size)
    
    assert_ae(len(msh1.nbrFace2bufIdxs), buf_size)
    assert_ae(len(msh2.nbrFace2bufIdxs), buf_size)
    
    #
    # buffers in partition 1
    #
    sendbuf1 = np.zeros((buf_size,3), 'i4')  # (partTag, elemIdx, fidx)
    recvbuf1 = np.zeros((buf_size,3), 'i4')  # (partTag, nbr_elemTag, nbr_fidx)
    
    for i, (eidx, fidx, ridx) in enumerate(msh1.part2faces[2]):
        sendbuf1[i][:] = (1, msh1.elemIdx2Tags[eidx], fidx)
    
    for (partTag, elemTag, fidx), buf_idx in msh1.nbrFace2bufIdxs.items():
        recvbuf1[buf_idx][:] = (partTag, elemTag, fidx) 
        
    #
    # buffers in partition 2
    #
    sendbuf2 = np.zeros((buf_size,3), 'i4')  # (partTag, elemIdx, fidx)
    recvbuf2 = np.zeros((buf_size,3), 'i4')  # (partTag, nbr_elemTag, nbr_fidx)
        
    for i, (eidx, fidx, ridx) in enumerate(msh2.part2faces[1]):
        sendbuf2[i][:] = (2, msh2.elemIdx2Tags[eidx], fidx)
    
    for (partTag, elemTag, fidx), buf_idx in msh2.nbrFace2bufIdxs.items():
        recvbuf2[buf_idx][:] = (partTag, elemTag, fidx) 
    
    #
    # check buffer consistency
    #
    assert_ae(sendbuf1, recvbuf2)
    assert_ae(sendbuf2, recvbuf1)
