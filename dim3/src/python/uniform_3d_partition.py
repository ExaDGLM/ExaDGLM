import os
import sys
import numpy as np
from numpy.testing import assert_array_equal as assert_ae

sys.path.append("../src/python")
from uniform_3d_single import Uniform3DSingle


class Uniform3DPartition(Uniform3DSingle):
    def __init__(self, mypart, npart, nx, ny, nz, x1, x2, y1, y2, z1, z2, bc={'x':'pbc', 'y':'pbc', 'z':'pbc'}):
        '''
        Partitioned Uniform tetahedral mesh
        
        Note: only support npart=2 or 3 when nx-1 is multiple of 2 or 3
    
        Outputs (addition to Uniform3DSingle)
         - Nelem
         - VXYZ  # shape (Nnode,3)
         - EtoV  # shape (Nelem,4)
         - EtoE  # shape (Nelem,Nface), fill_value=-99
         - EtoF  # shape (Nelem,Nface)
         - EtoR  # shape (Nelem,Nface)
         - EtoB  # shape (Nelem,Nface), fill_value=-99
         
         - npart
         - mypart
         - EtoP (Nelem,Nface) -> partition number
         - part2faces      # {partitionTag: [(eidx,fidx,ridx), ...], ...}
         - nbrFace2bufIdxs # {(partitionTag, elemTag, fidx):buf_idx, ...}         
        '''
        
        assert npart in [2,3], f"[Error] currently, npart is only supported 2 or 3 (npart={npart})"
        assert (nx-1)%npart == 0, f"[Error] nx-1 should be multiple of 2 or 3 (nx-1={nx-1})"
        
        super().__init__(nx, ny, nz, x1, x2, y1, y2, z1, z2, bc)
        self.global_Nelem = self.Nelem
        self.global_VXYZ = self.VXYZ
        self.global_EtoV = self.EtoV
        self.global_EtoE = self.EtoE
        self.global_EtoF = self.EtoF
        self.global_EtoR = self.EtoR
        self.global_EtoB = self.EtoB
        
        self.Nelem = self.global_Nelem//npart
        self.npart = npart
        self.mypart = mypart
        
        self.VXYZ, self.EtoV, self.EtoE, self.EtoF, self.EtoR, self.EtoB, self.EtoP, \
        self.part2faces, self.nbrFace2bufIdxs, self.elemTag2Idxs, self.elemIdx2Tags = \
            self.make_local_EtoVEFRBP()
        
        
    def make_local_EtoVEFRBP(self):
        #
        # global elemTag <-> local elemIdx
        #
        elem2partTags = np.zeros(self.global_Nelem, 'i4')
        elemTag2Idxs = np.zeros(self.global_Nelem, 'i4')
        elemIdx2Tags = np.zeros(self.Nelem, 'i4')
        
        elemTag = 0
        for partTag in range(1, self.npart+1):
            for elemIdx in range(self.Nelem):
                elem2partTags[elemTag] = partTag
                elemTag2Idxs[elemTag] = elemIdx
                
                if partTag == self.mypart:
                    elemIdx2Tags[elemIdx] = elemTag
                
                elemTag += 1
        
        #
        # extract local index arrays
        #
        mypart = self.mypart
        Nelem, Nface = self.Nelem, self.Nface        
        
        EtoV = np.zeros((Nelem, 4), 'i4')
        EtoE = np.full((Nelem, Nface), -99, 'i4')
        EtoF = np.zeros((Nelem, Nface), 'i4')
        EtoR = np.zeros((Nelem, Nface), 'i4')        
        EtoB = np.full((Nelem, Nface), -99, 'i4')
        EtoP = np.full((Nelem, Nface), mypart, 'i4')
        part2faces = {}       # {partTag: [(eidx,fidx,ridx), ...], ...}
        nbr_part2faces = {}   # {partTag: [(nbr_elemTag, nbr_fidx), ...], ...} to make nbrFace2bufIdxs
        nbrFace2bufIdxs = {}  # {(partTag, nbr_elemTag, nbr_fidx):buf_idx, ...}
        
        #
        # EtoVFRBP, part2faces
        #
        elemIdx = 0
        for elemTag in range(Nelem*(mypart-1), Nelem*mypart):
            EtoV[elemIdx,:] = self.global_EtoV[elemTag,:]
            EtoF[elemIdx,:] = self.global_EtoF[elemTag,:]
            EtoR[elemIdx,:] = self.global_EtoR[elemTag,:]
            EtoB[elemIdx,:] = self.global_EtoB[elemTag,:]
            
            for fidx in range(Nface):            
                nbr_elemTag = self.global_EtoE[elemTag,fidx]
                nbr_partTag = elem2partTags[nbr_elemTag]

                if nbr_partTag == mypart:
                    EtoE[elemIdx,fidx] = elemTag2Idxs[nbr_elemTag]
                    
                else:
                    EtoE[elemIdx,fidx] = nbr_elemTag
                    EtoP[elemIdx,fidx] = nbr_partTag
                    
                    ridx = EtoR[elemIdx,fidx]
                    if not nbr_partTag in part2faces:
                        part2faces[nbr_partTag] = []
                    part2faces[nbr_partTag].append( (elemIdx, fidx, ridx) )
                    
                    nbr_fidx = EtoF[elemIdx,fidx]                    
                    if not nbr_partTag in nbr_part2faces:
                        nbr_part2faces[nbr_partTag] = []
                    nbr_part2faces[nbr_partTag].append( (nbr_elemTag, nbr_fidx) )
            
            elemIdx += 1
            
        #
        # sorted part2faces
        #
        part2faces = {k: part2faces[k] for k in sorted(nbr_part2faces.keys())}
        
        #
        # nbrFace2bufIdxs
        #
        buf_idx = 0
        for nbr_partTag in sorted(nbr_part2faces.keys()):
            idx_list = sorted(nbr_part2faces[nbr_partTag], key=lambda x: x[0])
            
            for nbr_elemTag, nbr_fidx in idx_list:
                nbrFace2bufIdxs[(nbr_partTag, nbr_elemTag, nbr_fidx)] = buf_idx
                buf_idx += 1
                
        #
        # local VXYZ
        #
        vidxs = np.array( sorted( set( EtoV.ravel() ) ), 'i4')
        VXYZ = self.global_VXYZ[vidxs,:].copy()
        
        for ei in range(Nelem):
            for fi in range(Nface):
                global_idx = EtoV[ei,fi]
                local_idx = np.where(vidxs == global_idx)[0]
                EtoV[ei,fi] = local_idx
            
        return VXYZ, EtoV, EtoE, EtoF, EtoR, EtoB, EtoP, part2faces, nbrFace2bufIdxs, elemTag2Idxs, elemIdx2Tags
