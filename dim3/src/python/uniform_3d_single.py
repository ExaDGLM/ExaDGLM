import os
import re
import importlib
import numpy as np
from numpy.testing import assert_array_equal as assert_ae


class Uniform3DSingle:
    def __init__(self, nx, ny, nz, x1, x2, y1, y2, z1, z2, bc={'x':'pbc', 'y':'pbc', 'z':'pbc'}, mod_suffix='cpp'):
        '''
        Uniform tetahedral mesh
        using 6 tetrahedrons in a box
        
        Outputs
         - Nelem
         - VXYZ  # shape (Nnode,3)
         - EtoV  # shape (Nelem,4)
         - EtoE  # shape (Nelem,Nface), fill_value=-99
         - EtoF  # shape (Nelem,Nface)
         - EtoR  # shape (Nelem,Nface)
         - EtoB  # shape (Nelem,Nface), fill_value=-99
        '''
        
        self.nx, self.ny, self.nz = nx, ny, nz
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2
        self.bc = bc
        
        self.Ncube = (nx-1)*(ny-1)*(nz-1)
        self.Nelem = self.Ncube*6
        self.Nface = 4  # 4 faces in a tetrahedron
        
        # import core module
        self.mod = importlib.import_module(f"uniform_3d_single_{mod_suffix}")   
                
        self.VXYZ = self.make_VXYZ()
        self.EtoV = self.make_EtoV()
        self.EtoE, self.EtoF, self.EtoR, self.EtoB = self.make_EtoEFRB()
        
        
    def make_VXYZ(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        x1, x2 = self.x1, self.x2
        y1, y2 = self.y1, self.y2
        z1, z2 = self.z1, self.z2
        
        x1d = np.linspace(x1, x2, nx)
        y1d = np.linspace(y1, y2, ny)
        z1d = np.linspace(z1, z2, nz)
        x3d, y3d, z3d = np.meshgrid(x1d, y1d, z1d, indexing='ij')  # 'ij' means matrix indexing
        xcoord = x3d.ravel()
        ycoord = y3d.ravel()
        zcoord = z3d.ravel()
        
        # 3x(N,) arrays -> (N,3) array
        VXYZ = np.column_stack((xcoord, ycoord, zcoord))

        return VXYZ


    def make_EtoV(self):
        '''
        hexahedron nodeTags = [v0, v1, v2, v3, v4, v5, v6, v7]  
        elem0 nodeTags = [v0, v7, v3, v1]  
        elem1 nodeTags = [v0, v7, v2, v3]  
        elem2 nodeTags = [v0, v7, v6, v2]  
        elem3 nodeTags = [v0, v7, v4, v6]  
        elem4 nodeTags = [v0, v7, v5, v4]  
        elem5 nodeTags = [v0, v7, v1, v5] 
        '''
        
        Nelem, Ncube = self.Nelem, self.Ncube
        nx, ny, nz = self.nx, self.ny, self.nz
        ijk2nodeTags = np.arange(nx*ny*nz).reshape(nx,ny,nz)
        
        EtoV = np.zeros((Nelem, 4), 'i4')
        
        self.mod.update_EtoV(nx, ny, nz, ijk2nodeTags, EtoV)
                    
        return EtoV
    
    
    def make_EtoEFRB(self):
        '''
        tetrahedron nodeTags = [v0, v1, v2, v3]  
        faceTags = [(v1,v0,v2), (v0,v1,v3), (v1,v2,v3), (v2,v0,v3)]
              face0    face1    face2    face3  
        elem0 (7,0,3), (0,7,1), (7,3,1), (3,0,1) -> (inner, inner, zP, xM)  
        elem1 (7,0,2), (0,7,3), (7,2,3), (2,0,3) -> (inner, inner, yP, xM)  
        elem2 (7,0,6), (0,7,2), (7,6,2), (6,0,2) -> (inner, inner, yP, zM)  
        elem3 (7,0,4), (0,7,6), (7,4,6), (4,0,6) -> (inner, inner, xP, zM)  
        elem4 (7,0,5), (0,7,4), (7,5,4), (5,0,4) -> (inner, inner, xP, yM)  
        elem5 (7,0,1), (0,7,5), (7,1,5), (1,0,5) -> (inner, inner, zP, yM)
        
                                                            --> x
         r2,inv   r1,inv   r0,inv                          -------                          r0,inv   r1,inv   r2,inv
        (7,4,6), (6,7,4), (4,6,7) <- (6,4,7) <> (2,0,3) f3 |e1|e3| f2 (7,4,6) <> (3,0,2) -> (0,3,2), (3,2,0), (2,0,3)
        (5,4,7), (7,5,4), (4,7,5) <- (7,4,5) <> (3,0,1) f3 |e0|e4| f2 (7,5,4) <> (3,1,0) -> (1,3,0), (3,0,1), (0,1,3)

                                                            --> y
         r2,inv   r1,inv   r0,inv                          -------                          r0,inv   r1,inv   r2,inv
        (7,2,3), (3,7,2), (2,3,7) <- (3,2,7) <> (1,0,5) f3 |e5|e1| f2 (7,2,3) <> (5,0,1) -> (0,5,1), (5,1,0), (1,0,5)
        (6,2,7), (7,6,2), (2,7,6) <- (7,2,6) <> (5,0,4) f3 |e4|e2| f2 (7,6,2) <> (5,4,0) -> (4,5,0), (5,0,4), (0,4,5)

                                                            --> z
         r2,inv   r1,inv   r0,inv                          -------                          r0,inv   r1,inv   r2,inv
        (7,1,5), (5,7,1), (1,5,7) <- (5,1,7) <> (4,0,6) f3 |e3|e5| f2 (7,1,5) <> (6,0,4) -> (0,6,4), (6,4,0), (4,0,6)
        (3,1,7), (7,3,1), (1,7,3) <- (7,1,3) <> (6,0,2) f3 |e2|e0| f2 (7,3,1) <> (6,2,0) -> (2,6,0), (6,0,2), (0,2,6)
        '''
        
        Nelem, Nface = self.Nelem, self.Nface
        nx, ny, nz = self.nx, self.ny, self.nz
        xidxs = np.array([nx-2] + list(range(nx-1)) + [0], 'i4')
        yidxs = np.array([ny-2] + list(range(ny-1)) + [0], 'i4')
        zidxs = np.array([nz-2] + list(range(nz-1)) + [0], 'i4')
        ijk2cubeTags = np.arange((nx-1)*(ny-1)*(nz-1)).reshape((nx-1),(ny-1),(nz-1))
        phyTags = np.full(3, -1, 'i4')
        
        if self.bc['x'] != 'pbc':
            phyTags[0] = int( re.findall(r'bc(\d+)', self.bc['x'])[0] )
        if self.bc['y'] != 'pbc':
            phyTags[1] = int( re.findall(r'bc(\d+)', self.bc['y'])[0] )
        if self.bc['z'] != 'pbc':
            phyTags[2] = int( re.findall(r'bc(\d+)', self.bc['z'])[0] )
        
        EtoE = np.full((Nelem, Nface), -99, 'i4')
        EtoF = np.zeros((Nelem, Nface), 'i4')
        EtoR = np.zeros((Nelem, Nface), 'i4')
        EtoB = np.full((Nelem, Nface), -99, 'i4')
        
        self.mod.update_EtoEFRB(nx, ny, nz, xidxs, yidxs, zidxs, ijk2cubeTags, phyTags, EtoE, EtoF, EtoR, EtoB)
                    
        return EtoE, EtoF, EtoR, EtoB
