import numpy as np
import importlib

from tetra import make_nodes_equilateral, xyz2rst, rst2abc

NODETOL = 1e-12
                
                
class Mesh3D:
    def __init__(self, op_face, mesh_elem, mod_suffix='cpp'):
        '''
        [input]
        op_face   : Operator3DFace object
        mesh_elem : MeshElem3D object (Gmsh3DSingle, Gmsh3DPartition)
        mod_suffix: 'py' or 'cpp'
    
        [output]
        x, y, z : coordinates in the equilateral triangle
        r, s, t : coordinates in a standrad triangle
        a, b, c : coordinates for the Legendre polynomial (orthonomal basis)
        
        VX,VY,VZ: (x,y,z) coordinate of vertices
        PX,PY,PZ: (x,y,z) coordinate of all points, shape (Nelem,Np)
        NX,NY,NZ: (x,y,z) coordinate of normal vectors, shape (Nelem,Nface)        
        
        EtoE: element to neighbor elements
        EtoF: element to face number of neighbor elements
        EtoR: element to rotation number of neighbor elements
        EtoP: element to partitionTag of neighbor face (start from 1)
        EtoB: element to boundary condition number defined in Gmsh
        => shape (Nelem,Nface)
        
        Jacobian metrics
        J, rx, ry, rz, sx, sy, sz, tx, ty, tz
        => shape (Nelem,)
        
        sJ : surface Jacobian
        Fscale: sJ/J
        => shape (Nelem,Nface)
        
        # vmap index table
        vmapM: (Nelem,Nface,Nfp) Nfp indices on a surface
        vmapP: (Nelem,Nface,Nfp) corresponding Nfp indices on an adjacent surface applied inversion and rotation
        vmapF: (Nelem,Nface) simplified version of vmapP
        
        # for MPI/NCCL communications
        sendbuf_idxs: (face_size, Nfp), index array to copy from source array to sendbuf
        comm_idxs: (nbr_rank, start_index, data_count)
        '''
        
        self.op_face = op_face
        self.op_ref  = op_face.op_ref
        self.tet     = op_face.tet
        self.mesh_elem = mesh_elem        
        
        # dimensions
        self.Nelem = mesh_elem.Nelem
        self.Nface = self.tet.Nface
        self.Np    = self.tet.Np
        self.Nfp   = self.tet.Nfp
        
        # physical coordinates of vertices
        self.VX = mesh_elem.VXYZ[:,0]
        self.VY = mesh_elem.VXYZ[:,1]
        self.VZ = mesh_elem.VXYZ[:,2]
        
        # vertex index of elements
        self.EtoV  = mesh_elem.EtoV
        
        # connectivity between vertices, elements and faces
        self.EtoE = mesh_elem.EtoE
        self.EtoF = mesh_elem.EtoF
        self.EtoR = mesh_elem.EtoR
        self.EtoB = mesh_elem.EtoB
        
        # import core module
        self.mod = importlib.import_module(f"mesh_3d_{mod_suffix}")   
                
        # generate quadrature points in physical coordinates
        self.PX, self.PY, self.PZ = self.make_inner_points(self.tet.rst)
        
        # Jacobian metrics
        self.J, \
        self.rx, self.ry, self.rz, \
        self.sx, self.sy, self.sz, \
        self.tx, self.ty, self.tz = self.make_jacobian_elem()        
        
        # normal vectors and surface Jacobian
        self.NX, self.NY, self.NZ, self.sJ, self.Fscale = self.make_normal_vectors()
        
        # ratio between volume and face Jacobian to choose timestep
        N = self.tet.N
        self.dt_scale = 1/(np.max(self.Fscale)*N*N)
        
        #
        # vmap and communication indices
        # serial/parallel version
        #
        if not hasattr(mesh_elem, 'EtoP'):  # partitioned version            
            self.myrank = -1
            self.vmapM, self.vmapP, self.vmapF = self.make_vmapM_vmapP_serial()
            
        else:            
            self.myrank = mesh_elem.mypart - 1
            self.EtoP = mesh_elem.EtoP
            self.nbrFace2bufIdxs = mesh_elem.nbrFace2bufIdxs  # {(partitionTag, elemTag, fidx):buf_idx,...}
            self.part2faces = mesh_elem.part2faces # {partitionTag: [(eidx,fidx,ridx), ...], ...}
            
            self.vmapM, self.vmapP, self.vmapF = self.make_vmapM_vmapP_parallel()
            
            self.part_size, \
            self.sendbuf_face_idxs, self.comm_face_idxs, \
            self.sendbuf_nfp_idxs, self.comm_nfp_idxs = self.make_comm_idxs()
                
        # high-order coordinates
        self.high_PX, self.high_PY, self.high_PZ = self.make_inner_points(self.tet.high_rst)
        
    
    def make_inner_points(self, rst):
        '''
        make inner points in each element using the standard tetrahedron
        with polynomial order N
        '''        
        Nelem = self.Nelem
        Np = len(rst)
        VX, VY, VZ, EtoV = self.VX, self.VY, self.VZ, self.EtoV
        
        PX = np.zeros((Nelem, Np))
        PY = np.zeros((Nelem, Np))
        PZ = np.zeros((Nelem, Np))
        
        r = rst[:,0]
        s = rst[:,1]
        t = rst[:,2]
        self.mod.update_PXYZ(EtoV, r, s, t, VX, VY, VZ, PX, PY, PZ)
            
        return PX, PY, PZ
        
        
    def make_jacobian_elem(self):
        '''
        jacobian metric between physical and standard coordinates
        note : the metrices are the same in an element
        '''
        Nelem = self.Nelem
        VX, VY, VZ, EtoV = self.VX, self.VY, self.VZ, self.EtoV
        
        J, rx, ry, rz, sx, sy, sz, tx, ty, tz = [np.zeros(Nelem) for _ in range(10)]
        
        self.mod.update_jacobian(EtoV, VX, VY, VZ, J, rx, ry, rz, sx, sy, sz, tx, ty, tz)        
        
        return J, rx, ry, rz, sx, sy, sz, tx, ty, tz
    
    
    def make_jacobian_np(self):
        '''
        jacobian metric between physical and standard coordinates
        in all quadrature points
        '''
        Nelem, Np = self.Nelem, self.tet.Np        
        PX, PY, PZ = self.PX, self.PY, self.PZ
        op = self.op_ref
        
        J, rx, ry, rz, sx, sy, sz, tx, ty, tz = [np.zeros((Nelem,Np)) for _ in range(10)]
        
        self.mod.update_jacobian_np(op.Dr, op.Ds, op.Dt, PX, PY, PZ, J, rx, ry, rz, sx, sy, sz, tx, ty, tz)
        
        return J, rx, ry, rz, sx, sy, sz, tx, ty, tz
    
    
    def make_normal_vectors(self):
        '''
        Compute outward pointing normals at elements faces and surface Jacobians
        '''
        Nelem, Nface = self.Nelem, self.tet.Nface
        J = self.J
        rx, ry, rz = self.rx, self.ry, self.rz
        sx, sy, sz = self.sx, self.sy, self.sz
        tx, ty, tz = self.tx, self.ty, self.tz
                
        # normal vectors and surface Jacobian
        nx = np.zeros((Nelem, Nface))
        ny = np.zeros((Nelem, Nface))
        nz = np.zeros((Nelem, Nface))
        sJ = np.zeros((Nelem, Nface))
        Fscale = np.zeros((Nelem, Nface))
    
        self.mod.update_normal_vectors(J, rx, ry, rz, sx, sy, sz, tx, ty, tz, nx, ny, nz, sJ, Fscale)
        
        return nx, ny, nz, sJ, Fscale
    
    
    def calc_spatial_h(self, verbose=0):
        '''
        Compute the spatial h from the diameter of the inscribed sphere
        '''
        VXYZ = self.mesh_elem.VXYZ
        EtoV = self.EtoV
        
        h = np.zeros(self.Nelem)  # using tetrahedron geometry
        
        self.mod.update_spatial_h(VXYZ, EtoV, h)
        h_min, h_max = h.min(), h.max()
        
        if verbose >= 1:
            hJ = np.zeros(self.Nelem)  # using J/sJ
            for eidx in range(self.Nelem):
                hJ[eidx] = self.J[eidx] / self.sJ[eidx,:].min()
            print(f"min(h) ={h_min:1.5f}, max(h) ={h_max:1.5f}") 
            print(f"min(hJ)={hJ.min():1.5f}, max(hJ)={hJ.max():1.5f}")
            
        return h_min, h_max
    
    
    def make_vmapM_vmapP_serial(self):
        '''
        vmapM corresponding to the interior node, u^−
        vmapP corresponding to the exterior node, u^+
        
        PX,PY,PZ의 크기가 (Nelem,Np)이므로, Nelem*Np 크기 배열에서의 인덱스 계산
        
        serial version
        communication indices
        '''
        Nelem = self.Nelem
        Np, Nface, Nfp = self.tet.Np, self.tet.Nface, self.tet.Nfp
        Fmask  = self.op_face.Fmask
        FmaskP  = self.op_face.FmaskP
        
        vmapM = np.zeros((Nelem, Nface, Nfp), 'i4')
        vmapP = np.zeros((Nelem, Nface, Nfp), 'i4')
        vmapF = np.zeros((Nelem, Nface), 'i4')
        
        self.mod.update_vmapM_vmapP_serial(
                Np,
                self.EtoE, self.EtoF, self.EtoR,
                Fmask, FmaskP, vmapM, vmapP, vmapF)
        
        return vmapM, vmapP, vmapF
    
    
    def make_vmapM_vmapP_parallel(self):
        '''
        parallel version
        communication indices
        '''
        Nelem = self.Nelem
        Np, Nface, Nfp = self.tet.Np, self.tet.Nface, self.tet.Nfp
        Fmask  = self.op_face.Fmask
        FmaskP  = self.op_face.FmaskP
        
        vmapM = np.zeros((Nelem, Nface, Nfp), 'i4')
        vmapP = np.zeros((Nelem, Nface, Nfp), 'i4')
        vmapF = np.zeros((Nelem, Nface), 'i4')
        
        self.mod.update_vmapM_vmapP_parallel(
                self.myrank, Np,
                self.nbrFace2bufIdxs,
                self.EtoE, self.EtoF, self.EtoR, self.EtoP,                    
                Fmask, FmaskP, vmapM, vmapP, vmapF)
        
        return vmapM, vmapP, vmapF
    
    
    def make_comm_idxs(self):        
        '''
        define indices for communications

        part2faces: {partitionTag: [(eidx,fidx,ridx), ...], ...}
        
        # 1 data per face, for 'tau' variable
        sendbuf_face_idxs: copy data from solution array to sendbuf 
        comm_face_idxs: (nbr_rank, start_index, data_count) of each communication
        
        # Nfp data per face, for 'ub' variable
        sendbuf_nfp_idxs: copy data from solution array to sendbuf
        comm_nfp_idxs: (nbr_rank, start_index, data_count) of each communication
        '''
        Nface, Np, Nfp = self.tet.Nface, self.tet.Np, self.tet.Nfp
        FmaskP  = self.op_face.FmaskP
        
        part_size = len(self.part2faces)
        face_size = sum( [len(face_list) for face_list in self.part2faces.values()] )
        
        sendbuf_face_idxs = np.zeros(face_size, 'i4')  # buf_face_size = face_size = sendbuf_face_idxs.size
        comm_face_idxs = np.zeros((part_size, 3), 'i4')        
        
        sendbuf_nfp_idxs = np.zeros((face_size, Nfp), 'i4')  # buf_nfp_size = face_size*Nfp = sendbuf_nfp_idxs.size
        comm_nfp_idxs = np.zeros((part_size, 3), 'i4')
        
        face_idx = 0
        start_face_idx = 0
        start_nfp_idx = 0
        for ci, (nbr_part, face_list) in enumerate(self.part2faces.items()):
            for eidx, fidx, ridx in face_list:
                sendbuf_face_idxs[face_idx] = eidx*Nface + fidx
                sendbuf_nfp_idxs[face_idx,:] = eidx*Np + FmaskP[fidx,ridx,:]
                face_idx += 1
                
            count_face = len(face_list)
            comm_face_idxs[ci,:] = (nbr_part-1, start_face_idx, count_face)
            start_face_idx += count_face
            
            count_nfp = len(face_list)*Nfp
            comm_nfp_idxs[ci,:] = (nbr_part-1, start_nfp_idx, count_nfp)
            start_nfp_idx += count_nfp
            
        return part_size, sendbuf_face_idxs, comm_face_idxs, sendbuf_nfp_idxs, comm_nfp_idxs
