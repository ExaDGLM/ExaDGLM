import numpy as np
import pyvista as pv
import re
from glob import glob
pv.global_theme.trame.server_proxy_enabled = True


class Plot3DTetraSingle:
    def __init__(self, dg, res='low', backend='trame', use_xvfb=True, out_fpaths=None):
        '''
        use variables in dg object
          - Nelem, N, Np, Nfp          
          - dg.mesh.PX, dg.mesh.PY, dg.mesh.PZ
          - dg.tet.xyz
          
        res: 'high' - use full quadrature points
             'low'  - use only 4 vertices on an tetrahedral element
             
        backend: jupyter bakend of pyvista
                 options - 'trame', 'server', 'client'
                 
        use_xvfb: use X virtual frame buffer
                  required when there is no monitor
        '''
        self.pv  = pv
        self.dg  = dg        
        self.res = res
        self.backend  = backend
        self.use_xvfb = use_xvfb
        self.out_fpaths = out_fpaths
        
        self.setup_pyvista()
        
        if res == 'high':
            self.pvmesh = self.make_vtk_grid_high()
        elif res == 'low':
            self.pvmesh, self.ref_vts = self.make_vtk_grid_low()
        
        
    def setup_pyvista(self):
        pv.set_jupyter_backend(self.backend)  # 'trame', 'server', 'client'
        
        if self.use_xvfb:
            pv.start_xvfb()
            
            
    def make_vtk_grid_high(self):
        '''
        make VTK unstructured grid with tetrahedron
        high resolution - use full points
        '''
        dg = self.dg
        
        # generate unstructured mesh using Delaunay triangulation in an element
        #points = np.column_stack([dg.tet.x, dg.tet.y, dg.tet.z])
        points = dg.tet.xyz
        pset = pv.PointSet(points)
        tmp_mesh = pset.delaunay_3d()
        subcell_vts = tmp_mesh.cells_dict[10]  # 10 means tetrahedral cell, shape(num_cells, 4)
        num_subcell = len(subcell_vts)
        print(f"num_subcell={num_subcell}")
        
        # generate unstructured mesh in all elements
        celltypes = np.full(dg.Nelem*num_subcell, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        cells = np.full((dg.Nelem*num_subcell, 5), fill_value=4, dtype='i4')
                
        for ei in range(dg.Nelem):
            for j, vts in enumerate(subcell_vts):
                cells[ei*num_subcell+j,1:] = ei*dg.Np + vts[:]
                
        points = np.column_stack([dg.mesh.PX.ravel(), dg.mesh.PY.ravel(), dg.mesh.PZ.ravel()])        
        pvmesh = pv.UnstructuredGrid(cells, celltypes, points)
        
        return pvmesh
        
    
    def make_vtk_grid_low(self):
        '''
        make VTK unstructured grid with tetrahedron
        low resolution - use only 4 vertices on an tetrahedral element
        '''
        dg = self.dg
        
        celltypes = np.full(dg.Nelem, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        cells = np.full((dg.Nelem, 5), fill_value=4, dtype='i4')
                
        for ei in range(dg.Nelem):
            cells[ei,1:] = ei*4 + np.arange(4)
        
        ref_vts = np.array([0, dg.N, dg.Nfp-1, dg.Np-1])
        points = np.column_stack([
            dg.mesh.PX[:,ref_vts].ravel(),
            dg.mesh.PY[:,ref_vts].ravel(),
            dg.mesh.PZ[:,ref_vts].ravel()])
        
        pvmesh = pv.UnstructuredGrid(cells, celltypes, points)
        
        return pvmesh, ref_vts
        
        
    def set_point_data(self, var, var_name):
        if self.res == 'high':
            self.pvmesh.point_data[var_name] = var[:]
            
        elif self.res == 'low':
            Nelem, Np = self.dg.Nelem, self.dg.Np
            ref_vts = self.ref_vts
            self.pvmesh.point_data[var_name] = var.reshape((Nelem, Np))[:,ref_vts].ravel()
            
        
    def plot(self, var, var_name):
        self.set_point_data(var, var_name)
                
        pl = pv.Plotter()
        #pl.add_mesh(self.pvmesh.outline(), color="k")
        #pl.add_mesh(self.pvmesh, colormap='CET_L18', opacity='sigmoid')
        #pl.camera_position = 'xz'
        #pl.camera.elevation = 20
        #pl.camera.azimuth = 20
        pl.show_axes()
        pl.show_bounds(all_edges=True, grid=True)
        pl.add_bounding_box()
        #pl.show()
        
        return pl, self.pvmesh
    
    
    def get_num_output_files(self):
        return len(self.out_fpaths)
    
    
    def read_output_file(self, seq=-1):
        fpath = self.out_fpaths[seq]
        u = np.fromfile(fpath) # Nelem*Np        
        tstep = int( re.findall(r'_(\d+)\.bin', fpath)[0] )
        
        return u, tstep
    
    
        
class Plot3DTetraPartition(Plot3DTetraSingle):
    def __init__(self, dg_list, res='low', backend='trame', use_xvfb=True, out_fpaths=None):
        '''
        used variables and arrays
          - Nelem, N, Np, Nfp 
          - dg.tet.xyz (Np,3)
          - dg.mesh.PX, dg.mesh.PY, dg.mesh.PZ (Nelem,Np)          
        '''        
        self.dg_list = dg_list
        self.global_dg = self.DGLMDummy(dg_list)
        super().__init__(self.global_dg, res, backend, use_xvfb, out_fpaths)
    
        
    class DGLMDummy:
        def __init__(self, dg_list):
            dg0 = dg_list[0]
            self.N, self.Np, self.Nfp = dg0.N, dg0.Np, dg0.Nfp
            self.tet = self.TetraDummy(dg0)
            self.mesh = self.MeshDummy(dg_list)
            self.Nelem = self.mesh.Nelem

        class TetraDummy:
            def __init__(self, dg):
                self.xyz = dg.tet.xyz

        class MeshDummy:
            def __init__(self, dg_list):
                self.PX = np.concatenate([dg.mesh.PX for dg in dg_list], axis=0)
                self.PY = np.concatenate([dg.mesh.PY for dg in dg_list], axis=0)
                self.PZ = np.concatenate([dg.mesh.PZ for dg in dg_list], axis=0)
                self.Nelem = np.sum([dg.Nelem for dg in dg_list])
            

    def get_num_output_files(self):
        num_list = []
        
        for fpaths in self.out_fpaths:
            num = len(fpaths)
            num_list.append(num)
            
        assert all(x == num_list[0] for x in num_list), \
               f"number of files are not the same {num_list}"
        
        return num_list[0]
    
    
    def read_output_file(self, seq=-1):
        u_list = []
        tstep_list = []
        
        for fpaths in self.out_fpaths:
            fpath = fpaths[seq]
            u = np.fromfile(fpath) # Nelem*Np
            tstep = int( re.findall(r'_(\d+)\.bin', fpath)[0] )
            
            u_list.append(u)
            tstep_list.append(tstep)
            
        assert all(x == tstep_list[0] for x in tstep_list), \
               f"tsteps are not the same {tstep_list}"
        
        global_u = np.concatenate(u_list)
        
        return global_u, tstep_list[0]
    
    
    
def Plot3DTetra(dg_objs, res='low', backend='trame', use_xvfb=True, file_exist=False):
    if type(dg_objs) == list:
        if file_exist:
            out_fpaths = [sorted(glob(dg.dataout_dir + "/*.bin")) for dg in dg_objs]
        else:
            out_fpaths = None
            
        plot3d = Plot3DTetraPartition(dg_objs, res, backend, use_xvfb, out_fpaths)

    else:
        dg = dg_objs
        if file_exist:
            out_fpaths = sorted(glob(dg.dataout_dir + "/*.bin"))
        else:
            out_fpaths = None

        plot3d = Plot3DTetraSingle(dg, res, backend, use_xvfb, out_fpaths)

    return plot3d
