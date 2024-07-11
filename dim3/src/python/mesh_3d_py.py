import numpy as np


def update_PXYZ(EtoV, r, s, t, VX, VY, VZ, PX, PY, PZ):
    '''
    Make inner points in each element
    '''
    for ei, vertices in enumerate(EtoV):
        vx0, vx1, vx2, vx3 = VX[vertices]
        vy0, vy1, vy2, vy3 = VY[vertices]
        vz0, vz1, vz2, vz3 = VZ[vertices]

        PX[ei,:] = 0.5*(-(r+s+t+1)*vx0 + (r+1)*vx1 + (s+1)*vx2 + (t+1)*vx3)
        PY[ei,:] = 0.5*(-(r+s+t+1)*vy0 + (r+1)*vy1 + (s+1)*vy2 + (t+1)*vy3)
        PZ[ei,:] = 0.5*(-(r+s+t+1)*vz0 + (r+1)*vz1 + (s+1)*vz2 + (t+1)*vz3)
        
        
def update_jacobian(EtoV, VX, VY, VZ, J, rx, ry, rz, sx, sy, sz, tx, ty, tz):
    for ei, vertices in enumerate(EtoV):
        vx0, vx1, vx2, vx3 = VX[vertices]
        vy0, vy1, vy2, vy3 = VY[vertices]
        vz0, vz1, vz2, vz3 = VZ[vertices]

        xr = 0.5*(vx1 - vx0)
        xs = 0.5*(vx2 - vx0)
        xt = 0.5*(vx3 - vx0)            
        yr = 0.5*(vy1 - vy0)            
        ys = 0.5*(vy2 - vy0)
        yt = 0.5*(vy3 - vy0)
        zr = 0.5*(vz1 - vz0)            
        zs = 0.5*(vz2 - vz0)
        zt = 0.5*(vz3 - vz0)

        jac = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
        J[ei] = jac

        rx[ei] =  (ys*zt - zs*yt)/jac
        ry[ei] = -(xs*zt - zs*xt)/jac
        rz[ei] =  (xs*yt - ys*xt)/jac

        sx[ei] = -(yr*zt - zr*yt)/jac
        sy[ei] =  (xr*zt - zr*xt)/jac
        sz[ei] = -(xr*yt - yr*xt)/jac

        tx[ei] =  (yr*zs - zr*ys)/jac
        ty[ei] = -(xr*zs - zr*xs)/jac
        tz[ei] =  (xr*ys - yr*xs)/jac
        
        
def update_jacobian_np(Dr, Ds, Dt, PX, PY, PZ, J, rx, ry, rz, sx, sy, sz, tx, ty, tz):
    for ei, (x, y, z) in enumerate(zip(PX, PY, PZ)):                        
        xr = Dr@x
        xs = Ds@x
        xt = Dt@x            
        yr = Dr@y
        ys = Ds@y
        yt = Dt@y
        zr = Dr@z
        zs = Ds@z
        zt = Dt@z

        jac = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
        J[ei,:] = jac

        rx[ei,:] =  (ys*zt - zs*yt)/jac
        ry[ei,:] = -(xs*zt - zs*xt)/jac
        rz[ei,:] =  (xs*yt - ys*xt)/jac

        sx[ei,:] = -(yr*zt - zr*yt)/jac
        sy[ei,:] =  (xr*zt - zr*xt)/jac
        sz[ei,:] = -(xr*yt - yr*xt)/jac

        tx[ei,:] =  (yr*zs - zr*ys)/jac
        ty[ei,:] = -(xr*zs - zr*xs)/jac
        tz[ei,:] =  (xr*ys - yr*xs)/jac
        

def update_normal_vectors(J, rx, ry, rz, sx, sy, sz, tx, ty, tz, nx, ny, nz, sJ, Fscale):
    # face 0
    nx[:,0] = -tx[:]
    ny[:,0] = -ty[:]
    nz[:,0] = -tz[:]

    # face 1
    nx[:,1] = -sx[:]
    ny[:,1] = -sy[:]
    nz[:,1] = -sz[:]

    # face 2
    nx[:,2] = rx[:] + sx[:] + tx[:]
    ny[:,2] = ry[:] + sy[:] + ty[:]
    nz[:,2] = rz[:] + sz[:] + tz[:]

    # face 4
    nx[:,3] = -rx[:]
    ny[:,3] = -ry[:]
    nz[:,3] = -rz[:]

    # normalize
    # face surface에서의 jacobian은 각 face에서 normal vector의 크기?
    sJ[:,:] = np.sqrt(nx*nx + ny*ny + nz*nz)  # face
    nx[:,:] = nx/sJ
    ny[:,:] = ny/sJ
    nz[:,:] = nz/sJ        

    # Fscale at faces
    sJ[:,:] = sJ[:,:]*J[:,np.newaxis]
    Fscale[:,:] = sJ[:,:]/J[:,np.newaxis]
    
    
def update_vmapM_vmapP_serial(Np, EtoE, EtoF, EtoR, Fmask, FmaskP, vmapM, vmapP, vmapF):
    Nelem, Nface, Nfp = vmapM.shape

    for ei in range(Nelem):
        vmapM[ei,:,:] = ei*Np + Fmask[:,:]

        for fi in range(Nface):                
            nbr_ei = EtoE[ei,fi]
            nbr_fi = EtoF[ei,fi]
            rot    = EtoR[ei,fi]

            if nbr_ei < 0:  # boundary condition
                vmapP[ei,fi,:] = vmapM[ei,fi,:]
                vmapF[ei,fi]   = ei*Nface + fi
            else:
                vmapP[ei,fi,:] = nbr_ei*Np + FmaskP[nbr_fi,rot,:]
                vmapF[ei,fi]   = nbr_ei*Nface + nbr_fi
                
                
def update_vmapM_vmapP_parallel(
        myrank, Np,
        nbrFace2bufIdxs,
        EtoE, EtoF, EtoR, EtoP,        
        Fmask, FmaskP, vmapM, vmapP, vmapF):
    
    '''
    define new vmapP index for faces adjacent partitions
    
    nbrFace2bufIdxs: {(partitionTag, elemTag, fidx):buf_idx,...}
    '''
    Nelem, Nface, Nfp = vmapM.shape

    for ei in range(Nelem):
        vmapM[ei,:,:] = ei*Np + Fmask[:,:]

        for fi in range(Nface):
            nbr_ei   = EtoE[ei,fi]
            nbr_fi   = EtoF[ei,fi]
            rot      = EtoR[ei,fi]
            nbr_part = EtoP[ei,fi]

            if nbr_ei < 0:  # boundary condition EtoB[ei,fi]
                vmapP[ei,fi,:] = vmapM[ei,fi,:]
                vmapF[ei,fi]   = ei*Nface + fi
                
            elif nbr_part == myrank + 1:  # myself
                vmapP[ei,fi,:] = nbr_ei*Np + FmaskP[nbr_fi,rot,:]
                vmapF[ei,fi]   = nbr_ei*Nface + nbr_fi
                
            else:
                buf_idx = nbrFace2bufIdxs[(nbr_part, nbr_ei, nbr_fi)]                
                vmapP[ei,fi,:] = Nelem*Np + buf_idx*Nfp + np.arange(Nfp)
                vmapF[ei,fi]   = Nelem*Nface + buf_idx
                
                
def update_spatial_h(VXYZ, EtoV, h):
    '''
    calculate h as diameter of the inscribed sphere in a tetrahedron
    '''
    def tri_area(xyz0, xyz1, xyz2):
        '''
        area of a triangle
        '''
        u = xyz1[:] - xyz0[:]
        v = xyz2[:] - xyz0[:]        
        cross = (u[1]*v[2] - u[2]*v[1], u[0]*v[2] - u[2]*v[0], u[0]*v[1] - u[1]*v[0])
        area = 0.5*np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        return area

    def calc_diameter(xyz0, xyz1, xyz2, xyz3):
        '''
        4x4 determinant
        ref: https://github.com/Shashank02051997/C-Programming/blob/master/Determinent%20of%202X2%20matrix%20and%203X3%20matrix%20and%204X4%20matrix.CPP
        '''
        s1 = xyz0[0]*(xyz1[1]*(xyz2[2] - xyz3[2]) - xyz1[2]*(xyz2[1] - xyz3[1]) + (xyz2[1]*xyz3[2] - xyz2[2]*xyz3[1]))
        s2 = xyz0[1]*(xyz1[0]*(xyz2[2] - xyz3[2]) - xyz1[2]*(xyz2[0] - xyz3[0]) + (xyz2[0]*xyz3[2] - xyz2[2]*xyz3[0]))
        s3 = xyz0[2]*(xyz1[0]*(xyz2[1] - xyz3[1]) - xyz1[1]*(xyz2[0] - xyz3[0]) + (xyz2[0]*xyz3[1] - xyz2[1]*xyz3[0]))
        s4 = xyz1[0]*(xyz2[1]*xyz3[2] - xyz3[1]*xyz2[2]) - xyz1[1]*(xyz2[0]*xyz3[2] - xyz2[2]*xyz3[0]) + xyz1[2]*(xyz2[0]*xyz3[1] - xyz2[1]*xyz3[0])
        det = s1 - s2 + s3 - s4
        V = abs(det)/6  # volume of the tetrahedron

        area1 = tri_area(xyz1, xyz0, xyz2)  # area of a triangle face
        area2 = tri_area(xyz0, xyz1, xyz3)
        area3 = tri_area(xyz1, xyz2, xyz3)
        area4 = tri_area(xyz2, xyz0, xyz3)

        d = 6*V/(area1 + area2 + area3 + area4)
        return d
    
    for ei, (v0,v1,v2,v3) in enumerate(EtoV):
        h[ei] = calc_diameter(VXYZ[v0,:], VXYZ[v1,:], VXYZ[v2,:], VXYZ[v3,:])
