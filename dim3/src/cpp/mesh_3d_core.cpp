#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

namespace py = pybind11;


void update_PXYZ(
        py::array_t<int> &d_EtoV,
        py::array_t<double> &d_r,
        py::array_t<double> &d_s,
        py::array_t<double> &d_t,        
        py::array_t<double> &d_VX,
        py::array_t<double> &d_VY,
        py::array_t<double> &d_VZ, 
        py::array_t<double> &d_PX,
        py::array_t<double> &d_PY,
        py::array_t<double> &d_PZ) {                   

    auto EtoV = d_EtoV.unchecked<2>();
    auto r = d_r.unchecked<1>();
    auto s = d_s.unchecked<1>();
    auto t = d_t.unchecked<1>();
    auto VX = d_VX.unchecked<1>();
    auto VY = d_VY.unchecked<1>();
    auto VZ = d_VZ.unchecked<1>();
    auto PX = d_PX.mutable_unchecked<2>();
    auto PY = d_PY.mutable_unchecked<2>();
    auto PZ = d_PZ.mutable_unchecked<2>();
    
    int Nelem = d_PX.shape(0);
    int Np    = d_PX.shape(1);

    for (int ei=0; ei<Nelem; ei++) {
        int v0 = EtoV(ei, 0);
        int v1 = EtoV(ei, 1);
        int v2 = EtoV(ei, 2);
        int v3 = EtoV(ei, 3);
        
        double vx0 = VX[v0];
        double vx1 = VX[v1];
        double vx2 = VX[v2];
        double vx3 = VX[v3];
        
        double vy0 = VY[v0];
        double vy1 = VY[v1];
        double vy2 = VY[v2];
        double vy3 = VY[v3];

        double vz0 = VZ[v0];
        double vz1 = VZ[v1];
        double vz2 = VZ[v2];
        double vz3 = VZ[v3];

        for (int j=0; j<Np; j++) {
            PX(ei,j) = 0.5*(-(r[j]+s[j]+t[j]+1)*vx0 + (r[j]+1)*vx1 + (s[j]+1)*vx2 + (t[j]+1)*vx3);
            PY(ei,j) = 0.5*(-(r[j]+s[j]+t[j]+1)*vy0 + (r[j]+1)*vy1 + (s[j]+1)*vy2 + (t[j]+1)*vy3);
            PZ(ei,j) = 0.5*(-(r[j]+s[j]+t[j]+1)*vz0 + (r[j]+1)*vz1 + (s[j]+1)*vz2 + (t[j]+1)*vz3);
        }
    }
}


void update_jacobian(
        py::array_t<int> &d_EtoV,
        py::array_t<double> &d_VX,
        py::array_t<double> &d_VY,
        py::array_t<double> &d_VZ, 
        py::array_t<double> &d_J,
        py::array_t<double> &d_rx,
        py::array_t<double> &d_ry,
        py::array_t<double> &d_rz,
        py::array_t<double> &d_sx,
        py::array_t<double> &d_sy,
        py::array_t<double> &d_sz,
        py::array_t<double> &d_tx,
        py::array_t<double> &d_ty,
        py::array_t<double> &d_tz) {                   

    auto EtoV = d_EtoV.unchecked<2>();
    auto VX = d_VX.unchecked<1>();
    auto VY = d_VY.unchecked<1>();
    auto VZ = d_VZ.unchecked<1>();
    auto J  = d_J.mutable_unchecked<1>();
    auto rx = d_rx.mutable_unchecked<1>();
    auto ry = d_ry.mutable_unchecked<1>();
    auto rz = d_rz.mutable_unchecked<1>();
    auto sx = d_sx.mutable_unchecked<1>();
    auto sy = d_sy.mutable_unchecked<1>();
    auto sz = d_sz.mutable_unchecked<1>();
    auto tx = d_tx.mutable_unchecked<1>();
    auto ty = d_ty.mutable_unchecked<1>();
    auto tz = d_tz.mutable_unchecked<1>();
    
    int Nelem = d_J.shape(0);

    for (int ei=0; ei<Nelem; ei++) {
        int v0 = EtoV(ei, 0);
        int v1 = EtoV(ei, 1);
        int v2 = EtoV(ei, 2);
        int v3 = EtoV(ei, 3);
        
        double vx0 = VX[v0];
        double vx1 = VX[v1];
        double vx2 = VX[v2];
        double vx3 = VX[v3];
        
        double vy0 = VY[v0];
        double vy1 = VY[v1];
        double vy2 = VY[v2];
        double vy3 = VY[v3];

        double vz0 = VZ[v0];
        double vz1 = VZ[v1];
        double vz2 = VZ[v2];
        double vz3 = VZ[v3];
        
        double xr = 0.5*(vx1 - vx0);
        double xs = 0.5*(vx2 - vx0);
        double xt = 0.5*(vx3 - vx0);           
        double yr = 0.5*(vy1 - vy0);          
        double ys = 0.5*(vy2 - vy0);
        double yt = 0.5*(vy3 - vy0);
        double zr = 0.5*(vz1 - vz0);
        double zs = 0.5*(vz2 - vz0);
        double zt = 0.5*(vz3 - vz0);

        double jac = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt);
        J(ei) = jac;

        rx(ei) =  (ys*zt - zs*yt)/jac;
        ry(ei) = -(xs*zt - zs*xt)/jac;
        rz(ei) =  (xs*yt - ys*xt)/jac;

        sx(ei) = -(yr*zt - zr*yt)/jac;
        sy(ei) =  (xr*zt - zr*xt)/jac;
        sz(ei) = -(xr*yt - yr*xt)/jac;

        tx(ei) =  (yr*zs - zr*ys)/jac;
        ty(ei) = -(xr*zs - zr*xs)/jac;
        tz(ei) =  (xr*ys - yr*xs)/jac;
    }
}


std::vector<double> matmul(
        int ei,
        py::array_t<double> &d_D,
        py::array_t<double> &d_P) {
    
    auto D = d_D.unchecked<2>();
    auto P = d_P.unchecked<2>();    
    int Np = d_D.shape(0);
    
    std::vector<double> v(Np);
    
    for (int i=0; i<Np; i++) {
        double jsum=0;

        for (int j=0; j<Np; j++)
            jsum += D(i,j)*P(ei,j);

        v[i] = jsum;
    }
    
    return v;
}


void update_jacobian_np(
        py::array_t<double> &d_Dr,
        py::array_t<double> &d_Ds,
        py::array_t<double> &d_Dt,
        py::array_t<double> &d_PX,
        py::array_t<double> &d_PY,
        py::array_t<double> &d_PZ, 
        py::array_t<double> &d_J,
        py::array_t<double> &d_rx,
        py::array_t<double> &d_ry,
        py::array_t<double> &d_rz,
        py::array_t<double> &d_sx,
        py::array_t<double> &d_sy,
        py::array_t<double> &d_sz,
        py::array_t<double> &d_tx,
        py::array_t<double> &d_ty,
        py::array_t<double> &d_tz) {                   

    auto J  = d_J.mutable_unchecked<2>();
    auto rx = d_rx.mutable_unchecked<2>();
    auto ry = d_ry.mutable_unchecked<2>();
    auto rz = d_rz.mutable_unchecked<2>();
    auto sx = d_sx.mutable_unchecked<2>();
    auto sy = d_sy.mutable_unchecked<2>();
    auto sz = d_sz.mutable_unchecked<2>();
    auto tx = d_tx.mutable_unchecked<2>();
    auto ty = d_ty.mutable_unchecked<2>();
    auto tz = d_tz.mutable_unchecked<2>();
    
    int Nelem = d_PX.shape(0);
    int Np    = d_PY.shape(1);
    
    for (int ei=0; ei<Nelem; ei++) {
        auto xr = matmul(ei, d_Dr, d_PX);
        auto xs = matmul(ei, d_Ds, d_PX);
        auto xt = matmul(ei, d_Dt, d_PX);
        auto yr = matmul(ei, d_Dr, d_PY);
        auto ys = matmul(ei, d_Ds, d_PY);
        auto yt = matmul(ei, d_Dt, d_PY);
        auto zr = matmul(ei, d_Dr, d_PZ);
        auto zs = matmul(ei, d_Ds, d_PZ);
        auto zt = matmul(ei, d_Dt, d_PZ);
        
        for (int j=0; j<Np; j++) {
            double jac = xr[j]*(ys[j]*zt[j] - zs[j]*yt[j])
                       - yr[j]*(xs[j]*zt[j] - zs[j]*xt[j])
                       + zr[j]*(xs[j]*yt[j] - ys[j]*xt[j]);
            J(ei,j) = jac;

            rx(ei,j) =  (ys[j]*zt[j] - zs[j]*yt[j])/jac;
            ry(ei,j) = -(xs[j]*zt[j] - zs[j]*xt[j])/jac;
            rz(ei,j) =  (xs[j]*yt[j] - ys[j]*xt[j])/jac;

            sx(ei,j) = -(yr[j]*zt[j] - zr[j]*yt[j])/jac;
            sy(ei,j) =  (xr[j]*zt[j] - zr[j]*xt[j])/jac;
            sz(ei,j) = -(xr[j]*yt[j] - yr[j]*xt[j])/jac;

            tx(ei,j) =  (yr[j]*zs[j] - zr[j]*ys[j])/jac;
            ty(ei,j) = -(xr[j]*zs[j] - zr[j]*xs[j])/jac;
            tz(ei,j) =  (xr[j]*ys[j] - yr[j]*xs[j])/jac;
        }
    }
}


void update_normal_vectors(
        py::array_t<double> &d_J,
        py::array_t<double> &d_rx,
        py::array_t<double> &d_ry,
        py::array_t<double> &d_rz,
        py::array_t<double> &d_sx,
        py::array_t<double> &d_sy,
        py::array_t<double> &d_sz,
        py::array_t<double> &d_tx,
        py::array_t<double> &d_ty,
        py::array_t<double> &d_tz,
        py::array_t<double> &d_nx,
        py::array_t<double> &d_ny,
        py::array_t<double> &d_nz,
        py::array_t<double> &d_sJ,
        py::array_t<double> &d_Fscale) {
    
    auto J = d_J.unchecked<1>();
    auto rx = d_rx.unchecked<1>();
    auto ry = d_ry.unchecked<1>();
    auto rz = d_rz.unchecked<1>();
    auto sx = d_sx.unchecked<1>();
    auto sy = d_sy.unchecked<1>();
    auto sz = d_sz.unchecked<1>();
    auto tx = d_tx.unchecked<1>();
    auto ty = d_ty.unchecked<1>();
    auto tz = d_tz.unchecked<1>();    
    auto nx = d_nx.mutable_unchecked<2>();
    auto ny = d_ny.mutable_unchecked<2>();
    auto nz = d_nz.mutable_unchecked<2>();
    auto sJ = d_sJ.mutable_unchecked<2>();
    auto Fscale = d_Fscale.mutable_unchecked<2>();

    int Nelem = sJ.shape(0);
    int Nface = sJ.shape(1);
    
    for (int i=0; i<Nelem; i++) {
        // face 0
        nx(i,0) = -tx(i);
        ny(i,0) = -ty(i);
        nz(i,0) = -tz(i);

        // face 1
        nx(i,1) = -sx(i);
        ny(i,1) = -sy(i);
        nz(i,1) = -sz(i);

        // face 2
        nx(i,2) = rx(i) + sx(i) + tx(i);
        ny(i,2) = ry(i) + sy(i) + ty(i);
        nz(i,2) = rz(i) + sz(i) + tz(i);

        // face 4
        nx(i,3) = -rx(i);
        ny(i,3) = -ry(i);
        nz(i,3) = -rz(i);

        // normalize
        for (int j=0; j<Nface; j++) {
            double magnitude = std::sqrt(nx(i,j)*nx(i,j) + ny(i,j)*ny(i,j) + nz(i,j)*nz(i,j));
            sJ(i,j) = magnitude;
            nx(i,j) /= magnitude;
            ny(i,j) /= magnitude;
            nz(i,j) /= magnitude;

            sJ(i,j) *= J(i);
            Fscale(i,j) = sJ(i,j)/J(i);
        }
    }
}


void update_vmapM_vmapP_serial(
        int Np,
        py::array_t<int> &d_EtoE,
        py::array_t<int> &d_EtoF,
        py::array_t<int> &d_EtoR,
        py::array_t<int> &d_Fmask,
        py::array_t<int> &d_FmaskP,
        py::array_t<int> &d_vmapM,
        py::array_t<int> &d_vmapP,
        py::array_t<int> &d_vmapF) {    
    
    auto EtoE = d_EtoE.unchecked<2>();
    auto EtoF = d_EtoF.unchecked<2>();
    auto EtoR = d_EtoR.unchecked<2>();
    auto Fmask  = d_Fmask.unchecked<2>();
    auto FmaskP = d_FmaskP.unchecked<3>();
    auto vmapM = d_vmapM.mutable_unchecked<3>();
    auto vmapP = d_vmapP.mutable_unchecked<3>();
    auto vmapF = d_vmapF.mutable_unchecked<2>();
    
    int Nelem = d_vmapM.shape(0);
    int Nface = d_vmapM.shape(1);
    int Nfp   = d_vmapM.shape(2);

    for (int ei=0; ei<Nelem; ei++) {
        for (int fi=0; fi<Nface; fi++) {
            int nbr_ei = EtoE(ei,fi);
            int nbr_fi = EtoF(ei,fi);
            int rot    = EtoR(ei,fi);
            
            for (int j=0; j<Nfp; j++)
                vmapM(ei,fi,j) = ei*Np + Fmask(fi,j);
            
            if (nbr_ei < 0) {  // boundary condition
                vmapF(ei,fi) = ei*Nface + fi;
            
                for (int j=0; j<Nfp; j++)
                    vmapP(ei,fi,j) = vmapM(ei,fi,j);
            }
            else {
                vmapF(ei,fi) = nbr_ei*Nface + nbr_fi;
                
                for (int j=0; j<Nfp; j++)
                    vmapP(ei,fi,j) = nbr_ei*Np + FmaskP(nbr_fi,rot,j);
            }
        }
    }
}


void update_vmapM_vmapP_parallel(
        int myrank,
        int Np,
        py::dict &nbrFace2bufIdxs,
        py::array_t<int> &d_EtoE,
        py::array_t<int> &d_EtoF,
        py::array_t<int> &d_EtoR,
        py::array_t<int> &d_EtoP,
        py::array_t<int> &d_Fmask,
        py::array_t<int> &d_FmaskP,
        py::array_t<int> &d_vmapM,
        py::array_t<int> &d_vmapP,
        py::array_t<int> &d_vmapF) {    
    
    auto EtoE = d_EtoE.unchecked<2>();
    auto EtoF = d_EtoF.unchecked<2>();
    auto EtoR = d_EtoR.unchecked<2>();
    auto EtoP = d_EtoP.unchecked<2>();
    auto Fmask  = d_Fmask.unchecked<2>();
    auto FmaskP = d_FmaskP.unchecked<3>();
    auto vmapM = d_vmapM.mutable_unchecked<3>();
    auto vmapP = d_vmapP.mutable_unchecked<3>();
    auto vmapF = d_vmapF.mutable_unchecked<2>();
    
    int Nelem = d_vmapM.shape(0);
    int Nface = d_vmapM.shape(1);
    int Nfp   = d_vmapM.shape(2);

    for (int ei=0; ei<Nelem; ei++) {
        for (int fi=0; fi<Nface; fi++) {
            int nbr_ei   = EtoE(ei,fi);
            int nbr_fi   = EtoF(ei,fi);
            int rot      = EtoR(ei,fi);
            int nbr_part = EtoP(ei,fi);            
            
            for (int j=0; j<Nfp; j++)
                vmapM(ei,fi,j) = ei*Np + Fmask(fi,j);
                
            if (nbr_ei < 0) { // boundary condition
                vmapF(ei,fi) = ei*Nface + fi;
                             
                for (int j=0; j<Nfp; j++)
                    vmapP(ei,fi,j) = vmapM(ei,fi,j);
            }
            else if (nbr_part == myrank + 1) {
                vmapF(ei,fi) = nbr_ei*Nface + nbr_fi;
                
                for (int j=0; j<Nfp; j++)
                    vmapP(ei,fi,j) = nbr_ei*Np + FmaskP(nbr_fi,rot,j);
            }
            else {
                auto key = py::make_tuple(nbr_part, nbr_ei, nbr_fi);
                int buf_idx = nbrFace2bufIdxs[key].cast<int>();
                
                vmapF(ei,fi) = Nelem*Nface + buf_idx;
                
                for (int j=0; j<Nfp; j++)
                    vmapP(ei,fi,j) = Nelem*Np + buf_idx*Nfp + j;
            }
        }
    }
}


double tri_area(
        std::vector<double> &xyz0,
        std::vector<double> &xyz1,
        std::vector<double> &xyz2) {
    
    // area of a triangle
    double ux = xyz1[0] - xyz0[0];
    double uy = xyz1[1] - xyz0[1];
    double uz = xyz1[2] - xyz0[2];
    double vx = xyz2[0] - xyz0[0];
    double vy = xyz2[1] - xyz0[1];
    double vz = xyz2[2] - xyz0[2];
        
    double cx = uy*vz - uz*vy;  // cross product
    double cy = ux*vz - uz*vx;
    double cz = ux*vy - uy*vx;
    double area = 0.5*sqrt(cx*cx + cy*cy + cz*cz);
    return area;
}

    
double calc_diameter(
        std::vector<double> &xyz0,
        std::vector<double> &xyz1,
        std::vector<double> &xyz2,
        std::vector<double> &xyz3) {
            
    // diameter of the inscribed sphere in a tetrahedron
    double s1 = xyz0[0]*(xyz1[1]*(xyz2[2] - xyz3[2]) - xyz1[2]*(xyz2[1] - xyz3[1]) + (xyz2[1]*xyz3[2] - xyz2[2]*xyz3[1]));
    double s2 = xyz0[1]*(xyz1[0]*(xyz2[2] - xyz3[2]) - xyz1[2]*(xyz2[0] - xyz3[0]) + (xyz2[0]*xyz3[2] - xyz2[2]*xyz3[0]));
    double s3 = xyz0[2]*(xyz1[0]*(xyz2[1] - xyz3[1]) - xyz1[1]*(xyz2[0] - xyz3[0]) + (xyz2[0]*xyz3[1] - xyz2[1]*xyz3[0]));
    double s4 = xyz1[0]*(xyz2[1]*xyz3[2] - xyz3[1]*xyz2[2]) - xyz1[1]*(xyz2[0]*xyz3[2] - xyz2[2]*xyz3[0]) + xyz1[2]*(xyz2[0]*xyz3[1] - xyz2[1]*xyz3[0]);
    double det = s1 - s2 + s3 - s4;
    double V = abs(det)/6;  // volume of the tetrahedron

    double area1 = tri_area(xyz1, xyz0, xyz2);  // area of a triangle face
    double area2 = tri_area(xyz0, xyz1, xyz3);
    double area3 = tri_area(xyz1, xyz2, xyz3);
    double area4 = tri_area(xyz2, xyz0, xyz3);

    double d = 6*V/(area1 + area2 + area3 + area4);
    return d;
}


void update_spatial_h(
        py::array_t<double> &d_VXYZ,
        py::array_t<int> &d_EtoV,
        py::array_t<double> &d_h) {
    
    auto VXYZ = d_VXYZ.unchecked<2>();
    auto EtoV = d_EtoV.unchecked<2>();
    auto h = d_h.mutable_unchecked<1>();
    
    int Nelem = d_EtoV.shape(0);
    
    std::vector<double> xyz0(3);
    std::vector<double> xyz1(3);
    std::vector<double> xyz2(3);
    std::vector<double> xyz3(3);
    
    for (int ei=0; ei<Nelem; ei++) {
        double v0 = EtoV(ei,0);
        double v1 = EtoV(ei,1);
        double v2 = EtoV(ei,2);
        double v3 = EtoV(ei,3);
        
        xyz0[0] = VXYZ(v0,0);
        xyz0[1] = VXYZ(v0,1);
        xyz0[2] = VXYZ(v0,2);
        xyz1[0] = VXYZ(v1,0);
        xyz1[1] = VXYZ(v1,1);
        xyz1[2] = VXYZ(v1,2);
        xyz2[0] = VXYZ(v2,0);
        xyz2[1] = VXYZ(v2,1);
        xyz2[2] = VXYZ(v2,2);
        xyz3[0] = VXYZ(v3,0);
        xyz3[1] = VXYZ(v3,1);
        xyz3[2] = VXYZ(v3,2);
        
        h(ei) = calc_diameter(xyz0, xyz1, xyz2, xyz3);
    }
}


PYBIND11_MODULE(mesh_3d_cpp, mod) {
    mod.def("update_PXYZ", &update_PXYZ, "");
    mod.def("update_jacobian", &update_jacobian, "");
    mod.def("update_jacobian_np", &update_jacobian_np, "");
    mod.def("update_normal_vectors", &update_normal_vectors, "");
    mod.def("update_vmapM_vmapP_serial", &update_vmapM_vmapP_serial, "");
    mod.def("update_vmapM_vmapP_parallel", &update_vmapM_vmapP_parallel, "");
    mod.def("update_spatial_h", &update_spatial_h, "");
}
