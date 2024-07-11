#include <vector>
#include <cmath>
#include "common.h"


void calc_maxv(
        int nelem,
        int *Fmask,
        REAL *Fscale,
        REAL *d_vx, REAL *d_vy, REAL *d_vz,
        REAL *maxv, REAL *maxv_fs) {
    
    #pragma omp parallel for
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        int ei = fidx/NFACE;
        int fi = fidx%NFACE;
        REAL max_v2 = 0;
        
        for (int j=0; j<NFP; j++) {
            int idxM = ei*NP + Fmask[fi*NFP+j];
            
            REAL vx = d_vx[idxM];
            REAL vy = d_vy[idxM];
            REAL vz = d_vz[idxM];
                       
            // Compute max velocity
            REAL v2 = vx*vx + vy*vy + vz*vz;  // (velocity)^2
            if (v2 > max_v2) max_v2 = v2;
        }
        
        REAL max_v = sqrt(max_v2);
        maxv[fidx] = max_v;
        maxv_fs[fidx] = max_v*Fscale[fidx];
    }
}


REAL calc_dt(int nelem, REAL *maxv_fs) {    
    REAL max_v = 0;
    
    #pragma omp parallel for reduction(max:max_v)
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        REAL v = maxv_fs[fidx];
        
        if (v > max_v) max_v = v;
    }     
    
    REAL dt = (CFL/(2*N+1))*(1/max_v);
  
    return dt;
}


void update_tau(
        int nelem,
        int *vmapF,
        REAL *maxv,
        REAL *tau) {
    
    #pragma omp parallel for
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        int nbr_fidx = vmapF[fidx];
        
        tau[fidx] = 0.25*(maxv[fidx] + maxv[nbr_fidx])*TAU_SCALE;
    }
}
                
                
void update_ub(
        int nelem,
        int *Fmask, int *vmapP,
        REAL *u,
        REAL *ub) {
    
    #pragma omp parallel for
    for (int idx=0; idx<nelem*NFACE*NFP; idx++) {
        int ei = idx/(NFACE*NFP);
        int fi = (idx/NFP)%NFACE;
        
        int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
        int idxP = vmapP[idx];
    
        ub[idx] = 0.5*(u[idxM] + u[idxP]);
    }    
}
    

void calc_fluxLM(
        int nelem,
        REAL t,
        int *Fmask, int *EtoB,
        REAL *d_nx, REAL *d_ny, REAL *d_nz,
        REAL *px, REAL *py, REAL *pz,
        REAL *vx, REAL *vy, REAL *vz,
        REAL *tau, REAL *Fscale,
        REAL *d_ub, REAL *u,
        REAL *fluxLM) {
    
    #pragma omp parallel for
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        int bc = EtoB[fidx];
        REAL nx = d_nx[fidx];  // normal vector
        REAL ny = d_ny[fidx];
        REAL nz = d_nz[fidx];
        REAL fs = Fscale[fidx];

        for (int j=0; j<NFP; j++) {
            int idx = fidx*NFP + j;
            int ei = fidx/NFACE;
            int fi = fidx%NFACE;
            int idxM = ei*NP + Fmask[fi*NFP + j];
            REAL ub;

            if (bc == 1) {
                // Dirichlet boundary conditions
                //REAL x = px[idxM];
                //REAL y = py[idxM];
                //REAL z = pz[idxM];
                //ub = function of x,y,z,t
            }
            else {
                ub = d_ub[idx];
            }

            REAL fluxF = vx[idxM]*ub;
            REAL fluxG = vy[idxM]*ub;
            REAL fluxW = vz[idxM]*ub;
            REAL n_dot_flux = nx*fluxF + ny*fluxG + nz*fluxW;
            
            fluxLM[idx] = (n_dot_flux - tau[fidx]*(ub - u[idxM]))*fs;
        }
    }
}


void calc_fluxes(
        int nelem,
        REAL *d_rx, REAL *d_ry, REAL *d_rz,
        REAL *d_sx, REAL *d_sy, REAL *d_sz,
        REAL *d_tx, REAL *d_ty, REAL *d_tz,
        REAL *vx, REAL *vy, REAL *vz, 
        REAL *u,
        REAL *fluxR, REAL *fluxS, REAL *fluxT) {
    
    #pragma omp parallel for
    for (int ei=0; ei<nelem; ei++) {        
        REAL rx = d_rx[ei];
        REAL ry = d_ry[ei];
        REAL rz = d_rz[ei];
        REAL sx = d_sx[ei];
        REAL sy = d_sy[ei];
        REAL sz = d_sz[ei];
        REAL tx = d_tx[ei];
        REAL ty = d_ty[ei];
        REAL tz = d_tz[ei];
        
        for (int i=0; i<NP; i++) {
            int idx = ei*NP + i;
            
            REAL fluxF = vx[idx]*u[idx];
            REAL fluxG = vy[idx]*u[idx];
            REAL fluxW = vz[idx]*u[idx];
            
            fluxR[idx] = rx*fluxF + ry*fluxG + rz*fluxW;
            fluxS[idx] = sx*fluxF + sy*fluxG + sz*fluxW;
            fluxT[idx] = tx*fluxF + ty*fluxG + tz*fluxW;
        }
    }
}


void update_rhs(
        int nelem,
        REAL *WDr, REAL *WDs, REAL *WDt, REAL *LIFT,
        REAL *fluxR, REAL *fluxS, REAL *fluxT, REAL *fluxLM,
        REAL *utmp) {
    
    #pragma omp parallel for
    for (int ei=0; ei<nelem; ei++) {  
        for (int i=0; i<NP; i++) {            
            REAL rhs = 0;
                
            for (int j=0; j<NP; j++) {
                rhs += WDr[i*NP+j]*fluxR[ei*NP+j]
                     + WDs[i*NP+j]*fluxS[ei*NP+j]
                     + WDt[i*NP+j]*fluxT[ei*NP+j];
            }
            
            for (int j=0; j<NFACE*NFP; j++)
                rhs -= LIFT[i*NFACE*NFP+j]*fluxLM[ei*NFACE*NFP+j];
            
            utmp[ei*NP+i] = rhs;
        }
    }
}
