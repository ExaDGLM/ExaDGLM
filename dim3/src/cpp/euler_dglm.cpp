#include <vector>
#include <cmath>
#include "common.h"


void calc_maxv(
        int nelem,
        int *Fmask,
        REAL *Fscale,
        vector<REAL> *p_u,
        REAL *maxv, REAL *maxv_fs) {
    
    #pragma omp parallel for
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        int ei = fidx/NFACE;
        int fi = fidx%NFACE;
        REAL max_v2 = 0;
        REAL max_c2 = 0;
        REAL min_u1 = 1e10;
        
        for (int j=0; j<NFP; j++) {
            int idxM = ei*NP + Fmask[fi*NFP+j];
            
            REAL u1 = p_u[0][idxM];
            REAL u2 = p_u[1][idxM];
            REAL u3 = p_u[2][idxM];
            REAL u4 = p_u[3][idxM];
            REAL u5 = p_u[4][idxM];
            
            REAL vx = u2/u1;
            REAL vy = u3/u1;
            REAL vz = u4/u1;
                       
            // Compute max velocity
            REAL v2 = vx*vx + vy*vy + vz*vz;  // (velocity)^2
            if (v2 > max_v2) max_v2 = v2;  
            
            // Compute max sound speed
            REAL pres = (GAMMA - 1)*(u5 - 0.5*u1*v2);  // pressure
            REAL sound_c2 = abs(GAMMA*pres / u1);      // (sound speed)^2
            if (sound_c2 > max_c2) max_c2 = sound_c2;
            
            if (u1 < min_u1) min_u1 = u1;
        }
        
        REAL max_v = sqrt(max_v2) + sqrt(max_c2);
        maxv[fidx] = max_v;
        maxv_fs[fidx] = max_v*Fscale[fidx];
    }
}


REAL calc_dt(int nelem, REAL *maxv_fs) {    
    REAL max_v = 0;
    //int max_idx;
    
    #pragma omp parallel for reduction(max:max_v)
    for (int fidx=0; fidx<nelem*NFACE; fidx++) {
        REAL v = maxv_fs[fidx];
        
        if (v > max_v) {
            max_v = v;
        }
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
        vector<REAL> *p_u,
        vector<REAL> *p_ub) {
    
    #pragma omp parallel for
    for (int idx=0; idx<nelem*NFACE*NFP; idx++) {
        int ei = idx/(NFACE*NFP);
        int fi = (idx/NFP)%NFACE;
        
        int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
        int idxP = vmapP[idx];
    
        p_ub[0][idx] = 0.5*(p_u[0][idxM] + p_u[0][idxP]);
        p_ub[1][idx] = 0.5*(p_u[1][idxM] + p_u[1][idxP]);
        p_ub[2][idx] = 0.5*(p_u[2][idxM] + p_u[2][idxP]);
        p_ub[3][idx] = 0.5*(p_u[3][idxM] + p_u[3][idxP]);
        p_ub[4][idx] = 0.5*(p_u[4][idxM] + p_u[4][idxP]);
    }    
}
    

void calc_fluxLM(
        int nelem,
        REAL t,
        int *Fmask, int *EtoB,
        REAL *d_nx, REAL *d_ny, REAL *d_nz,
        REAL *px, REAL *py, REAL *pz, 
        REAL *tau, REAL *Fscale,
        vector<REAL> *p_ub,
        vector<REAL> *p_u,
        vector<REAL> *p_fluxLM) {
    
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
            REAL ub1, ub2, ub3, ub4, ub5, vb2;

            if (bc == 1) {
                // Dirichlet boundary conditions
                //REAL x = px[idxM];
                //REAL y = py[idxM];
                //REAL z = pz[idxM];
                //ub = function of x,y,z,t
            }
            else {
                ub1 = p_ub[0][idx];
                ub2 = p_ub[1][idx];
                ub3 = p_ub[2][idx];
                ub4 = p_ub[3][idx];
                ub5 = p_ub[4][idx];
                vb2 = ub2*ub2 + ub3*ub3 + ub4*ub4;
            }

            REAL fluxF1 = ub2;
            REAL fluxF2 = (ub2*ub2)/ub1 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1);
            REAL fluxF3 = (ub2*ub3)/ub1;
            REAL fluxF4 = (ub2*ub4)/ub1;
            REAL fluxF5 = (ub5 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1))*(ub2/ub1);
        
            REAL fluxG1 = ub3;
            REAL fluxG2 = ub2*ub3/ub1;
            REAL fluxG3 = (ub3*ub3)/ub1 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1);
            REAL fluxG4 = ub3*ub4/ub1;
            REAL fluxG5 = (ub5 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1))*(ub3/ub1);
                  
            REAL fluxW1 = ub4;
            REAL fluxW2 = ub2*ub4/ub1;
            REAL fluxW3 = ub3*ub4/ub1;
            REAL fluxW4 = (ub4*ub4)/ub1 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1);
            REAL fluxW5 = (ub5 + (GAMMA-1)*(ub5 - 0.5*vb2/ub1))*(ub4/ub1);
              
            REAL n_dot_flux1 = nx*fluxF1 + ny*fluxG1 + nz*fluxW1;
            REAL n_dot_flux2 = nx*fluxF2 + ny*fluxG2 + nz*fluxW2;
            REAL n_dot_flux3 = nx*fluxF3 + ny*fluxG3 + nz*fluxW3;
            REAL n_dot_flux4 = nx*fluxF4 + ny*fluxG4 + nz*fluxW4;
            REAL n_dot_flux5 = nx*fluxF5 + ny*fluxG5 + nz*fluxW5;
            
            p_fluxLM[0][idx] = (n_dot_flux1 - tau[fidx]*(ub1 - p_u[0][idxM]))*fs;
            p_fluxLM[1][idx] = (n_dot_flux2 - tau[fidx]*(ub2 - p_u[1][idxM]))*fs;
            p_fluxLM[2][idx] = (n_dot_flux3 - tau[fidx]*(ub3 - p_u[2][idxM]))*fs;
            p_fluxLM[3][idx] = (n_dot_flux4 - tau[fidx]*(ub4 - p_u[3][idxM]))*fs;
            p_fluxLM[4][idx] = (n_dot_flux5 - tau[fidx]*(ub5 - p_u[4][idxM]))*fs;
        }
    }
}


void calc_fluxes(
        int nelem,
        REAL *d_rx, REAL *d_ry, REAL *d_rz,
        REAL *d_sx, REAL *d_sy, REAL *d_sz,
        REAL *d_tx, REAL *d_ty, REAL *d_tz,
        vector<REAL> *p_u,
        vector<REAL> *p_fluxR, vector<REAL> *p_fluxS, vector<REAL> *p_fluxT) {
    
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
            
            REAL u1 = p_u[0][idx];
            REAL u2 = p_u[1][idx];
            REAL u3 = p_u[2][idx];
            REAL u4 = p_u[3][idx];
            REAL u5 = p_u[4][idx];
            REAL v2 = u2*u2 + u3*u3 + u4*u4;
            
            REAL fluxF1 = u2;
            REAL fluxF2 = (u2*u2)/u1 + (GAMMA-1)*(u5 - 0.5*v2/u1);
            REAL fluxF3 = (u2*u3)/u1;
            REAL fluxF4 = (u2*u4)/u1;
            REAL fluxF5 = (u5 + (GAMMA-1)*(u5 - 0.5*v2/u1))*(u2/u1);
        
            REAL fluxG1 = u3;
            REAL fluxG2 = u2*u3/u1;
            REAL fluxG3 = (u3*u3)/u1 + (GAMMA-1)*(u5 - 0.5*v2/u1);
            REAL fluxG4 = u3*u4/u1;
            REAL fluxG5 = (u5 + (GAMMA-1)*(u5 - 0.5*v2/u1))*(u3/u1);
                  
            REAL fluxW1 = u4;
            REAL fluxW2 = u2*u4/u1;
            REAL fluxW3 = u3*u4/u1;
            REAL fluxW4 = (u4*u4)/u1 + (GAMMA-1)*(u5 - 0.5*v2/u1);
            REAL fluxW5 = (u5 + (GAMMA-1)*(u5 - 0.5*v2/u1))*(u4/u1);
            
            p_fluxR[0][idx] = rx*fluxF1 + ry*fluxG1 + rz*fluxW1;
            p_fluxR[1][idx] = rx*fluxF2 + ry*fluxG2 + rz*fluxW2;
            p_fluxR[2][idx] = rx*fluxF3 + ry*fluxG3 + rz*fluxW3;
            p_fluxR[3][idx] = rx*fluxF4 + ry*fluxG4 + rz*fluxW4;
            p_fluxR[4][idx] = rx*fluxF5 + ry*fluxG5 + rz*fluxW5;
            p_fluxS[0][idx] = sx*fluxF1 + sy*fluxG1 + sz*fluxW1;
            p_fluxS[1][idx] = sx*fluxF2 + sy*fluxG2 + sz*fluxW2;
            p_fluxS[2][idx] = sx*fluxF3 + sy*fluxG3 + sz*fluxW3;
            p_fluxS[3][idx] = sx*fluxF4 + sy*fluxG4 + sz*fluxW4;
            p_fluxS[4][idx] = sx*fluxF5 + sy*fluxG5 + sz*fluxW5;
            p_fluxT[0][idx] = tx*fluxF1 + ty*fluxG1 + tz*fluxW1;
            p_fluxT[1][idx] = tx*fluxF2 + ty*fluxG2 + tz*fluxW2;
            p_fluxT[2][idx] = tx*fluxF3 + ty*fluxG3 + tz*fluxW3;
            p_fluxT[3][idx] = tx*fluxF4 + ty*fluxG4 + tz*fluxW4;
            p_fluxT[4][idx] = tx*fluxF5 + ty*fluxG5 + tz*fluxW5;
        }
    }
}


void update_rhs(
        int nelem,
        REAL *WDr, REAL *WDs, REAL *WDt, REAL *LIFT,
        vector<REAL> *p_fluxR, vector<REAL> *p_fluxS, vector<REAL> *p_fluxT, vector<REAL> *p_fluxLM,
        vector<REAL> *p_utmp) {
    
    #pragma omp parallel for
    for (int ei=0; ei<nelem; ei++) {  
        for (int i=0; i<NP; i++) {            
            REAL rhs1 = 0;
            REAL rhs2 = 0;
            REAL rhs3 = 0;
            REAL rhs4 = 0;
            REAL rhs5 = 0;
                
            for (int j=0; j<NP; j++) {
                rhs1 += WDr[i*NP+j]*p_fluxR[0][ei*NP+j]
                      + WDs[i*NP+j]*p_fluxS[0][ei*NP+j]
                      + WDt[i*NP+j]*p_fluxT[0][ei*NP+j];
                rhs2 += WDr[i*NP+j]*p_fluxR[1][ei*NP+j]
                      + WDs[i*NP+j]*p_fluxS[1][ei*NP+j]
                      + WDt[i*NP+j]*p_fluxT[1][ei*NP+j];
                rhs3 += WDr[i*NP+j]*p_fluxR[2][ei*NP+j]
                      + WDs[i*NP+j]*p_fluxS[2][ei*NP+j]
                      + WDt[i*NP+j]*p_fluxT[2][ei*NP+j];
                rhs4 += WDr[i*NP+j]*p_fluxR[3][ei*NP+j]
                      + WDs[i*NP+j]*p_fluxS[3][ei*NP+j]
                      + WDt[i*NP+j]*p_fluxT[3][ei*NP+j];
                rhs5 += WDr[i*NP+j]*p_fluxR[4][ei*NP+j]
                      + WDs[i*NP+j]*p_fluxS[4][ei*NP+j]
                      + WDt[i*NP+j]*p_fluxT[4][ei*NP+j];
            }
                
            for (int j=0; j<NFACE*NFP; j++) {
                rhs1 -= LIFT[i*NFACE*NFP+j]*p_fluxLM[0][ei*NFACE*NFP+j];
                rhs2 -= LIFT[i*NFACE*NFP+j]*p_fluxLM[1][ei*NFACE*NFP+j];
                rhs3 -= LIFT[i*NFACE*NFP+j]*p_fluxLM[2][ei*NFACE*NFP+j];
                rhs4 -= LIFT[i*NFACE*NFP+j]*p_fluxLM[3][ei*NFACE*NFP+j];
                rhs5 -= LIFT[i*NFACE*NFP+j]*p_fluxLM[4][ei*NFACE*NFP+j];
            }
            
            p_utmp[0][ei*NP+i] = rhs1;
            p_utmp[1][ei*NP+i] = rhs2;
            p_utmp[2][ei*NP+i] = rhs3;
            p_utmp[3][ei*NP+i] = rhs4;
            p_utmp[4][ei*NP+i] = rhs5;
        }
    }
}
