#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"


__global__ void calc_maxv(
        int nelem,
        int *Fmask,
        REAL *Fscale,
        REAL **pd_u,
        REAL *maxv, REAL *maxv_fs) {
    
    int tid = threadIdx.x;
    int idx = blockDim.x*blockIdx.x + tid;
    
    if (idx >= nelem*NFACE*NFP) return;
    
    //
    // dynamic shared memory for max reduction
    //
    extern __shared__ REAL smem[];
    REAL *smem_v2 = &smem[0];
    REAL *smem_c2 = &smem[blockDim.x];
    
    //
    // Compute max velocity and sound speed
    //
    int ei = idx/(NFACE*NFP);
    int fi = (idx/NFP)%NFACE;
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
    
    REAL u1 = pd_u[0][idxM];
    REAL u2 = pd_u[1][idxM];
    REAL u3 = pd_u[2][idxM];
    REAL u4 = pd_u[3][idxM];
    REAL u5 = pd_u[4][idxM];

    REAL vx = u2/u1;
    REAL vy = u3/u1;
    REAL vz = u4/u1;
    REAL v2 = vx*vx + vy*vy + vz*vz;         // (velocity)^2
    REAL pres = (GAMMA-1)*(u5 - 0.5*u1*v2);  // pressure
    REAL c2 = abs(GAMMA*pres/u1);            // (sound speed)^2
    
    smem_v2[tid] = v2;
    smem_c2[tid] = c2;
    __syncthreads();
    
    //
    // max reduction
    //    
    for (int n=NFP; n>=2; n=int((n+1)/2)) {
        int m = int( (n+1)/2 );
        int lfi = tid/NFP;
        int lj = tid%NFP;
        
        if (lj < m) {
            if ((tid+m)-lfi*NFP < n) {
                smem_v2[tid] = FMAX(smem_v2[tid], smem_v2[tid+m]);
                smem_c2[tid] = FMAX(smem_c2[tid], smem_c2[tid+m]);
            }
        }
        __syncthreads();        
    }
    
    if (tid < blockDim.x/NFP) {        
        int fidx = blockIdx.x*(blockDim.x/NFP) + tid;
        if (fidx >= nelem*NFACE) return;
        
        int tidx = tid*NFP;
        REAL max_v = sqrt(smem_v2[tidx]) + sqrt(smem_c2[tidx]);
        maxv[fidx] = max_v;
        maxv_fs[fidx] = max_v*Fscale[fidx];
    }
}


REAL calc_dt(int nelem, REAL *maxv_fs) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // maximum reduction
    int max_idx;
    cublasITAMAX(handle, nelem*NFACE, maxv_fs, 1, &max_idx);
    
    cublasDestroy(handle);

    // max velocity with max_idx
    REAL max_v;
    cudaMemcpy(&max_v, &maxv_fs[max_idx - 1], sizeof(REAL), cudaMemcpyDeviceToHost);    
    
    REAL dt = (CFL/(2*N+1))*(1/max_v);
  
    return dt;
}


__global__ void update_tau(
        int nelem,
        int *vmapF,
        REAL *maxv,
        REAL *tau) {
    
    int fidx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (fidx >= nelem*NFACE) return;
    
    int nbr_fidx = vmapF[fidx];
        
    tau[fidx] = 0.25*(maxv[fidx] + maxv[nbr_fidx])*TAU_SCALE;
}
                
                
__global__ void update_ub(
        int nelem,
        int *Fmask, int *vmapP,
        REAL **pd_u,
        REAL **pd_ub) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= nelem*NFACE*NFP) return;
    
    int ei = idx/(NFACE*NFP);
    int fi = (idx/NFP)%NFACE;
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
    int idxP = vmapP[idx];
    
    pd_ub[0][idx] = 0.5*(pd_u[0][idxM] + pd_u[0][idxP]);
    pd_ub[1][idx] = 0.5*(pd_u[1][idxM] + pd_u[1][idxP]);
    pd_ub[2][idx] = 0.5*(pd_u[2][idxM] + pd_u[2][idxP]);
    pd_ub[3][idx] = 0.5*(pd_u[3][idxM] + pd_u[3][idxP]);
    pd_ub[4][idx] = 0.5*(pd_u[4][idxM] + pd_u[4][idxP]);
}
    

__global__ void calc_fluxLM(
        int nelem,
        REAL t,
        int *Fmask, int *EtoB,
        REAL *d_nx, REAL *d_ny, REAL *d_nz,
        REAL *px, REAL *py, REAL *pz,
        REAL *tau, REAL *Fscale,
        REAL **pd_ub, REAL **pd_u,
        REAL **pd_fluxLM) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= nelem*NFACE*NFP) return;
        
    int ei  = idx/(NFACE*NFP);
    int fi  = (idx/NFP)%NFACE;
    int fidx = ei*NFACE + fi;
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
    
    REAL nx = d_nx[fidx];  // normal vector
    REAL ny = d_ny[fidx];
    REAL nz = d_nz[fidx];
    REAL fs = Fscale[fidx];

    int bc = EtoB[fidx];
    REAL ub1, ub2, ub3, ub4, ub5, vb2;
    
    if (bc == 1) {
        // Dirichlet boundary conditions
        //REAL x = px[idxM];
        //REAL y = py[idxM];
        //REAL z = pz[idxM];
        //ub = function of x,y,z,t
    }
    else {
        ub1 = pd_ub[0][idx];
        ub2 = pd_ub[1][idx];
        ub3 = pd_ub[2][idx];
        ub4 = pd_ub[3][idx];
        ub5 = pd_ub[4][idx];
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

    pd_fluxLM[0][idx] = (n_dot_flux1 - tau[fidx]*(ub1 - pd_u[0][idxM]))*fs;
    pd_fluxLM[1][idx] = (n_dot_flux2 - tau[fidx]*(ub2 - pd_u[1][idxM]))*fs;
    pd_fluxLM[2][idx] = (n_dot_flux3 - tau[fidx]*(ub3 - pd_u[2][idxM]))*fs;
    pd_fluxLM[3][idx] = (n_dot_flux4 - tau[fidx]*(ub4 - pd_u[3][idxM]))*fs;
    pd_fluxLM[4][idx] = (n_dot_flux5 - tau[fidx]*(ub5 - pd_u[4][idxM]))*fs;
}


__global__ void calc_fluxes(
        int nelem,
        REAL *rx, REAL *ry, REAL *rz,
        REAL *sx, REAL *sy, REAL *sz,
        REAL *tx, REAL *ty, REAL *tz,
        REAL **pd_u,
        REAL **pd_fluxR, REAL **pd_fluxS, REAL **pd_fluxT) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int ei= idx/NP;
    
    if (idx >= nelem*NP) return;
    
    REAL u1 = pd_u[0][idx];
    REAL u2 = pd_u[1][idx];
    REAL u3 = pd_u[2][idx];
    REAL u4 = pd_u[3][idx];
    REAL u5 = pd_u[4][idx];
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

    pd_fluxR[0][idx] = rx[ei]*fluxF1 + ry[ei]*fluxG1 + rz[ei]*fluxW1;
    pd_fluxR[1][idx] = rx[ei]*fluxF2 + ry[ei]*fluxG2 + rz[ei]*fluxW2;
    pd_fluxR[2][idx] = rx[ei]*fluxF3 + ry[ei]*fluxG3 + rz[ei]*fluxW3;
    pd_fluxR[3][idx] = rx[ei]*fluxF4 + ry[ei]*fluxG4 + rz[ei]*fluxW4;
    pd_fluxR[4][idx] = rx[ei]*fluxF5 + ry[ei]*fluxG5 + rz[ei]*fluxW5;
    pd_fluxS[0][idx] = sx[ei]*fluxF1 + sy[ei]*fluxG1 + sz[ei]*fluxW1;
    pd_fluxS[1][idx] = sx[ei]*fluxF2 + sy[ei]*fluxG2 + sz[ei]*fluxW2;
    pd_fluxS[2][idx] = sx[ei]*fluxF3 + sy[ei]*fluxG3 + sz[ei]*fluxW3;
    pd_fluxS[3][idx] = sx[ei]*fluxF4 + sy[ei]*fluxG4 + sz[ei]*fluxW4;
    pd_fluxS[4][idx] = sx[ei]*fluxF5 + sy[ei]*fluxG5 + sz[ei]*fluxW5;
    pd_fluxT[0][idx] = tx[ei]*fluxF1 + ty[ei]*fluxG1 + tz[ei]*fluxW1;
    pd_fluxT[1][idx] = tx[ei]*fluxF2 + ty[ei]*fluxG2 + tz[ei]*fluxW2;
    pd_fluxT[2][idx] = tx[ei]*fluxF3 + ty[ei]*fluxG3 + tz[ei]*fluxW3;
    pd_fluxT[3][idx] = tx[ei]*fluxF4 + ty[ei]*fluxG4 + tz[ei]*fluxW4;
    pd_fluxT[4][idx] = tx[ei]*fluxF5 + ty[ei]*fluxG5 + tz[ei]*fluxW5;
}


void update_rhs(
        int nelem,
        REAL **pd_WDr, REAL **pd_WDs, REAL **pd_WDt, REAL **pd_LIFT,
        REAL **pd_fluxR, REAL **pd_fluxS, REAL **pd_fluxT, REAL **pd_fluxLM,
        REAL **pd_utmp) {
    
    const REAL zero=0., one=1., minus=-1.;
    
    //
    // C = alpha*AxB + beta*C
    //
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasTGEMMBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, pd_WDr, NP, pd_fluxR, NP, &zero, pd_utmp, NP, NVAR);
    
    cublasTGEMMBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, pd_WDs, NP, pd_fluxS, NP, &one, pd_utmp, NP, NVAR);
    
    cublasTGEMMBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, pd_WDt, NP, pd_fluxT, NP, &one, pd_utmp, NP, NVAR);
    
    cublasTGEMMBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NFACE*NFP,
                &minus, pd_LIFT, NP, pd_fluxLM, NFACE*NFP, &one, pd_utmp, NP, NVAR);
    
    cublasDestroy(handle);
}
