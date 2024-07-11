#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"


__global__ void calc_maxv(
        int nelem,
        int *Fmask,
        REAL *Fscale,
        REAL *d_vx, REAL *d_vy, REAL *d_vz,
        REAL *maxv, REAL *maxv_fs) {
    
    int tid = threadIdx.x;
    int idx = blockDim.x*blockIdx.x + tid;
    
    if (idx >= nelem*NFACE*NFP) return;
    
    //
    // dynamic shared memory for max reduction
    //
    extern __shared__ REAL smem[];
    REAL *smem_v2 = &smem[0];
    
    //
    // Compute max velocity and sound speed
    //
    int ei = idx/(NFACE*NFP);
    int fi = (idx/NFP)%NFACE;
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];

    REAL vx = d_vx[idxM];
    REAL vy = d_vy[idxM];
    REAL vz = d_vz[idxM];
    smem_v2[tid] = vx*vx + vy*vy + vz*vz;  // (velocity)^2
    __syncthreads();
    
    //
    // max reduction
    //    
    for (int n=NFP; n>=2; n=int((n+1)/2)) {
        int m = int( (n+1)/2 );
        int lfi = tid/NFP;
        int lj = tid%NFP;
        
        if (lj < m) {
            if ((tid+m)-lfi*NFP < n)
                smem_v2[tid] = FMAX(smem_v2[tid], smem_v2[tid+m]);
        }
        __syncthreads();        
    }
    
    if (tid < blockDim.x/NFP) {        
        int fidx = blockIdx.x*(blockDim.x/NFP) + tid;
        if (fidx >= nelem*NFACE) return;
        
        int tidx = tid*NFP;
        REAL max_v = sqrt(smem_v2[tidx]);
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
        REAL *u,
        REAL *ub) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= nelem*NFACE*NFP) return;
    
    int ei = idx/(NFACE*NFP);
    int fi = (idx/NFP)%NFACE;
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
    int idxP = vmapP[idx];
    
    ub[idx] = 0.5*(u[idxM] + u[idxP]);
}
    

__global__ void calc_fluxLM(
        int nelem,
        REAL t,
        int *Fmask, int *EtoB,
        REAL *d_nx, REAL *d_ny, REAL *d_nz,
        REAL *px, REAL *py, REAL *pz,
        REAL *vx, REAL *vy, REAL *vz,
        REAL *tau, REAL *Fscale,
        REAL *d_ub, REAL *u,
        REAL *fluxLM) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= nelem*NFACE*NFP) return;
        
    int ei  = idx/(NFACE*NFP);
    int fi  = (idx/NFP)%NFACE;
    int fidx = ei*NFACE + fi;
    int pidx = ei*pNFF + idx%(NFACE*NFP);
    int idxM = ei*NP + Fmask[fi*NFP + idx%NFP];
    
    REAL nx = d_nx[fidx];  // normal vector
    REAL ny = d_ny[fidx];
    REAL nz = d_nz[fidx];
    REAL fs = Fscale[fidx];

    int bc = EtoB[fidx];
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

    fluxLM[pidx] = (n_dot_flux - tau[fidx]*(ub - u[idxM]))*fs;
}


__global__ void calc_fluxes(
        int nelem,
        REAL *rx, REAL *ry, REAL *rz,
        REAL *sx, REAL *sy, REAL *sz,
        REAL *tx, REAL *ty, REAL *tz,
        REAL *vx, REAL *vy, REAL *vz,
        REAL *u,
        REAL *fluxR, REAL *fluxS, REAL *fluxT) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int ei = idx/NP;
    int pidx = ei*pNP + idx%NP;
    
    if (idx >= nelem*NP) return;
    
    REAL fluxF = vx[idx]*u[idx];
    REAL fluxG = vy[idx]*u[idx];
    REAL fluxW = vz[idx]*u[idx];

    fluxR[pidx] = rx[ei]*fluxF + ry[ei]*fluxG + rz[ei]*fluxW;
    fluxS[pidx] = sx[ei]*fluxF + sy[ei]*fluxG + sz[ei]*fluxW;
    fluxT[pidx] = tx[ei]*fluxF + ty[ei]*fluxG + tz[ei]*fluxW;
}


void update_rhs(
        int nelem,
        REAL *WDr, REAL *WDs, REAL *WDt, REAL *LIFT,
        REAL *fluxR, REAL *fluxS, REAL *fluxT, REAL *fluxLM,
        REAL *utmp) {
    
    const REAL zero=0., one=1., minus=-1.;
    
    //
    // C = alpha*AxB + beta*C
    //
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasTGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, WDr, pNP, fluxR, pNP, &zero, utmp, pNP);
    
    cublasTGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, WDs, pNP, fluxS, pNP, &one, utmp, pNP);
    
    cublasTGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NP,
                &one, WDt, pNP, fluxT, pNP, &one, utmp, pNP);
    
    cublasTGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NP, nelem, NFACE*NFP,
                &minus, LIFT, pNP, fluxLM, pNFF, &one, utmp, pNP);
    
    cublasDestroy(handle);
}
