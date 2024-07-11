#include <cmath>
#include <cuda_runtime.h>
#include "common.h"


__global__ void init_u(
        int nelem,
        REAL *px, REAL *py, REAL *pz,
        REAL *vx, REAL *vy, REAL *vz,
        REAL *u) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;    
    
    if (idx >= nelem*NP) return;

    // physical coordinates
    REAL x = px[idx];
    REAL y = py[idx];
    REAL z = pz[idx];
    
    vx[idx] = 1.;
    vy[idx] = 1.;
    vz[idx] = 1.;
    u[idx] = sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
}


void init(DataHost &host, DataDev &dev) {
    int bpg;
    
    bpg = host.nelem*NP/TPB + 1;
    init_u<<<bpg,TPB>>>(
        host.nelem,
        dev.px, dev.py, dev.pz,
        dev.vx, dev.vy, dev.vz,
        dev.u);
    
    dev.write_u(0);
}
