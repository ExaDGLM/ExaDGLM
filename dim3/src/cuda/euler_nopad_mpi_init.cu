#include <cmath>
#include <cuda_runtime.h>
#include "common.h"


__global__ void init_u(
        int nelem,
        REAL *px, REAL *py, REAL *pz,
        REAL **pd_u) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;    
    
    if (idx >= nelem*NP) return;
    
    // setup parameters
    REAL vx =  1.;
    REAL vy = -0.5;
    REAL vz =  1.;
    REAL pres = 1.;

    // physical coordinates
    REAL x = px[idx];
    REAL y = py[idx];
    REAL z = pz[idx];

    // u1: density
    // u2: x-momentum
    // u3: y-momentum
    // u4: z-momentum
    // u5: energy
    REAL u1 = 1 + 0.2*sin(2*PI*(x + y + z));
    pd_u[0][idx] = u1;
    pd_u[1][idx] = u1*vx;
    pd_u[2][idx] = u1*vy;
    pd_u[3][idx] = u1*vz;
    pd_u[4][idx] = pres/(GAMMA-1) + 0.5*u1*(vx*vx + vy*vy + vz*vz);
}


void init(DataHost &host, DataDev &dev) {
    int bpg;
    
    bpg = host.nelem*NP/TPB + 1;
    init_u<<<bpg,TPB>>>(
        host.nelem,
        dev.px, dev.py, dev.pz,
        dev.pd_u);
    
    dev.write_u(0);
}
