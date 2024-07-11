#include <cmath>
#include "common.h"


void init_u(
        int nelem,
        REAL *px, REAL *py, REAL *pz,
        REAL *vx, REAL *vy, REAL *vz,
        REAL *u) {

    #pragma omp parallel for
    for (int idx=0; idx<nelem*NP; idx++) {
        // physical coordinates
        REAL x = px[idx];
        REAL y = py[idx];
        REAL z = pz[idx];

        vx[idx] = 1.;
        vy[idx] = 1.;
        vz[idx] = 1.;
        u[idx] = sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
    }
}


void init(DataHost &host) {
    init_u(
        host.nelem,
        host.px.data(), host.py.data(), host.pz.data(),
        host.vx.data(), host.vy.data(), host.vz.data(),
        host.u.data());
    
    host.write_u(0);
}
