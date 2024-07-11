#include <cmath>
#include "common.h"


void init_u(
        int nelem,
        REAL *px, REAL *py, REAL *pz,
        vector<REAL> *p_u) {
    
    // setup parameters
    REAL vx =  1.;
    REAL vy = -0.5;
    REAL vz =  1.;
    REAL pres = 1.;

    #pragma omp parallel for
    for (int idx=0; idx<nelem*NP; idx++) {
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
        p_u[0][idx] = u1;
        p_u[1][idx] = u1*vx;
        p_u[2][idx] = u1*vy;
        p_u[3][idx] = u1*vz;
        p_u[4][idx] = pres/(GAMMA-1) + 0.5*u1*(vx*vx + vy*vy + vz*vz);
    }
}


void init(DataHost &host) {
    init_u(
        host.nelem,
        host.px.data(), host.py.data(), host.pz.data(),
        host.p_u.data());
    
    host.write_u(0);
}
