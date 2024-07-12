#include "common.h"

int main() {
    DGLM3DComm comm;
    DataHost host(comm.myrank);
    
    init(host);
    
    while (host.t < TMAX) {
        runRK(host, comm);
        
        if (host.tstep%host.print_tstep == 0 or host.last_tstep) {
            if (host.myrank == 0) 
                printf("tstep=%d, dt=%.8f, t=%.5f\n", host.tstep, host.dt, host.t);
            host.write_u(host.tstep);
        }
    }
    
    return 0;
}