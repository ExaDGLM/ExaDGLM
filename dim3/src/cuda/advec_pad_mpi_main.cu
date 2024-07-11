#include "common.h"

        
int main(int argc, char** argv) {
    DGLM3DComm comm(argc, argv);
    DataHost host(comm.myrank);
    DataDev dev(host);
    
    init(host, dev);
    
    while (host.t < TMAX) {        
        runRK(host, dev, comm);
        
        if (host.tstep%host.print_tstep == 0 or host.last_tstep) {
            if (host.myrank == 0)
                printf("tstep=%d, dt=%.8f, t=%.5f\n", host.tstep, host.dt, host.t);
            dev.write_u(host.tstep);
        }
    }
    
    return 0;
}
