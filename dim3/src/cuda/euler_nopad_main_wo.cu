#include "common.h"

        
int main() {
    DataHost host;
    DataDev dev(host);
    
    init(host, dev);
    
    while (host.t < TMAX) {        
        runRK(host, dev);
        
        if (host.tstep%host.print_tstep == 0 or host.last_tstep) {
            printf("tstep=%d, dt=%.8f, t=%.5f\n", host.tstep, host.dt, host.t);

        }
    }
    
    return 0;
}
