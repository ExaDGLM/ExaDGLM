#include "common.h"

        
int main(int argc, char** argv) {
    DGLM3DComm comm(argc, argv);
    DataHost host(comm.myrank);
    DataDev dev(host);
    
    init(host, dev);
    
    for (int tstep=1; tstep<=10; tstep++) {
        runRK(host, dev, comm);
    }
    
    // Wait for all kernels to complete
    cudaDeviceSynchronize();
    
    return 0;
}
