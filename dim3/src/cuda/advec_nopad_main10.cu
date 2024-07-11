#include "common.h"

        
int main() {
    DataHost host;
    DataDev dev(host);
    
    init(host, dev);
    
    for (int tstep=1; tstep<=10; tstep++) {
        runRK(host, dev);
    }
    
    // Wait for all kernels to complete
    cudaDeviceSynchronize();
    
    return 0;
}
