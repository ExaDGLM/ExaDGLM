#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "common.h"

using namespace std;


class DataDev {    
public:
    // DG operator matrix
    REAL *WDr, *WDs, *WDt, *LIFT;
    
    // mesh coordinate transformations
    REAL *rx, *ry, *rz, *sx, *sy, *sz, *tx, *ty, *tz;
    REAL *Fscale;
    
    // mesh normal vectors
    REAL *nx, *ny, *nz;
    
    // mesh physical coordinates
    REAL *px, *py, *pz;
    
    // velocity vectors
    REAL *vx, *vy, *vz;
    
    // DG parameters
    REAL *tau, *maxv, *maxv_fs;
    
    // mesh index arrays
    int *EtoB, *vmapF, *vmapP, *Fmask;
    
    // copy device to host
    vector<REAL> u0;
    
    // governing variable arrays
    REAL *u, *k, *utmp, *ub;
    REAL *fluxR, *fluxS, *fluxT, *fluxLM;
    
    int nelem;
    DataHost host;
    
    
    DataDev(DataHost &host) : host(host) {
        nelem = host.nelem;
        
        // CUDA device
        if (DEVID >= 0) {
            cudaError_t err = cudaSetDevice(DEVID);
            if (err != cudaSuccess)
                printf("Error setting CUDA device: %s\n", cudaGetErrorString(err));
        }
        
        // vector arrays
        u0.resize(nelem*NP);
            
        // CUDA arrays
        alloc_dev_arrays();
        init_dev_arrays();
        
        // set the shared memory bank size to 4(float) or 8(double) bytes
        cudaSharedMemConfig config = cudaSharedMemBankSizeTByte;
        cudaDeviceSetSharedMemConfig(config);        
        check_cuda_error("copy_h2d_all_shared_mem");
    }
    
    ~DataDev() {
        free_dev_arrays();
    }

    void check_cuda_error(string funcname) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            cout << "CUDA error(" << funcname << "): " << cudaGetErrorString(err) << endl;
    }    
    
    void alloc_dev_arrays() {
        cudaMalloc(&WDr,  pNP*NP*sizeof(REAL));
        cudaMalloc(&WDs,  pNP*NP*sizeof(REAL));
        cudaMalloc(&WDt,  pNP*NP*sizeof(REAL));
        cudaMalloc(&LIFT, pNP*NFACE*NFP*sizeof(REAL));
        cudaMalloc(&rx,     nelem*sizeof(REAL));
        cudaMalloc(&ry,     nelem*sizeof(REAL));
        cudaMalloc(&rz,     nelem*sizeof(REAL));
        cudaMalloc(&sx,     nelem*sizeof(REAL));
        cudaMalloc(&sy,     nelem*sizeof(REAL));
        cudaMalloc(&sz,     nelem*sizeof(REAL));
        cudaMalloc(&tx,     nelem*sizeof(REAL));        
        cudaMalloc(&ty,     nelem*sizeof(REAL));        
        cudaMalloc(&tz,     nelem*sizeof(REAL));
        cudaMalloc(&Fscale, nelem*NFACE*sizeof(REAL));
        cudaMalloc(&nx,     nelem*NFACE*sizeof(REAL));
        cudaMalloc(&ny,     nelem*NFACE*sizeof(REAL));
        cudaMalloc(&nz,     nelem*NFACE*sizeof(REAL));
        cudaMalloc(&px,     nelem*NP*sizeof(REAL));
        cudaMalloc(&py,     nelem*NP*sizeof(REAL));
        cudaMalloc(&pz,     nelem*NP*sizeof(REAL));
        cudaMalloc(&vx,     nelem*NP*sizeof(REAL));
        cudaMalloc(&vy,     nelem*NP*sizeof(REAL));
        cudaMalloc(&vz,     nelem*NP*sizeof(REAL));
        cudaMalloc(&EtoB,   nelem*NFACE*sizeof(int));
        cudaMalloc(&vmapF,  nelem*NFACE*sizeof(int));
        cudaMalloc(&vmapP,  nelem*NFACE*NFP*sizeof(int));
        cudaMalloc(&Fmask,  NFACE*NFP*sizeof(int));        
        cudaMalloc(&tau,     nelem*NFACE*sizeof(REAL));
        cudaMalloc(&maxv,    nelem*NFACE*sizeof(REAL));
        cudaMalloc(&maxv_fs, nelem*NFACE*sizeof(REAL));
        cudaMalloc(&u,      nelem*NP*sizeof(REAL));
        cudaMalloc(&k,      nelem*NP*sizeof(REAL));
        cudaMalloc(&ub,     nelem*NFACE*NFP*sizeof(REAL));
        cudaMalloc(&utmp,   nelem*pNP*sizeof(REAL));
        cudaMalloc(&fluxR,  nelem*pNP*sizeof(REAL));
        cudaMalloc(&fluxS,  nelem*pNP*sizeof(REAL));
        cudaMalloc(&fluxT,  nelem*pNP*sizeof(REAL));
        cudaMalloc(&fluxLM, nelem*pNFF*sizeof(REAL));

        cudaMemset(k, 0, nelem*NP*sizeof(REAL));  // zero initialize
        
        check_cuda_error("alloc_dev_arrays");
    }
    
    void init_dev_arrays() {
        constexpr auto h2d = cudaMemcpyHostToDevice;
        
        cudaMemcpy2D(WDr, pNP*sizeof(REAL), host.WDr.data(), NP*sizeof(REAL), NP*sizeof(REAL), NP, h2d);
        cudaMemcpy2D(WDs, pNP*sizeof(REAL), host.WDs.data(), NP*sizeof(REAL), NP*sizeof(REAL), NP, h2d);
        cudaMemcpy2D(WDt, pNP*sizeof(REAL), host.WDt.data(), NP*sizeof(REAL), NP*sizeof(REAL), NP, h2d);
        cudaMemcpy2D(LIFT, pNP*sizeof(REAL), host.LIFT.data(), NP*sizeof(REAL), NP*sizeof(REAL), NFACE*NFP, h2d);
        
        cudaMemcpy(rx, host.rx.data(), host.rx.size()*sizeof(REAL), h2d);
        cudaMemcpy(ry, host.ry.data(), host.ry.size()*sizeof(REAL), h2d);
        cudaMemcpy(rz, host.rz.data(), host.rz.size()*sizeof(REAL), h2d);
        cudaMemcpy(sx, host.sx.data(), host.sx.size()*sizeof(REAL), h2d);
        cudaMemcpy(sy, host.sy.data(), host.sy.size()*sizeof(REAL), h2d);
        cudaMemcpy(sz, host.sz.data(), host.sz.size()*sizeof(REAL), h2d);
        cudaMemcpy(tx, host.tx.data(), host.tx.size()*sizeof(REAL), h2d);        
        cudaMemcpy(ty, host.ty.data(), host.ty.size()*sizeof(REAL), h2d);
        cudaMemcpy(tz, host.tz.data(), host.tz.size()*sizeof(REAL), h2d);
        cudaMemcpy(Fscale, host.Fscale.data(), host.Fscale.size()*sizeof(REAL), h2d);
        cudaMemcpy(nx, host.nx.data(), host.nx.size()*sizeof(REAL), h2d);
        cudaMemcpy(ny, host.ny.data(), host.ny.size()*sizeof(REAL), h2d);
        cudaMemcpy(nz, host.nz.data(), host.nz.size()*sizeof(REAL), h2d);
        cudaMemcpy(px, host.px.data(), host.px.size()*sizeof(REAL), h2d);
        cudaMemcpy(py, host.py.data(), host.py.size()*sizeof(REAL), h2d);
        cudaMemcpy(pz, host.pz.data(), host.pz.size()*sizeof(REAL), h2d);
        cudaMemcpy(EtoB, host.EtoB.data(), host.EtoB.size()*sizeof(int), h2d);
        cudaMemcpy(vmapF, host.vmapF.data(), host.vmapF.size()*sizeof(int), h2d);
        cudaMemcpy(vmapP, host.vmapP.data(), host.vmapP.size()*sizeof(int), h2d);
        cudaMemcpy(Fmask, host.Fmask.data(), host.Fmask.size()*sizeof(int), h2d);
        
        check_cuda_error("init_dev_arrays");
    }
    
    void copy_h2d(vector<REAL> &h_src, REAL *d_dst) {
        constexpr auto h2d = cudaMemcpyHostToDevice;
        cudaMemcpy(d_dst, h_src.data(), h_src.size()*sizeof(REAL), h2d);
        check_cuda_error("copy_h2d");
    }
    
    void copy_d2h(REAL *d_src, vector<REAL> &h_dst) {
        constexpr auto d2h = cudaMemcpyDeviceToHost;
        cudaMemcpy(h_dst.data(), d_src, h_dst.size()*sizeof(REAL), d2h);
        check_cuda_error("copy_d2h");
    }
        
    void write_u(int tstep) {
        string s = to_string(tstep);
        string fname = "u_" + string(TDIGIT - s.length(), '0') + s + ".bin";
        
        copy_d2h(u, u0);
        write_bin_file<REAL>(u0, host.dataout + fname);
    }    
    
    void free_dev_arrays() {
        cudaFree(WDr);
        cudaFree(WDs);
        cudaFree(WDt);
        cudaFree(LIFT);
        cudaFree(rx);
        cudaFree(ry);
        cudaFree(rz);
        cudaFree(sx);
        cudaFree(sy);
        cudaFree(sz);
        cudaFree(tx);
        cudaFree(ty);
        cudaFree(tz);
        cudaFree(Fscale);
        cudaFree(nx);
        cudaFree(ny);
        cudaFree(nz);
        cudaFree(px);
        cudaFree(py);
        cudaFree(pz);        
        cudaFree(EtoB);
        cudaFree(vmapF);
        cudaFree(vmapP);        
        cudaFree(Fmask);
        cudaFree(tau);
        cudaFree(maxv);
        cudaFree(maxv_fs);
        cudaFree(u);
        cudaFree(k);
        cudaFree(utmp);
        cudaFree(ub);            
        cudaFree(fluxR);
        cudaFree(fluxS);
        cudaFree(fluxT);
        cudaFree(fluxLM);
    }
};
