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
    
    // DG parameters
    REAL *tau, *maxv, *maxv_fs;
    
    // mesh index arrays
    int *EtoB, *vmapF, *vmapP, *Fmask;
    
    // MPI communication index arrays
    int *sendbuf_face_idxs, *sendbuf_nfp_idxs;
    REAL *sendbuf_face, *sendbuf_nfp;
    REAL *recvbuf_face, *recvbuf_nfp;
    
    // copy device to host
    vector<REAL> u;
    
    // Pack of CUDA array pointers for cublasTgemmBatched
    // Note that they require both host vectors and device pointers
    vector<REAL*> p_WDr, p_WDs, p_WDt, p_LIFT;
    vector<REAL*> p_u, p_k, p_utmp, p_ub;
    vector<REAL*> p_fluxR, p_fluxS, p_fluxT, p_fluxLM; 
    REAL **pd_WDr, **pd_WDs, **pd_WDt, **pd_LIFT;
    REAL **pd_u, **pd_k, **pd_utmp, **pd_ub;
    REAL **pd_fluxR, **pd_fluxS, **pd_fluxT, **pd_fluxLM;
    
    int nelem;
    int buf_size;
    DataHost host;
    
    
    DataDev(DataHost &host) : host(host) {
        nelem = host.nelem;
        buf_size = host.buf_size;
        
        // vector arrays
        u.resize(nelem*NP);
            
        // CUDA arrays
        alloc_dev_arrays();
        alloc_dev_pack_arrays();
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
        cudaMalloc(&EtoB,   nelem*NFACE*sizeof(int));
        cudaMalloc(&vmapF,  nelem*NFACE*sizeof(int));
        cudaMalloc(&vmapP,  nelem*NFACE*NFP*sizeof(int));
        cudaMalloc(&Fmask,  NFACE*NFP*sizeof(int));        
        cudaMalloc(&tau,     nelem*NFACE*sizeof(REAL));
        cudaMalloc(&maxv,    (nelem*NFACE + buf_size)*sizeof(REAL));
        cudaMalloc(&maxv_fs, nelem*NFACE*sizeof(REAL));
        cudaMalloc(&sendbuf_face_idxs, buf_size*sizeof(int));
        cudaMalloc(&sendbuf_nfp_idxs,  buf_size*NFP*sizeof(int));
        cudaMalloc(&sendbuf_face, buf_size*sizeof(REAL));
        cudaMalloc(&recvbuf_face, buf_size*sizeof(REAL));
        cudaMalloc(&sendbuf_nfp,  buf_size*NFP*NVAR*sizeof(REAL));
        cudaMalloc(&recvbuf_nfp,  buf_size*NFP*NVAR*sizeof(REAL));
        
        check_cuda_error("alloc_dev_arrays");
    }
    
    void alloc_dev_pack_arrays() {
        // C++ vector arrays are used for saving the pointer of CUDA arrays
        p_WDr.resize(NVAR);
        p_WDs.resize(NVAR);
        p_WDt.resize(NVAR);
        p_LIFT.resize(NVAR);
        p_u.resize(NVAR);
        p_k.resize(NVAR);
        p_utmp.resize(NVAR);
        p_ub.resize(NVAR);
        p_fluxR.resize(NVAR);
        p_fluxS.resize(NVAR);
        p_fluxT.resize(NVAR);
        p_fluxLM.resize(NVAR);
                
        for (int i=0; i<NVAR; i++) {
            cudaMalloc(&p_u[i],      (nelem*NP + buf_size*NFP)*sizeof(REAL));
            cudaMalloc(&p_k[i],      nelem*NP*sizeof(REAL));
            cudaMalloc(&p_ub[i],     nelem*NFACE*NFP*sizeof(REAL));
            cudaMalloc(&p_utmp[i],   nelem*pNP*sizeof(REAL));
            cudaMalloc(&p_fluxR[i],  nelem*pNP*sizeof(REAL));
            cudaMalloc(&p_fluxS[i],  nelem*pNP*sizeof(REAL));
            cudaMalloc(&p_fluxT[i],  nelem*pNP*sizeof(REAL));
            cudaMalloc(&p_fluxLM[i], nelem*pNFF*sizeof(REAL));
            cudaMemset(p_k[i], 0, nelem*NP*sizeof(REAL));  // zero initialize
        }        
        
        // Pack of CUDA array pointers
        cudaMalloc((void**)&pd_WDr,    NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_WDs,    NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_WDt,    NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_LIFT,   NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_u,      NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_k,      NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_utmp,   NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_ub,     NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_fluxR,  NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_fluxS,  NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_fluxT,  NVAR*sizeof(REAL*));
        cudaMalloc((void**)&pd_fluxLM, NVAR*sizeof(REAL*));
        
        // copy pointers
        p_WDr  = {WDr,  WDr,  WDr,  WDr,  WDr};
        p_WDs  = {WDs,  WDs,  WDs,  WDs,  WDs};
        p_WDt  = {WDt,  WDt,  WDt,  WDt,  WDt};
        p_LIFT = {LIFT, LIFT, LIFT, LIFT, LIFT};        
        
        constexpr auto h2d = cudaMemcpyHostToDevice;
        cudaMemcpy(pd_WDr,    p_WDr.data(),    NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_WDs,    p_WDs.data(),    NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_WDt,    p_WDt.data(),    NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_LIFT,   p_LIFT.data(),   NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_u,      p_u.data(),      NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_k,      p_k.data(),      NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_utmp,   p_utmp.data(),   NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_ub,     p_ub.data(),     NVAR*sizeof(REAL*), h2d);        
        cudaMemcpy(pd_fluxR,  p_fluxR.data(),  NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_fluxS,  p_fluxS.data(),  NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_fluxT,  p_fluxT.data(),  NVAR*sizeof(REAL*), h2d);
        cudaMemcpy(pd_fluxLM, p_fluxLM.data(), NVAR*sizeof(REAL*), h2d);
        
        check_cuda_error("alloc_dev_pack_arrays");          
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
        cudaMemcpy(sendbuf_face_idxs, host.sendbuf_face_idxs.data(), host.sendbuf_face_idxs.size()*sizeof(int), h2d);
        cudaMemcpy(sendbuf_nfp_idxs,  host.sendbuf_nfp_idxs.data(),  host.sendbuf_nfp_idxs.size()*sizeof(int), h2d);
        
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
        string fname = "u1_" + string(TDIGIT - s.length(), '0') + s + ".bin";
        
        copy_d2h(p_u[0], u);
        write_bin_file<REAL>(u, host.dataout + fname);
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
                
        for (int i=0; i<NVAR; i++) {
            cudaFree(p_u[i]);
            cudaFree(p_k[i]);
            cudaFree(p_utmp[i]);
            cudaFree(p_ub[i]);            
            cudaFree(p_fluxR[i]);
            cudaFree(p_fluxS[i]);
            cudaFree(p_fluxT[i]);
            cudaFree(p_fluxLM[i]);
        }
        
        cudaFree(sendbuf_face_idxs);
        cudaFree(sendbuf_nfp_idxs);
        cudaFree(sendbuf_face);
        cudaFree(recvbuf_face);
        cudaFree(sendbuf_nfp);
        cudaFree(recvbuf_nfp);
    }
};
