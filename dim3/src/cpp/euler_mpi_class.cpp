#include <vector>
#include <string>
#include <cassert>
#include "common.h"

using namespace std;


class DGLM3DEuler {
public:
    // dimension array (Nelem, myrank, comm_size, buf_size)
    vector<int> dim = vector<int>(4);  // c++11 member initializer
    
    // DG operator matrix
    vector<REAL> WDr, WDs, WDt, LIFT;
    
    // mesh coordinate transformations
    vector<REAL> rx, ry, rz, sx, sy, sz, tx, ty, tz;
    vector<REAL> Fscale;
    
    // mesh normal vectors
    vector<REAL> nx, ny, nz;
    
    // mesh physical coordinates
    vector<REAL> px, py, pz;
    
    // mesh index arrays
    vector<int> EtoB, vmapF, vmapP, Fmask;
    
    // DG parameters
    vector<REAL> tau, maxv, maxv_fs;
    
    // Pack of C++ vector arrays
    vector<vector<REAL>> p_u, p_k, p_utmp, p_ub;
    vector<vector<REAL>> p_fluxR, p_fluxS, p_fluxT, p_fluxLM;
    
    // To write data output
    vector<REAL> u;
    
    // MPI communication index arrays
    vector<int> comm_face_idxs, comm_nfp_idxs;
    vector<int> sendbuf_face_idxs, sendbuf_nfp_idxs;
    
    // MPI buffer arrays
    vector<REAL> sendbuf_face, sendbuf_nfp;
    vector<REAL> recvbuf_face, recvbuf_nfp;
    
    // Low-Storage Explicit Runge-Kutta    
    REAL rka[5] = 
            {                                 0,
              -567301805773.0 / 1357537059087.0,
             -2404267990393.0 / 2016746695238.0,
             -3550918686646.0 / 2091501179385.0,
             -1275806237668.0 /  842570457699.0};

    REAL rkb[5] =
            {1432997174477.0 /  9575080441755.0,
             5161836677717.0 / 13612068292357.0,
             1720146321549.0 /  2090206949498.0,
             3134564353537.0 /  4481467310338.0,
             2277821191437.0 / 14882151754819.0};

    REAL rkc[5] =
            {                                 0,
              1432997174477.0 / 9575080441755.0,
              2526269341429.0 / 6820363962896.0,
              2006345519317.0 / 3224310063776.0,
              2802321613138.0 / 2924317926251.0};
    
    int nelem;
    int myrank, comm_size, buf_size;
    string datain, dataout;
    REAL dt;
    REAL t = 0;
    int tstep = 0;
    int print_tstep;
    bool last_tstep = false;
    
    
    DGLM3DEuler(int myrank) : myrank(myrank) {
        read_dim();
        alloc_vec_arrays();
        init_vec_arrays();
    }
    
    ~DGLM3DEuler() {
    }
    
    void read_dim() {
        datain = replace_string(DATAIN, "__RANK__", std::to_string(myrank));
        dataout = replace_string(DATAOUT, "__RANK__", std::to_string(myrank));
        
        read_bin_file<int>(dim, datain + "dim.bin");
        assert(myrank == dim[1]);
        nelem = dim[0];
        comm_size = dim[2];
        buf_size = dim[3];
    }
    
    void alloc_vec_arrays() {        
        // DG operator matrix
        WDr.resize(NP*NP);
        WDs.resize(NP*NP);
        WDt.resize(NP*NP);
        LIFT.resize(NP*NFACE*NFP);    

        // mesh coordinate transformations
        rx.resize(nelem);
        ry.resize(nelem);
        rz.resize(nelem);
        sx.resize(nelem);
        sy.resize(nelem);
        sz.resize(nelem);
        tx.resize(nelem);
        ty.resize(nelem);
        tz.resize(nelem);
        Fscale.resize(nelem*NFACE);
        
        // mesh normal vectors
        nx.resize(nelem*NFACE);
        ny.resize(nelem*NFACE);
        nz.resize(nelem*NFACE);
        
        // mesh physical coordinates
        px.resize(nelem*NP);
        py.resize(nelem*NP);
        pz.resize(nelem*NP);

        // mesh index arrays
        EtoB.resize(nelem*NFACE);
        vmapF.resize(nelem*NFACE);
        vmapP.resize(nelem*NFACE*NFP);        
        Fmask.resize(NFACE*NFP);

        // DG parameters
        tau.resize(nelem*NFACE);
        maxv.resize(nelem*NFACE + buf_size);  // MPI communication
        maxv_fs.resize(nelem*NFACE);
        
        // Pack of C++ vector arrays
        p_u.resize(NVAR);
        p_k.resize(NVAR);
        p_utmp.resize(NVAR);
        p_ub.resize(NVAR);
        p_fluxR.resize(NVAR);
        p_fluxS.resize(NVAR);
        p_fluxT.resize(NVAR);
        p_fluxLM.resize(NVAR);
        
        for (int i=0; i<NVAR; i++) {
            p_u[i]      = vector<REAL>(nelem*NP + buf_size*NFP);  // MPI communication
            p_k[i]      = vector<REAL>(nelem*NP, 0);  // zero initialization
            p_utmp[i]   = vector<REAL>(nelem*NP);
            p_ub[i]     = vector<REAL>(nelem*NFACE*NFP);
            p_fluxR[i]  = vector<REAL>(nelem*NP);
            p_fluxS[i]  = vector<REAL>(nelem*NP);
            p_fluxT[i]  = vector<REAL>(nelem*NP);
            p_fluxLM[i] = vector<REAL>(nelem*NFACE*NFP);            
        }
        
        // To write data output
        u.resize(nelem*NP);
        
        // MPI communication index arrays
        comm_face_idxs.resize(comm_size*3);
        comm_nfp_idxs.resize(comm_size*3);
        sendbuf_face_idxs.resize(buf_size);
        sendbuf_nfp_idxs.resize(buf_size*NFP);
        
        // MPI buffer arrays
        sendbuf_face.resize(buf_size);
        recvbuf_face.resize(buf_size);
        sendbuf_nfp.resize(buf_size*NFP*NVAR);
        recvbuf_nfp.resize(buf_size*NFP*NVAR);
    }
    
    void init_vec_arrays() {
        read_bin_file<REAL>(WDr, datain + "WDr.bin");
        read_bin_file<REAL>(WDs, datain + "WDs.bin");
        read_bin_file<REAL>(WDt, datain + "WDt.bin");
        read_bin_file<REAL>(LIFT, datain + "LIFT.bin");
        read_bin_file<REAL>(rx, datain + "rx.bin");
        read_bin_file<REAL>(ry, datain + "ry.bin");
        read_bin_file<REAL>(rz, datain + "rz.bin");
        read_bin_file<REAL>(sx, datain + "sx.bin");
        read_bin_file<REAL>(sy, datain + "sy.bin");
        read_bin_file<REAL>(sz, datain + "sz.bin");
        read_bin_file<REAL>(tx, datain + "tx.bin");
        read_bin_file<REAL>(ty, datain + "ty.bin");
        read_bin_file<REAL>(tz, datain + "tz.bin");
        read_bin_file<REAL>(Fscale, datain + "Fscale.bin");
        read_bin_file<REAL>(nx, datain + "nx.bin");
        read_bin_file<REAL>(ny, datain + "ny.bin");
        read_bin_file<REAL>(nz, datain + "nz.bin");
        read_bin_file<REAL>(px, datain + "px.bin");
        read_bin_file<REAL>(py, datain + "py.bin");
        read_bin_file<REAL>(pz, datain + "pz.bin");
        read_bin_file<int>(EtoB, datain + "EtoB.bin");        
        read_bin_file<int>(vmapF, datain + "vmapF.bin");
        read_bin_file<int>(vmapP, datain + "vmapP.bin");
        read_bin_file<int>(Fmask, datain + "Fmask.bin");
        read_bin_file<int>(comm_face_idxs, datain + "comm_face_idxs.bin");
        read_bin_file<int>(comm_nfp_idxs, datain + "comm_nfp_idxs.bin");
        read_bin_file<int>(sendbuf_face_idxs, datain + "sendbuf_face_idxs.bin");
        read_bin_file<int>(sendbuf_nfp_idxs, datain + "sendbuf_nfp_idxs.bin");
    }    
    
    void write_u(int tstep) {
        for (int i=0; i<nelem*NP; i++)
            u[i] = p_u[0][i];
        
        string s = to_string(tstep);
        string fname = "u1_" + string(TDIGIT - s.length(), '0') + s + ".bin";
        write_bin_file<REAL>(u, dataout + fname);
    }
};
