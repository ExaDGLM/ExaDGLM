#include <vector>
#include <string>
#include "common.h"

using namespace std;


class DGLM3DEuler {
public:
    // dimension array (Nelem,)
    vector<int> dim = vector<int>(1);  // c++11 member initializer
    
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
    REAL dt;
    REAL t = 0;
    int tstep = 0;
    int print_tstep;
    bool last_tstep = false;
    
    
    DGLM3DEuler() {
        read_dim();
        alloc_vec_arrays();
        init_vec_arrays();
    }
    
    ~DGLM3DEuler() {
    }
    
    void read_dim() {        
        read_bin_file<int>(dim, DATAIN "dim.bin");
        nelem = dim[0];
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
        maxv.resize(nelem*NFACE);
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
            p_u[i]      = vector<REAL>(nelem*NP);
            p_k[i]      = vector<REAL>(nelem*NP, 0);  // zero initialization
            p_utmp[i]   = vector<REAL>(nelem*NP);
            p_ub[i]     = vector<REAL>(nelem*NFACE*NFP);
            p_fluxR[i]  = vector<REAL>(nelem*NP);
            p_fluxS[i]  = vector<REAL>(nelem*NP);
            p_fluxT[i]  = vector<REAL>(nelem*NP);
            p_fluxLM[i] = vector<REAL>(nelem*NFACE*NFP);            
        }
    }
    
    void init_vec_arrays() {
        read_bin_file<REAL>(WDr, DATAIN "WDr.bin");
        read_bin_file<REAL>(WDs, DATAIN "WDs.bin");
        read_bin_file<REAL>(WDt, DATAIN "WDt.bin");
        read_bin_file<REAL>(LIFT, DATAIN "LIFT.bin");
        read_bin_file<REAL>(rx, DATAIN "rx.bin");
        read_bin_file<REAL>(ry, DATAIN "ry.bin");
        read_bin_file<REAL>(rz, DATAIN "rz.bin");
        read_bin_file<REAL>(sx, DATAIN "sx.bin");
        read_bin_file<REAL>(sy, DATAIN "sy.bin");
        read_bin_file<REAL>(sz, DATAIN "sz.bin");
        read_bin_file<REAL>(tx, DATAIN "tx.bin");
        read_bin_file<REAL>(ty, DATAIN "ty.bin");
        read_bin_file<REAL>(tz, DATAIN "tz.bin");
        read_bin_file<REAL>(Fscale, DATAIN "Fscale.bin");
        read_bin_file<REAL>(nx, DATAIN "nx.bin");
        read_bin_file<REAL>(ny, DATAIN "ny.bin");
        read_bin_file<REAL>(nz, DATAIN "nz.bin");
        read_bin_file<REAL>(px, DATAIN "px.bin");
        read_bin_file<REAL>(py, DATAIN "py.bin");
        read_bin_file<REAL>(pz, DATAIN "pz.bin");
        read_bin_file<int>(EtoB, DATAIN "EtoB.bin");        
        read_bin_file<int>(vmapF, DATAIN "vmapF.bin");
        read_bin_file<int>(vmapP, DATAIN "vmapP.bin");
        read_bin_file<int>(Fmask, DATAIN "Fmask.bin");
    }    
    
    void write_u(int tstep) {
        string s = to_string(tstep);
        string fname = "u1_" + string(TDIGIT - s.length(), '0') + s + ".bin";
        write_bin_file<REAL>(p_u[0], string(DATAOUT) + fname);
    }
};
