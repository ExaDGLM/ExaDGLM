#include "common.h"


int find_print_tstep(REAL tmax, REAL dt, int base_num_tstep) {
    int interval = int(tmax/dt)/base_num_tstep;
    string strN = to_string(interval);
    int head_digit = strN[0] - '0';
    int power = strN.size() - 1;
    int power_of_10 = head_digit*pow(10, power);
    
    return power_of_10;
}


__global__ void lserk_stage(
        int nelem,
        REAL dt, REAL rka, REAL rkb,
        REAL *utmp,
        REAL *k, REAL *u) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int pidx = (idx/NP)*pNP + (idx%NP);
    
    if (idx >= nelem*NP) return;
    
    k[idx]  = rka*k[idx] + dt*utmp[pidx];
    u[idx] += rkb*k[idx];
}


void runRK(DataHost &host, DataDev &dev, DGLM3DComm &comm) {
    int tpb, bpg, smem;
        
    tpb = TPBV;
    bpg = host.nelem*NFACE*NFP/tpb + 1;
    smem = (2*tpb)*sizeof(REAL);
    calc_maxv<<<bpg,tpb,smem>>>(
        host.nelem,
        dev.Fmask,
        dev.Fscale,
        dev.vx, dev.vy, dev.vz,
        dev.maxv, dev.maxv_fs);
    
    REAL local_dt = calc_dt(host.nelem, dev.maxv_fs);
    REAL dt = comm.allreduce_dt(local_dt);
    
    if (host.tstep == 0)
        host.print_tstep = find_print_tstep(TMAX, dt, 30);    
    
    if (host.t + dt > TMAX) {
        dt = TMAX - host.t;
        host.last_tstep = true;
    }
    
    comm.sendrecv_maxv(host, dev);
            
    bpg = host.nelem*NFACE/TPB + 1;
    update_tau<<<bpg,TPB>>>(
        host.nelem,
        dev.vmapF, 
        dev.maxv,
        dev.tau);
    
    // five stages
    for (int s=0; s<5; s++) {
        comm.sendrecv_u(host, dev);
                                 
        bpg = host.nelem*NFACE*NFP/TPB + 1;
        update_ub<<<bpg,TPB>>>(
            host.nelem,
            dev.Fmask, dev.vmapP,
            dev.u,
            dev.ub);
        
        bpg = host.nelem*NFACE*NFP/TPB + 1;
        calc_fluxLM<<<bpg,TPB>>>(
            host.nelem,
            host.t + host.rkc[s]*dt,
            dev.Fmask, dev.EtoB,
            dev.nx, dev.ny, dev.nz,
            dev.px, dev.py, dev.pz,
            dev.vx, dev.vy, dev.vz,
            dev.tau, dev.Fscale,
            dev.ub, dev.u,
            dev.fluxLM);
        
        bpg = host.nelem*NP/TPB + 1;
        calc_fluxes<<<bpg,TPB>>>(
            host.nelem,
            dev.rx, dev.ry, dev.rz,
            dev.sx, dev.sy, dev.sz,
            dev.tx, dev.ty, dev.tz,
            dev.vx, dev.vy, dev.vz,
            dev.u,
            dev.fluxR, dev.fluxS, dev.fluxT);
        
        update_rhs(
            host.nelem,
            dev.WDr, dev.WDs, dev.WDt, dev.LIFT,
            dev.fluxR, dev.fluxS, dev.fluxT, dev.fluxLM,
            dev.utmp);

        bpg = host.nelem*NP/TPB + 1;
        lserk_stage<<<bpg,TPB>>>(
            host.nelem,
            dt, host.rka[s], host.rkb[s],
            dev.utmp,
            dev.k, dev.u);
    }
    
    host.dt = dt;
    host.t += dt;
    host.tstep += 1;
}
