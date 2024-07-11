#include "common.h"


int find_print_tstep(REAL tmax, REAL dt, int base_num_tstep) {
    int interval = int(tmax/dt)/base_num_tstep;
    string strN = to_string(interval);
    int head_digit = strN[0] - '0';
    int power = strN.size() - 1;
    int power_of_10 = head_digit*pow(10, power);
    
    return power_of_10;
}


void lserk_stage(
        int nelem,
        REAL dt, REAL rka, REAL rkb,
        REAL *utmp,
        REAL *k, REAL *u) {
    
    #pragma omp parallel for
    for (int idx=0; idx<nelem*NP; idx++) {
        k[idx]  = rka*k[idx] + dt*utmp[idx];
        u[idx] += rkb*k[idx];
    }
}


void runRK(DataHost &host, DGLM3DComm &comm) {
    calc_maxv(
        host.nelem,
        host.Fmask.data(),
        host.Fscale.data(),
        host.vx.data(), host.vy.data(), host.vz.data(),
        host.maxv.data(), host.maxv_fs.data());
        
    REAL local_dt = calc_dt(host.nelem, host.maxv_fs.data());
    REAL dt = comm.allreduce_dt(local_dt);
    
    if (host.tstep == 0)
        host.print_tstep = find_print_tstep(TMAX, dt, 30);    
    
    if (host.t + dt > TMAX) {
        dt = TMAX - host.t;
        host.last_tstep = true;
    }
    
    comm.sendrecv_maxv(host);
    
    update_tau(
        host.nelem,
        host.vmapF.data(),
        host.maxv.data(),
        host.tau.data());
    
    // five stages
    for (int s=0; s<5; s++) {
        comm.sendrecv_u(host);
        
        update_ub(
            host.nelem,
            host.Fmask.data(), host.vmapP.data(),
            host.u.data(),
            host.ub.data());
        
        calc_fluxLM(
            host.nelem,
            host.t + host.rkc[s]*dt,
            host.Fmask.data(), host.EtoB.data(),
            host.nx.data(), host.ny.data(), host.nz.data(),
            host.px.data(), host.py.data(), host.pz.data(),
            host.vx.data(), host.vy.data(), host.vz.data(),
            host.tau.data(), host.Fscale.data(),
            host.ub.data(), host.u.data(),
            host.fluxLM.data());
        
        calc_fluxes(
            host.nelem,
            host.rx.data(), host.ry.data(), host.rz.data(),
            host.sx.data(), host.sy.data(), host.sz.data(),
            host.tx.data(), host.ty.data(), host.tz.data(),
            host.vx.data(), host.vy.data(), host.vz.data(),
            host.u.data(),
            host.fluxR.data(), host.fluxS.data(), host.fluxT.data());
            
        update_rhs(
            host.nelem,
            host.WDr.data(), host.WDs.data(), host.WDt.data(), host.LIFT.data(),
            host.fluxR.data(), host.fluxS.data(), host.fluxT.data(), host.fluxLM.data(),
            host.utmp.data());

        lserk_stage(
            host.nelem,
            dt, host.rka[s], host.rkb[s],
            host.utmp.data(),
            host.k.data(), host.u.data());
    }
    
    host.dt = dt;
    host.t += dt;
    host.tstep += 1;
}
