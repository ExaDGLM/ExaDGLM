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
        vector<REAL> *p_utmp,
        vector<REAL> *p_k, vector<REAL> *p_u) {
    
    #pragma omp parallel for
    for (int idx=0; idx<nelem*NP; idx++) {
        p_k[0][idx]  = rka*p_k[0][idx] + dt*p_utmp[0][idx];
        p_u[0][idx] += rkb*p_k[0][idx];
        
        p_k[1][idx]  = rka*p_k[1][idx] + dt*p_utmp[1][idx];
        p_u[1][idx] += rkb*p_k[1][idx];
        
        p_k[2][idx]  = rka*p_k[2][idx] + dt*p_utmp[2][idx];
        p_u[2][idx] += rkb*p_k[2][idx];
        
        p_k[3][idx]  = rka*p_k[3][idx] + dt*p_utmp[3][idx];
        p_u[3][idx] += rkb*p_k[3][idx];
        
        p_k[4][idx]  = rka*p_k[4][idx] + dt*p_utmp[4][idx];
        p_u[4][idx] += rkb*p_k[4][idx];
    }
}


void runRK(DataHost &host) {
    calc_maxv(
        host.nelem,
        host.Fmask.data(),
        host.Fscale.data(),
        host.p_u.data(),
        host.maxv.data(), host.maxv_fs.data());
        
    REAL dt = calc_dt(host.nelem, host.maxv_fs.data());
    
    if (host.tstep == 0)
        host.print_tstep = find_print_tstep(TMAX, dt, 30);    
    
    if (host.t + dt > TMAX) {
        dt = TMAX - host.t;
        host.last_tstep = true;
    }
    
    update_tau(
        host.nelem,
        host.vmapF.data(),
        host.maxv.data(),
        host.tau.data());
    
    // five stages
    for (int s=0; s<5; s++) {
        update_ub(
            host.nelem,
            host.Fmask.data(), host.vmapP.data(),
            host.p_u.data(),
            host.p_ub.data());
        
        calc_fluxLM(
            host.nelem,
            host.t + host.rkc[s]*dt,
            host.Fmask.data(), host.EtoB.data(),
            host.nx.data(), host.ny.data(), host.nz.data(),
            host.px.data(), host.py.data(), host.pz.data(),
            host.tau.data(), host.Fscale.data(),
            host.p_ub.data(), host.p_u.data(),
            host.p_fluxLM.data());
        
        calc_fluxes(
            host.nelem,
            host.rx.data(), host.ry.data(), host.rz.data(),
            host.sx.data(), host.sy.data(), host.sz.data(),
            host.tx.data(), host.ty.data(), host.tz.data(),
            host.p_u.data(),
            host.p_fluxR.data(), host.p_fluxS.data(), host.p_fluxT.data());
                        
        update_rhs(
            host.nelem,
            host.WDr.data(), host.WDs.data(), host.WDt.data(), host.LIFT.data(),
            host.p_fluxR.data(), host.p_fluxS.data(), host.p_fluxT.data(), host.p_fluxLM.data(),
            host.p_utmp.data());

        lserk_stage(
            host.nelem,
            dt, host.rka[s], host.rkb[s],
            host.p_utmp.data(),
            host.p_k.data(), host.p_u.data());
    }
    
    host.dt = dt;
    host.t += dt;
    host.tstep += 1;
}
