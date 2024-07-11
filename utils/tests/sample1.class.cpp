class DGLM3D {    
public:
    vector<REAL> rx, ry, rz;
    vector<REAL> Fscale;
    vector<REAL> maxv;
    vector<int> EtoB, vmapP;
    vector<REAL> k;
    vector<vector<REAL>> p_u, p_k;
    vector<int> comm_face_idxs;
    
    ...

    void alloc_vec_arrays() {
        rx.resize(nelem);
        ry.resize(nelem);
        rz.resize(nelem);
        Fscale.resize(nelem*NFACE);
        maxv.resize(nelem*NFACE + buf_size);
        EtoB.resize(nelem*NFACE);
        vmapP.resize(nelem*NFACE*NFP);

        k.resize(nelem*NP, 0);
        p_u.resize(NVAR);
        p_k.resize(NVAR);
        for (int i=0; i<NVAR; i++) {
            p_u[i]   = vector<REAL>(nelem*NP + buf_size*NFP);
            p_k[i]   = vector<REAL>(nelem*NP, 0);  // zero initialization
        }
        
        comm_face_idxs.resize(comm_size*3);
    }
};
