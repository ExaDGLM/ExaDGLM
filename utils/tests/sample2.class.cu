class DGLM3D {    
public:
    cudaMalloc(&WDr,  NP*NP*sizeof(REAL));
    cudaMalloc(&LIFT, NP*NFACE*NFP*sizeof(REAL));
    cudaMalloc(&rx,   nelem*sizeof(REAL));
    cudaMalloc(&vmapP, nelem*NFACE*NFP*sizeof(int));
    cudaMalloc(&maxv, (nelem*NFACE + buf_size)*sizeof(REAL));
    
    ...
    
    for (int i=0; i<NVAR; i++) {
        cudaMalloc(&p_u[i],      (nelem*NP + buf_size*NFP)*sizeof(REAL));
        cudaMalloc(&p_ub[i],     nelem*NFACE*NFP*sizeof(REAL));        
        cudaMalloc(&p_fluxLM[i], nelem*NFACE*NFP*sizeof(REAL));
    }    
};
