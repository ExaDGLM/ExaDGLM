#include <mpi.h>
#include "common.h"


//-----------------------------------------------------------------------------
// Debugging macros
//-----------------------------------------------------------------------------
#define MPICHECK(cmd) do {                        \
    int e = cmd;                                  \
    if( e != MPI_SUCCESS ) {                      \
        printf("Failed: MPI error %s:%d '%d'\n",  \
               __FILE__,__LINE__, e);             \
        exit(EXIT_FAILURE);                       \
    }                                             \
} while(0)


//-----------------------------------------------------------------------------
// MPI communications
//-----------------------------------------------------------------------------
class DGLM3DComm {
public:
    int nproc, myrank;
    
    DGLM3DComm() {
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        //printf("nproc=%d, myrank=%d\n", nproc, myrank);
    }
    
    ~DGLM3DComm() {
        MPI_Finalize();
    }
    
    
    REAL allreduce_dt(REAL &local_min) {
        REAL global_min;    

        MPI_Allreduce(&local_min, &global_min, 1, MPIREAL, MPI_MIN, MPI_COMM_WORLD);

        return global_min;
    }

    
    void sendrecv_maxv(DataHost &host) {
        int buf_size = host.buf_size;
        int comm_size= host.comm_size;

        int tag = 10;
        MPI_Request send_request[comm_size];
        MPI_Request recv_request[comm_size];

        // Copy data to send buffer
        for (int i=0; i<buf_size; i++) {
            int sidx = host.sendbuf_face_idxs[i];
            host.sendbuf_face[i] = host.maxv[sidx];
        }    

        // Exchange data across MPI processes
        for (int j=0; j<comm_size; j++) {
            int nbr_rank   = host.comm_face_idxs[j*3  ];
            int start_idx  = host.comm_face_idxs[j*3+1];
            int data_count = host.comm_face_idxs[j*3+2];

            MPI_Isend(&host.sendbuf_face[start_idx], data_count, MPIREAL, nbr_rank, tag,
                      MPI_COMM_WORLD, &send_request[j]);

            MPI_Irecv(&host.recvbuf_face[start_idx], data_count, MPIREAL, nbr_rank, tag, 
                      MPI_COMM_WORLD, &recv_request[j]);
        }

        for (int j=0; j<comm_size; j++) {
            MPI_Wait(&send_request[j], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[j], MPI_STATUS_IGNORE);
        }

        // copy data from recv buffer
        int shift = host.nelem*NFACE;
        for (int i=0; i<buf_size; i++) {
            host.maxv[shift+i] = host.recvbuf_face[i];
        }
    }
    

    void sendrecv_p_u(DataHost &host) {
        int buf_size = host.buf_size;
        int comm_size= host.comm_size;
        
        int tag = 20;
        MPI_Request send_request[comm_size];
        MPI_Request recv_request[comm_size];

        // Copy data to send buffer
        for (int j=0; j<comm_size; j++) {
            int start_idx  = host.comm_nfp_idxs[j*3+1];
            int data_count = host.comm_nfp_idxs[j*3+2];

            for (int n=0; n<NVAR; n++) {
                for (int k=0; k<data_count; k++) {
                    int sidx = host.sendbuf_nfp_idxs[start_idx+k];
                    host.sendbuf_nfp[start_idx*NVAR + data_count*n + k] = host.p_u[n][sidx];
                }
            }
        } 

        // Exchange data across MPI processes
        for (int j=0; j<comm_size; j++) {
            int nbr_rank   = host.comm_nfp_idxs[j*3  ];
            int start_idx  = host.comm_nfp_idxs[j*3+1]*NVAR;
            int data_count = host.comm_nfp_idxs[j*3+2]*NVAR;

            MPI_Isend(&host.sendbuf_nfp[start_idx], data_count, MPIREAL, nbr_rank, tag,
                      MPI_COMM_WORLD, &send_request[j]);

            MPI_Irecv(&host.recvbuf_nfp[start_idx], data_count, MPIREAL, nbr_rank, tag, 
                      MPI_COMM_WORLD, &recv_request[j]);
        }

        for (int j=0; j<comm_size; j++) {
            MPI_Wait(&send_request[j], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[j], MPI_STATUS_IGNORE);
        }

        // Copy data from recv buffer
        int shift = host.nelem*NP;
        for (int j=0; j<comm_size; j++) {
            int start_idx  = host.comm_nfp_idxs[j*3+1];
            int data_count = host.comm_nfp_idxs[j*3+2];

            for (int n=0; n<NVAR; n++) {
                for (int k=0; k<data_count; k++) {
                    host.p_u[n][shift + start_idx + k] = host.recvbuf_nfp[start_idx*NVAR + data_count*n + k];
                }
            }
        }
    }
};
