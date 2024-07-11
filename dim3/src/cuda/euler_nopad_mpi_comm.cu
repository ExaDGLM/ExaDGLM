#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
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


#define CUDACHECK(cmd) do {                        \
    cudaError_t err = cmd;                         \
    if (err != cudaSuccess) {                      \
        printf("Failed: Cuda error %s:%d '%s'\n",  \
               __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(EXIT_FAILURE);                        \
    }                                              \
} while(0)
    
    
#define NCCLCHECK(cmd) do {                        \
    ncclResult_t r = cmd;                          \
    if (r!= ncclSuccess) {                         \
        printf("Failed, NCCL error %s:%d '%s'\n",  \
               __FILE__, __LINE__, ncclGetErrorString(r));  \
        exit(EXIT_FAILURE);                        \
    }                                              \
} while(0)


//-----------------------------------------------------------------------------
// Identify device ID in the same machine
//-----------------------------------------------------------------------------
static uint64_t getHostHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}


static void getHostName(char* hostname) {
    int namelen;
    MPI_Get_processor_name(hostname, &namelen);
    for (int i=0; i< namelen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}


int get_devid(int myrank, int nproc) {
    // calculating local_gpu based on hostname    
    uint64_t hostHashs[nproc];
    char hostname[1024];
    getHostName(hostname);
    hostHashs[myrank] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    
    int devid = 0;
    for (int p=0; p<nproc; p++) {
        if (p == myrank) break;
        if (hostHashs[p] == hostHashs[myrank]) devid++;
    }
    
    return devid;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// copy buffer kernels
//-----------------------------------------------------------------------------
__global__ void copy_send_buffer_face(
        int buf_size,
        int *sendbuf_idxs,
        REAL *maxv,
        REAL *sendbuf) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= buf_size) return;
    
    int sidx = sendbuf_idxs[idx];
    sendbuf[idx] = maxv[sidx];
}


__global__ void copy_recv_buffer_face(
        int nelem, int buf_size,
        REAL *recvbuf,
        REAL *maxv) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= buf_size) return;
    
    maxv[nelem*NFACE + idx] = recvbuf[idx];
}


__global__ void copy_send_buffer_nfp(
        int start_idx, int data_count,
        int *sendbuf_idxs,
        REAL **pd_u,
        REAL *sendbuf) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;    
    
    if (idx >= data_count*NVAR) return;
    
    int n = idx/data_count;
    int k = idx%data_count;    
    int sidx = sendbuf_idxs[start_idx+k];
    sendbuf[start_idx*NVAR + data_count*n + k] = pd_u[n][sidx];
}


__global__ void copy_recv_buffer_nfp(
        int nelem, int start_idx, int data_count,
        REAL *recvbuf,
        REAL **pd_u) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;    
    
    if (idx >= data_count*NVAR) return;
    
    int n = idx/data_count;
    int k = idx%data_count;
    pd_u[n][nelem*NP + start_idx + k] = recvbuf[start_idx*NVAR + data_count*n + k];
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// MPI-NCCL communications
//-----------------------------------------------------------------------------
class DGLM3DComm {
public:
    int nproc, myrank;
    int devid;
    ncclComm_t ncclComm;
    cudaStream_t stream;
    
    DGLM3DComm(int argc, char** argv) {
        init_mpi(argc, argv);
        init_nccl();
        cudaStreamCreate(&stream);
        //printf("nproc=%d, myrank=%d, devid=%d, ncclComm=%p\n", nproc, myrank, devid, (void*)ncclComm);
    }
    
    ~DGLM3DComm() {
        cudaStreamDestroy(stream);
        ncclCommDestroy(ncclComm);        
        MPI_Finalize();
    }
    
    
    void init_mpi(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    }
    
    
    void init_nccl() {
        devid = get_devid(myrank, nproc);
        cudaSetDevice(devid);

        // the root rank creates a unique ID and share to other ranks
        ncclUniqueId id;
        if (myrank == 0) ncclGetUniqueId(&id);        
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);        

        // initialize NCCL communicator
        ncclCommInitRank(&ncclComm, nproc, id, myrank);
    }
    
    
    REAL allreduce_dt(REAL &local_min) {
        REAL global_min;    

        MPI_Allreduce(&local_min, &global_min, 1, MPIREAL, MPI_MIN, MPI_COMM_WORLD);

        return global_min;
    }


    void sendrecv_maxv(DataHost &host, DataDev &dev) {
        //
        // Copy data to send buffer
        //
        int bpg = host.buf_size/TPB + 1;
        copy_send_buffer_face<<<bpg,TPB>>>(
            host.buf_size, dev.sendbuf_face_idxs, dev.maxv, dev.sendbuf_face);

        //
        // Exchange data across MPI processes
        //
        ncclGroupStart();
        for (int j=0; j<host.comm_size; j++) {
            int nbr_rank   = host.comm_face_idxs[j*3  ];
            int start_idx  = host.comm_face_idxs[j*3+1];
            int data_count = host.comm_face_idxs[j*3+2];

            ncclSend(&dev.sendbuf_face[start_idx], data_count, NCCLREAL, nbr_rank, ncclComm, stream);
            ncclRecv(&dev.recvbuf_face[start_idx], data_count, NCCLREAL, nbr_rank, ncclComm, stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(stream);

        //
        // copy data from recv buffer
        //    
        copy_recv_buffer_face<<<bpg,TPB>>>(
            host.nelem, host.buf_size, dev.recvbuf_face, dev.maxv);
    }


    void sendrecv_p_u(DataHost &host, DataDev &dev) {
        int bpg;

        //
        // Copy data to send buffer
        //
        for (int j=0; j<host.comm_size; j++) {
            int start_idx  = host.comm_nfp_idxs[j*3+1];
            int data_count = host.comm_nfp_idxs[j*3+2];

            bpg = data_count*NVAR/TPB + 1;
            copy_send_buffer_nfp<<<bpg,TPB>>>(
                start_idx, data_count, dev.sendbuf_nfp_idxs, dev.pd_u, dev.sendbuf_nfp);
        }

        //
        // Exchange data across MPI processes
        //
        ncclGroupStart();
        for (int j=0; j<host.comm_size; j++) {
            int nbr_rank   = host.comm_nfp_idxs[j*3  ];
            int start_idx  = host.comm_nfp_idxs[j*3+1]*NVAR;
            int data_count = host.comm_nfp_idxs[j*3+2]*NVAR;

            ncclSend(&dev.sendbuf_nfp[start_idx], data_count, NCCLREAL, nbr_rank, ncclComm, stream);
            ncclRecv(&dev.recvbuf_nfp[start_idx], data_count, NCCLREAL, nbr_rank, ncclComm, stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(stream);

        //
        // Copy data from recv buffer
        //
        for (int j=0; j<host.comm_size; j++) {
            int start_idx  = host.comm_nfp_idxs[j*3+1];
            int data_count = host.comm_nfp_idxs[j*3+2];

            bpg = data_count*NVAR/TPB + 1;
            copy_recv_buffer_nfp<<<bpg,TPB>>>(
                host.nelem, start_idx, data_count, dev.recvbuf_nfp, dev.pd_u);
        }
    }
};
