import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae


def check_sendbuf_nfp_idxs(mesh_list):
    comms = {mesh.myrank:[] for mesh in mesh_list}  # {rank: [ememIdx tuple, ...], ...}
    sorted_comms = {}
    
    for mesh in mesh_list:        
        NNp = mesh.Nelem*mesh.Np
        
        for ei in range(mesh.Nelem):
            for fi in range(mesh.Nface):
                if mesh.myrank != mesh.EtoP[ei,fi] - 1:
                    nbr_rank    = mesh.EtoP[ei,fi] - 1
                    nbr_elemTag = mesh.EtoE[ei,fi]
                    nbr_fidx    = mesh.EtoF[ei,fi]
                    nbr_ridx    = mesh.EtoR[ei,fi]
                    nbr_elemIdx = mesh_list[nbr_rank].mesh_elem.elemTag2Idxs[nbr_elemTag]
                    
                    comms[nbr_rank].append( ((mesh.myrank,ei,fi), (nbr_elemIdx,nbr_elemTag,nbr_fidx,nbr_ridx)) )
                    #if nbr_rank == 0:
                    #    print(f"({mesh.myrank},{ei},{fi}) -> {nbr_rank} ({nbr_elemIdx}/{nbr_elemTag},{nbr_fidx},{nbr_ridx})")
    #print()        
    for mesh in mesh_list:
        sorted_comm_list = sorted(comms[mesh.myrank], key=lambda x: (x[0][0], x[1][0], x[1][2]))
        
        for face_idx, ((rank,ei,fi), (elemIdx,elemTag,fidx,ridx)) in enumerate(sorted_comm_list):
            expect_idxs = elemIdx*mesh.Np + mesh.op_face.FmaskP[fidx,ridx]
            sendbuf_nfp_idx = mesh.sendbuf_nfp_idxs[face_idx,:]
            #if mesh.myrank == 0:
            #    print(f"({rank},{ei},{fi}) -> ({elemIdx}/{elemTag},{fidx},{ridx}) -> {sendbuf_nfp_idx}, {expect_idxs}")                
            assert_ae(sendbuf_nfp_idx, expect_idxs)
            
            
def check_comm_nfp_idxs(mesh_list):
    us = {}
    sendbufs = {}
    recvbufs = {}
        
    for rank, mesh in enumerate(mesh_list):
        assert_ae(mesh.myrank, rank)
        
        #
        # solution array
        #
        NNp = mesh.Nelem*mesh.Np
        buf_size = mesh.sendbuf_nfp_idxs.size
        var_size = NNp + buf_size
        #print(f"rank={rank}, Nelem={mesh.Nelem}, Np={mesh.Np}, NNp={NNp}")
        
        us[mesh.myrank] = np.zeros((var_size,4), 'i4')  # (rank, elemIdx, elemTag, np_seq)
        
        for ei in range(mesh.Nelem):
            for k in range(mesh.Np):
                us[mesh.myrank][ei*mesh.Np+k,:] = (mesh.myrank, ei, mesh.mesh_elem.elemIdx2Tags[ei], k)
        
        #print(f"us[{mesh.myrank}]\n{us[mesh.myrank]}")
        
        #
        # communication buffers
        #
        sendbufs[mesh.myrank] = np.zeros((buf_size,4), 'i4')
        recvbufs[mesh.myrank] = np.zeros((buf_size,4), 'i4')
        
        #
        # copy from solution array to sendbuf
        #
        for i, vidx in enumerate(mesh.sendbuf_nfp_idxs.ravel()):
            sendbufs[mesh.myrank][i,:] = us[mesh.myrank][vidx,:]
            
        #print(f"sendbufs[{mesh.myrank}]\n{sendbufs[mesh.myrank]}")
    
    for mesh in mesh_list:
        #
        # data exchange (emulate MPI)
        #
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            recv_sl = slice(start_idx, start_idx+data_count)
            
            nbr_mesh = mesh_list[nbr_rank]
            for rank, sidx, count in nbr_mesh.comm_nfp_idxs:
                if rank == mesh.myrank:
                    send_sl = slice(sidx, sidx+count)
                    break
            
            recvbufs[mesh.myrank][recv_sl,:] = sendbufs[nbr_rank][send_sl,:]
            
        #
        # copy from recvbuf to solution array
        #
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            src_sl = slice(start_idx, start_idx + data_count)
            dst_sl = slice(mesh.Nelem*mesh.Np + start_idx, mesh.Nelem*mesh.Np + start_idx + data_count)
            
            us[mesh.myrank][dst_sl,:] = recvbufs[mesh.myrank][src_sl,:]
            
        #
        # check vmapM and vmapP
        #
        u = us[mesh.myrank]
        
        for ei in range(mesh.Nelem):
            for fi in range(mesh.Nface):
                idxM = mesh.vmapM[ei,fi,:]
                idxP = mesh.vmapP[ei,fi,:]
                
                #print(f"ei={ei}, fi={fi}")
                #print(f"idxM={idxM}, idxP={idxP}")
                rank    = np.unique(u[idxM,0])
                elemIdx = np.unique(u[idxM,1])
                elemTag = np.unique(u[idxM,2])
                seqs    = u[idxM,3]
                
                nbr_rank    = np.unique(u[idxP,0])
                nbr_elemIdx = np.unique(u[idxP,1])
                nbr_elemTag = np.unique(u[idxP,2])
                nbr_seqs    = u[idxP,3]
                
                assert_ae(ei, elemIdx)
                                
                if rank == nbr_rank:
                    if mesh.EtoE[ei,fi] == -99: continue
                    assert_ae(mesh.EtoE[ei,fi], nbr_elemIdx)
                    assert_ae(mesh.EtoP[ei,fi]-1, rank)
                else:
                    assert_ae(mesh.EtoE[ei,fi], nbr_elemTag)
                    assert_ae(mesh.EtoP[ei,fi]-1, nbr_rank)
            
            
def check_comm_px(mesh_list):
    pxs = {}
    sendbufs = {}
    recvbufs = {}
        
    for rank, mesh in enumerate(mesh_list):
        assert_ae(mesh.myrank, rank)
        
        #
        # solution array
        #
        NNp = mesh.Nelem*mesh.Np
        buf_size = mesh.sendbuf_nfp_idxs.size
        #print(f"rank={rank}, Nelem={mesh.Nelem}, Np={mesh.Np}, NNp={NNp}")        
        pxs[mesh.myrank] = np.zeros(NNp + buf_size, 'f8')        
        pxs[mesh.myrank][:NNp] = mesh.PX.ravel()
        
        #
        # communication buffers
        #
        sendbufs[mesh.myrank] = np.zeros(buf_size, 'f8')
        recvbufs[mesh.myrank] = np.zeros(buf_size, 'f8')
        
        #
        # copy from solution array to sendbuf
        #
        for i, vidx in enumerate(mesh.sendbuf_nfp_idxs.ravel()):
            sendbufs[mesh.myrank][i] = pxs[mesh.myrank][vidx]
    
    for mesh in mesh_list:
        #
        # data exchange (emulate MPI)
        #
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            recv_sl = slice(start_idx, start_idx+data_count)
            
            nbr_mesh = mesh_list[nbr_rank]
            for rank, sidx, count in nbr_mesh.comm_nfp_idxs:
                if rank == mesh.myrank:
                    send_sl = slice(sidx, sidx+count)
                    break
            
            recvbufs[mesh.myrank][recv_sl] = sendbufs[nbr_rank][send_sl]
            
        #
        # copy from recvbuf to solution array
        #
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            src_sl = slice(start_idx, start_idx + data_count)
            dst_sl = slice(mesh.Nelem*mesh.Np + start_idx, mesh.Nelem*mesh.Np + start_idx + data_count)
            
            pxs[mesh.myrank][dst_sl] = recvbufs[mesh.myrank][src_sl]
            
        #
        # check vmapM and vmapP
        #
        px = pxs[mesh.myrank]
        
        for ei in range(mesh.Nelem):
            for fi in range(mesh.Nface):
                idxM = mesh.vmapM[ei,fi,:]
                idxP = mesh.vmapP[ei,fi,:]
                
                #print(f"ei={ei}, fi={fi}")
                #print(f"idxM={idxM}, idxP={idxP}")
                try:
                    assert_aae(px[idxM], px[idxP], 14)
                except:
                    assert_aae(np.abs(px[idxM] - px[idxP]), 1.4, 14)
                    
                    
def check_comm_pxyz(mesh_list):
    pxs = {}
    pys = {}
    pzs = {}
    sendbufs = {}
    recvbufs = {}
        
    for rank, mesh in enumerate(mesh_list):
        assert_ae(mesh.myrank, rank)
        
        #
        # solution array
        #
        NNp = mesh.Nelem*mesh.Np
        buf_size = mesh.sendbuf_nfp_idxs.size
        #print(f"rank={rank}, Nelem={mesh.Nelem}, Np={mesh.Np}, NNp={NNp}")
        
        pxs[mesh.myrank] = np.zeros(NNp + buf_size, 'f8')
        pys[mesh.myrank] = np.zeros(NNp + buf_size, 'f8')
        pzs[mesh.myrank] = np.zeros(NNp + buf_size, 'f8')
        
        pxs[mesh.myrank][:NNp] = mesh.PX.ravel()
        pys[mesh.myrank][:NNp] = mesh.PY.ravel()
        pzs[mesh.myrank][:NNp] = mesh.PZ.ravel()
        
        #
        # communication buffers
        #
        nvar = 3
        sendbufs[mesh.myrank] = np.zeros(buf_size*nvar, 'f8')
        recvbufs[mesh.myrank] = np.zeros(buf_size*nvar, 'f8')
        
        #
        # copy from solution array to sendbuf
        #
        for i, vidx in enumerate(mesh.sendbuf_nfp_idxs.ravel()):
            sendbufs[mesh.myrank][i*nvar  ] = pxs[mesh.myrank][vidx]
            sendbufs[mesh.myrank][i*nvar+1] = pys[mesh.myrank][vidx]
            sendbufs[mesh.myrank][i*nvar+2] = pzs[mesh.myrank][vidx]
    
    for mesh in mesh_list:
        #
        # data exchange (emulate MPI)
        #
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            recv_sl = slice(start_idx*nvar, (start_idx + data_count)*nvar)
            
            nbr_mesh = mesh_list[nbr_rank]
            for rank, sidx, count in nbr_mesh.comm_nfp_idxs:
                if rank == mesh.myrank:
                    send_sl = slice(sidx*nvar, (sidx + count)*nvar)
                    break
            
            recvbufs[mesh.myrank][recv_sl] = sendbufs[nbr_rank][send_sl]
            
        #
        # copy from recvbuf to solution array
        #
        NNp = mesh.Nelem*mesh.Np
        for nbr_rank, start_idx, data_count in mesh.comm_nfp_idxs:
            for i in range(start_idx, start_idx + data_count):
                pxs[mesh.myrank][NNp+i] = recvbufs[mesh.myrank][i*nvar  ]
                pys[mesh.myrank][NNp+i] = recvbufs[mesh.myrank][i*nvar+1]
                pzs[mesh.myrank][NNp+i] = recvbufs[mesh.myrank][i*nvar+2]
            
        #
        # check vmapM and vmapP
        #
        px = pxs[mesh.myrank]
        py = pys[mesh.myrank]
        pz = pzs[mesh.myrank]
        
        for ei in range(mesh.Nelem):
            for fi in range(mesh.Nface):
                idxM = mesh.vmapM[ei,fi,:]
                idxP = mesh.vmapP[ei,fi,:]
                
                #print(f"ei={ei}, fi={fi}")
                #print(f"idxM={idxM}, idxP={idxP}")
                try:
                    assert_aae(px[idxM], px[idxP], 14)
                except:
                    assert_aae(np.abs(px[idxM] - px[idxP]), 1.4, 14)
                    
                try:
                    assert_aae(py[idxM], py[idxP], 14)
                except:
                    assert_aae(np.abs(py[idxM] - py[idxP]), 1.2, 14)
                
                try:
                    assert_aae(pz[idxM], pz[idxP], 14)
                except:
                    assert_aae(np.abs(pz[idxM] - pz[idxP]), 1.0, 14)
