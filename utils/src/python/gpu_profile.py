import sys
import numpy as np
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

NCU_DIR = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/bin"

def calc_ref(N, Nelem, Neq, Nface=4):
    '''
    calculate reference FLOP and bytes
    '''
    Np  = (N+1)*(N+2)*(N+3)//6
    Nfp = (N+1)*(N+2)//2
    NFF = Nface*Nfp
    
    ref_flop = {}
    ref_byte = {}
    
    # FLOP
    ref_flop['WDr']  = (2*Np-1)*Np*Nelem*Neq
    ref_flop['WDs']  = (2*Np  )*Np*Nelem*Neq
    ref_flop['WDt']  = (2*Np  )*Np*Nelem*Neq
    ref_flop['LIFT'] = (2*NFF )*Np*Nelem*Neq
    
    # bytes (min, max)
    ref_byte['WDr']  = {'min':Np*(Np+2*Nelem)*Neq*8, 'max':(2*Np+1)*Np*Nelem*Neq*8}
    ref_byte['WDs']  = {'min':Np*(Np+3*Nelem)*Neq*8, 'max':(2*Np+2)*Np*Nelem*Neq*8}
    ref_byte['WDt']  = {'min':Np*(Np+3*Nelem)*Neq*8, 'max':(2*Np+2)*Np*Nelem*Neq*8}
    ref_byte['LIFT'] = {'min':(NFF*(Np+Nelem)+2*Np*Nelem)*Neq*8, 'max':(2*NFF+2)*Np*Nelem*Neq*8}
    
    return ref_flop, ref_byte


class NsightCompute:
    def __init__(self, target_file, ncu_dir=NCU_DIR, verbose=2):
        self.verbose = verbose  # print level
        
        #
        # prepare ncu_report.py
        #
        #self.ncu_python_dir = ncu_dir + "/extras/python"
        self.ncu_python_dir = ncu_dir + "/extras/python"
        sys.path.append(self.ncu_python_dir)
        import ncu_report as ncu        
        
        self.ctx = ncu.load_report(target_file)
        self.group = self.ctx.range_by_idx(0)
        
        # device information
        self.mv_dev = self.get_device_info()
        
        
    def get_device_info(self):
        metric_dict = {
            'dev_name'             : 'device__attribute_display_name',
            'sm_count'             : 'device__attribute_multiprocessor_count',
            'max_clock_speed'      : 'device__attribute_max_gpu_frequency_khz',
            'clock_speed'          : 'device__attribute_clock_rate',        
            'mem_clock_speed'      : 'device__attribute_memory_clock_rate',
            'mem_bus_width'        : 'device__attribute_global_memory_bus_width',
        }
        
        # get metric values
        ncu_kernel = self.group.action_by_idx(0)
        mv = {name: ncu_kernel[k].value() for name, k in metric_dict.items()}
        
        # peak FLOPs: num_cores*clock_speed*fma
        num_cores = mv["sm_count"]*4*8   # 4 sub partition, 8 fp64 cores
        clock_speed = mv["clock_speed"]  # Khz
        fma = 2
        mv['peak_flops'] = num_cores*clock_speed*fma
        
        # peak mem bandwidth: clock_speed*(bus_width/8)*ddr
        mem_clock_speed = mv["mem_clock_speed"]
        mem_bus_width = mv["mem_bus_width"]  # bytes unit
        ddr = 2
        mv['peak_mem_bw'] = mem_clock_speed*(mem_bus_width/8)*ddr
                
        if self.verbose >= 1:
            print(f"{mv['dev_name']}")
            
        if self.verbose >= 2:
            print(f"sm_count        : {mv['sm_count']}")
            print(f"max_clock_speed : {mv['max_clock_speed']/1e6:.3f} GHz")
            print(f"clock_speed     : {mv['clock_speed']/1e6:.3f} GHz")        
            print(f"mem_clock_speed : {mv['mem_clock_speed']/1e6:.3f} GHz")
            print(f"mem_bus_width   : {mv['mem_bus_width']} bit")
        
        if self.verbose >= 1:
            print()
            print(f"peak FLOPs : {mv['peak_flops']/1e9:.3f} TFLOPs")
            print(f"peak BW    : {mv['peak_mem_bw']/1e9:.3f} TB/s")                
            print()
        
        return mv


    def get_kernel_metrices(self, ncu_kernel):
        '''
        Metric prefix (https://docs.nvidia.com/nsight-compute/2022.2/ProfilingGuide/index.html)
        
        gpu__ : The entire Graphics Processing Unit
        gpc__ : The General Processing Cluster contains SM, Texture and L1 in the form of TPC(s).
        tpc__ : Thread Processing Clusters are units in the GPC. They contain one or more SM,
                Texture and L1 units, the Instruction Cache (ICC) and the Indexed Constant Cache (IDC).
        sm__  : The Streaming Multiprocessor handles execution of a kernel as groups of 32 threads, called warps.
        smsp__: Each SM is partitioned into 4 processing blocks, called SM sub partitions.
                The SM sub partitions are the primary processing elements on the SM.
                A sub partition manages a fixed size pool of warps.

        A100: sm=80, smsp=4
        '''
                
        metric_dict = {
            'time_duration'        : 'gpu__time_duration.sum',             # 총 수행시간 (nano second 단위)
            'cycles.avg/s'         : 'sm__cycles_elapsed.avg.per_second',  # 초당 평균 cycle 수
            'cycles.sum'           : 'smsp__cycles_elapsed.sum',           # 모든 sub-partition (processing block)에서 수행된 cycle 수의 총합
            'compute_throughput'   : 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'memory_throughput'    : 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
            'dram_throughput'      : 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
            'dram_read_bytes'      : 'dram__bytes_read.sum',
            'dram_write_bytes'     : 'dram__bytes_write.sum',
            'dram_read_throughput' : 'dram__bytes_read.sum.per_second',
            'dram_write_throughput': 'dram__bytes_write.sum.per_second',        
            'flop_dp_add.sum/cycle': 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed',
            'flop_dp_mul.sum/cycle': 'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed',
            'flop_dp_fma.sum/cycle': 'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed',
            'flop_sp_add.sum/cycle': 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed',
            'flop_sp_mul.sum/cycle': 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed',
            'flop_sp_fma.sum/cycle': 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed',
        }
    
        # get metric values
        mv = {name: ncu_kernel[k].value() for name, k in metric_dict.items()}
        
        # derived metric values
        dmv = {}
        dmv['time']       = mv['time_duration']*1e-9  # nano second
        dmv['cycles']     = mv['cycles.avg/s']*dmv['time']                
        dmv['flop/cycle'] = mv['flop_dp_add.sum/cycle'] + mv['flop_dp_mul.sum/cycle'] + mv['flop_dp_fma.sum/cycle']*2
        dmv['flop']       = dmv['flop/cycle']*dmv['cycles']
        dmv['flop/s']     = dmv['flop/cycle']*mv['cycles.avg/s']        
        dmv['byte']       = mv['dram_read_bytes'] + mv['dram_write_bytes'] 
        dmv['byte/s']     = dmv['byte']/dmv['time']
        dmv['flop/byte']  = dmv['flop']/dmv['byte']
        
        dmv['flop2'] = (mv['compute_throughput']/100)*self.mv_dev['peak_flops']*1e3*dmv['time']
        dmv['byte2'] = (mv['memory_throughput']/100)*self.mv_dev['peak_mem_bw']*1e3*dmv['time']
        
        # check consistency
        assert_aae(dmv['cycles'], mv['cycles.sum']/(self.mv_dev['sm_count']*4))
        assert_aae(dmv['byte/s']/1e12, (mv['dram_read_throughput'] + mv['dram_write_throughput'])/1e12, 14)
        
        return mv, dmv


    def get_kernel_metrices_avg(self, kernel_names1, kernel_names2, tstep=10, rk_stage=5):
        metric_names = ['time', 'flop', 'byte', 'flop2', 'byte2']
                
        metric_db = {}
        for k_name in kernel_names1:
            metric_db[k_name] = {m: np.zeros(tstep, 'f8') for m in metric_names}            

        for k_name in kernel_names2:
            metric_db[k_name] = {m: np.zeros(tstep*rk_stage, 'f8') for m in metric_names}
        
        #
        # exclude init kernels
        #
        num_kernels = self.group.num_actions()
        
        for n in range(num_kernels):
            ncu_kernel = self.group.action_by_idx(n)
            if ncu_kernel.name() == kernel_names1[0]:
                start_n = n
                break

        #
        # extract metrices
        #
        tidx = -1  # time step
        sidx = -1  # rk sub-stage
        for n in range(start_n, num_kernels):
            ncu_kernel = self.group.action_by_idx(n)
            ncu_kname = ncu_kernel.name()
            mv, dmv = self.get_kernel_metrices(ncu_kernel)
            
            if ncu_kname == kernel_names1[0]:
                tidx += 1
                rk_loop = False
                
            elif ncu_kname == kernel_names2[0]:
                sidx += 1
                rk_loop = True
                
            for m_name in metric_names:
                if ncu_kname in kernel_names1:
                    metric_db[ncu_kname][m_name][tidx] = dmv[m_name]
                    prev_kname = ncu_kname
                    
                elif ncu_kname in kernel_names2:
                    metric_db[ncu_kname][m_name][sidx] = dmv[m_name]
                    prev_kname = ncu_kname
                    
                else:
                    if not rk_loop:
                        k_name = kernel_names1[kernel_names1.index(prev_kname) + 1]
                        metric_db[k_name][m_name][tidx] += dmv[m_name]
                    else:
                        k_name = kernel_names2[kernel_names2.index(prev_kname) + 1]
                        metric_db[k_name][m_name][sidx] += dmv[m_name]

        #
        # average metrices per time step
        #
        avg_mdb = {}
        for k_name in kernel_names1 + kernel_names2:
            avg_mdb[k_name] = {}
            
            for m_name in metric_names:
                avg_mdb[k_name][m_name] = metric_db[k_name][m_name].mean()
                
                if k_name in kernel_names2:
                    avg_mdb[k_name][m_name] *= rk_stage
            
        #
        # total metrics per time step
        #
        for m_name in metric_names:            
            key1 = f'total_{m_name}'
            key2 = f'gemm_{m_name}'
            
            avg_mdb[key1] = 0
            avg_mdb[key2] = 0
            
            for k_name in kernel_names1 + kernel_names2:
                avg_mdb[key1] += avg_mdb[k_name][m_name]
                
                if 'gemm' in k_name:
                    avg_mdb[key2] += avg_mdb[k_name][m_name]
        
        return avg_mdb
    
    
    def calc_roofline(self, avg_mdb, k_name, use_throughput=True):
        '''
        k_name : kernel name, 'total', 'gemm'
        '''
        fname = 'flop2' if use_throughput else 'flop'
        bname = 'byte2' if use_throughput else 'byte'
        
        if k_name in ['total', 'gemm']:
            time = avg_mdb[f'{k_name}_time']
            flop = avg_mdb[f'{k_name}_{fname}']
            byte = avg_mdb[f'{k_name}_{bname}']
        else:
            time = avg_mdb[k_name]['time']
            flop = avg_mdb[k_name][fname]
            byte = avg_mdb[k_name][bname]
        
        ai = flop/byte  # arithmetic intensity
        tflops = flop/time/1e12  # TFLOP/s
        tbytes = byte/time/1e12  # TBytes            
        tflops_pct = tflops/(self.mv_dev['peak_flops']/1e9)*100
        tbytes_pct = tbytes/(self.mv_dev['peak_mem_bw']/1e9)*100
        time_pct = time/avg_mdb['total_time']*100
        
        return time, flop, byte, ai, tflops, tbytes, tflops_pct, tbytes_pct, time_pct
        
        
    def print_roofline_metrices(self, avg_mdb, kernel_names, use_throughput=True):        
        def to_unit(val):
            i = 0
            while val >= 100:
                val, i = val/1e3, i+1            
            unit = {0:' ', 1:'K', 2:'M', 3:'G', 4:'T'}[i]
            return val, 1e3**i, unit
        
        fname = 'flop2' if use_throughput else 'flop'
        bname = 'byte2' if use_throughput else 'byte'            
        
        max_len = 0
        max_flop = 0
        max_byte = 0
        for k_name in kernel_names:
            max_len = max(max_len, len(k_name))
            max_flop = max(max_flop, avg_mdb[k_name][fname])
            max_byte = max(max_byte, avg_mdb[k_name][bname])
            
        _, ef, uf = to_unit(max_flop)
        _, eb, ub = to_unit(max_byte)
        
        msg = "by throughput" if use_throughput else ""
        print(f"\nAverage metrices (10 time steps -> 1 time step) {msg}")
        print(f"{'kernel name'.ljust(max_len)}:    {uf}FLOP    {ub}Byte  FLOP/Byte     TFLOP/s(%)     TByte/s(%)    time[ms](%)")
        print("-"*89)            

        for k_name in kernel_names:
            time, flop, byte, ai, tflops, tbytes, tflops_pct, tbytes_pct, time_pct = \
                self.calc_roofline(avg_mdb, k_name, use_throughput)
            print(f"{k_name.ljust(max_len)}: {flop/ef:>8.3f} {byte/eb:>8.3f} {ai:>10.2f} {tflops:>8.3f}({tflops_pct:>4.1f}) {tbytes:>8.3f}({tbytes_pct:>4.1f}) {time*1e3:>8.3f}({time_pct:>4.1f})")

        print("-"*89)
        time, flop, byte, ai, tflops, tbytes, tflops_pct, tbytes_pct, time_pct = \
                self.calc_roofline(avg_mdb, 'total', use_throughput)
        print(f"{'total'.ljust(max_len)}: {flop/ef:>8.3f} {byte/eb:>8.3f} {ai:>10.2f} {tflops:>8.3f}({tflops_pct:>4.1f}) {tbytes:>8.3f}({tbytes_pct:>4.1f}) {time*1e3:>8.3f}")
        
        time, flop, byte, ai, tflops, tbytes, tflops_pct, tbytes_pct, time_pct = \
                self.calc_roofline(avg_mdb, 'gemm', use_throughput)
        print(f"{'gemm'.ljust(max_len)}: {flop/ef:>8.3f} {byte/eb:>8.3f} {ai:>10.2f} {tflops:>8.3f}({tflops_pct:>4.1f}) {tbytes:>8.3f}({tbytes_pct:>4.1f}) {time*1e3:>8.3f}({time_pct:>4.1f})")
