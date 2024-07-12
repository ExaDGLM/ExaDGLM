### CPU Simulation version script
<br>
 
* User specification of 'DGLM_DIR, WORK_DIR' in set work directory <br>

```
DGLM_DIR = "/home/user/0.4.1-OpenSource"
WORK_DIR = "ex3.run.cpu.uniform.npart-3"
```

* User specification of polynomial order (N) in Initialize DGLM object <br>
  (cf. select polynomal order(N) 1~10) <br>
```
N    = 5
```

* User specification of partition(npart) <br>
  (cf. Single: 1, Partition 2-8(currnently uniform tetrahedral mesh applied 2~3)) <br>
```
mpisize = npart = 3
```

* Elapsed time can be checked once simulation executed by Run <br>
  (For partition version with multi core, number_cpu_cores can be modified) <br>
```
CPU Core 64개 사용

num_cpu_cores = 64
```

* Post-processing (Verification of model performance by L2 error, flowfield visualization) <br>
