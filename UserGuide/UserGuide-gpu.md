### GPU Simulation version script
<br>

* User specification of 'DGLM_DIR, WORK_DIR' in set work directory <br>
```
DGLM_DIR = "/home/user/0.4.1-OpenSource"
WORK_DIR = "ex3.run.gpu.uniform.npart-3"
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
  (For partition version, the number of GPUs can be selected accordeing to the number of partitions chosen in Initialize DGLM oject
  
```
GPU 3개 사용

mpisize = npart = 3
```

* Post-processing (Verification of model performance by L2 error, flowfield visualization) <br>
