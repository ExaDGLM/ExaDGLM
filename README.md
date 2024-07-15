# ExaDGLM
The project 'ExaDGLM' aims for Exa-performance and Exa-Scalability by utilizing acceleration, including GPUs to perform simulations. <br>
The project 'ExaDGLM' is conducted to address issues of the low conversion rate from CPU to GPU in simulation models and the lower computational efficiency running on GPUs compared to their potential performance. <br>

The project 'ExaDGLM' has analyzed the low GPU conversion rates of simulation models and their inability to reach expected performance after conversion to GPU as being due to low computational efficiency compared to the potential performance of GPUs. <br> 
The primary reason for this low computational efficiency is that the traditional numerical analysis algorithms are not optimized for the GPU architecture, leading to low computation density and performance limitations by memory bandwidth constraint. <br> 
By developing and releasing the core components of the DGLM algorithm, the project 'ExaDGLM' aims to accelerate the use of GPUs in the field of simulations. <br>

<br>
※ DGLM (Discontinuous Galerkin with Lagrange multiplier) <br>
The algorithm has features to divide the domain into cells of a fixed size and allow for the representation of discontinuous solutions between cells. <br> 
Within each cell, solutions are approximated using high-order polynomials, and numerical methods are applied at cell boundaries to manage discontinuities. <br> <br>

This algorithm features element-wise computational independence resulting in high parallelism. Differential operators are transformed into Matrix-Matrix operations, increasing computational density, making it well-suited for large-scale, thread-based GPU architectures. <br> 
The algorithm is applicable to complex geometric structures and high-dimensional problems as well modeling of various physical phenomena. <br>

<br><br>
### Features
***
The implementation of the explicit DGLM in-house code successfully executed basic simulation cases (Advection, Compressible Euler) on both GPU and CPU version, and the source code is provided. <br>
Manual mesh generation is available for uniform mesh, and mesh generation tool, Gmsh, can be utilized for mesh partitioning (Gmsh version will be provided in the next release). Users can select the processor (GPU, CPU), the number of processors used, and the polynomial order. <br> 
In the future, we plan to add equations and develop core tools to enable various simulation cases to run within the ExaDGLM project. <br> 
Additionally, we plan to provide multi-node support to run simulation using more than nine GPUs. 

<br><br>
### Getting Started
***
##### Prerequisites
Softwares and pre-configurations

> OS : Ubuntu 22.04 <br>
> NVIDIA GPU Driver : 550.54.14 (525.60.13 or later) <br>
> Docker container ( https://docs.docker.com/engine/install/ubuntu ) <br>
> NVIDIA Container Toolkit ( https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html )

<br>

##### Installing
Environment setups and installation procedures

* Container build (cf. [dockerfile](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/dockerfile)) <br>
```bash
$ docker build --network host --no-cache --tag= container_image .
```

* Container execution
```bash
$ docker run -it --gpus all --cap-add=SYS_ADMIN --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name container_name -v host_directory:container_directory container_image /bin/bash &
```

* Container access
```bash
$ docker exec -it container_name /bin/bash
```

* Jupyter lab password setup and execution
```bash
$ python3 -m jupyter_server.auth password
Provide password:
Repeat password:

$ jupyter-lab --notebook-dir=jupyter_running_directory &
```

* Accessing Jupyter on the web
```
http://jupyter_server_ip:Jupyter_port
```

<br>

##### Running the tests
User guide for tests ( 0.4.1/dim3/examples/_ex#.case_name_ ) <br>

* _ex#.case_name_.01.defind.problem.ipynb <br>
Problem statement <br>

* [_ex#.case_name_.11.cpu.single.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-cpu.md) <br>
Simulation with a single CPU core and post-processing <br>

* [_ex#.case_name_.12.cpu.partition.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-cpu.md) <br>
Simulation with multi CPU cores (2-3 partiotioned) and post-processing <br>

* [_ex#.case_name_.21.gpu.single.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-gpu.md) <br>
Simulation with a single CPU core and post-processing <br>

* [_ex#.case_name_.22.gpu.partition.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-gpu.md) <br>
Simulation with multi CPU cores (2-3 partiotioned) and post-processing <br>

※ A uniform tetrahedral mesh can execute simulations with 1-3 partitions (with manual mesh generation, it can be partiotioned into 4 or more) <br>
※ Gmsh can execute simulations with 1-8 partitions (to be updated for the next public release due to Gmsh license issues)<br>

<br><br>
### Versioning
***
Version/Release (version history)

> Ver 0.4.1-OpenSource Explicit DGLM, Uniform tetrahedral mesh, CPU/GPU Single/Multi partition (Current) <br>

<br><br>
### License
***
SAMSUNG SDS Public License <br>
> [Samsung SDS Public License_kor](https://github.com/ExaDGLM/ExaDGLM/blob/master/license/Samsung%20SDS%20Public%20License_kor.md) <br>
> [Samsung SDS Public License_eng](https://github.com/ExaDGLM/ExaDGLM/blob/master/license/Samsung%20SDS%20Public%20License_eng.md) <br>

<br><br>
### Authors
***
High Performance Computing Research Lab / Technology Research / SAMSUNG SDS <br>
(For project contribution, request, or feedback, please contact us via email) <br>
> Ki-Hwan Kim ( kihwanwb.kim@samsung.com ) <br>
> Jaemin Shin ( jaemin8.shin@samsung.com )<br>
> Jae-Young Choi ( jyng.choi@samsung.com ) <br>
> Jin Yoo ( ujin.yoo@samsung.com ) <br>
> Kihyo Moon ( kihyo.moon@samsung.com ) <br>

<br><br>
***
Copyright (C) 2023 Samsung SDS Co., Ltd.
