# ExaDGLM
프로젝트 'ExaDGLM'은 GPU를 포함한 가속기를 활용하여 시뮬레이션을 수행함에 Exa-Performance, Exa-Scalability을 지향함. <br>
프로젝트 'ExaDGLM'은 CPU에서 실행되는 시뮬레이션 모델들의 낮은 GPU 전환률과 GPU로 전환된 모델들도 CPU에서 실행되는 것에 비해서는 높은 성능을 가지고 있으나, <br>
GPU의 잠재성능에 비해서 낮은 연산효율을 가지는 것을 확인하고 이를 극복하고자 진행됨. <br>

프로젝트 'ExaDGLM'은 시뮬레이션 모델들의 낮은 GPU 전환률과 GPU 전환 후 기대성능에 도달하지 못하는 원인을 GPU의 잠재성능에 비해서 낮은 연산효율을 가지는 것으로 분석하였고, <br>
GPU에서 잠재성능에 비해서 낮은 연산효율을 보여주는 주된 이유는 기존에 주로 사용된 수치해석알고리즘이 가지고 있는 GPU 아키텍처에서 최적화되지 못한 특징으로 인해 <br>
낮은 연산밀도를 가지고 메모리 대역폭 한계로 성능이 제한 받게 되는 것으로 이를 극복하고자 진행됨. <br>
프로젝트 'ExaDGLM'은 수치해석알고리즘인 DGLM을 기반으로 시뮬레이션 실행에 필요한 핵심코어를 개발하여 공개함으로써 시뮬레이션 분야에서 GPU 사용이 가속화되기를 기대함. <br>

※ DGLM (Discontinuous Galerkin with Lagrange multiplier) <br>
: 도메인을 일정한 크기의 셀로 분할하고, 셀들 간의 불연속적인 해의 표현을 허용하는 특징을 가지고 <br>
각 셀 내에서는 고차 다항식으로 해를 근사, 셀 간 경계에서는 해의 불연속성을 처리하기 위해 수치적인 방법을 적용한 수치해석 알고리즘. <br>
요소 단위의 연산 독립성으로 병렬성이 높고, 미분연산자들이 Matrix-Matrix 연산으로 변환되어 연산밀도가 높아서 대규모 쓰레드 기반의 GPU 아키텍처에 적합한 특성을 가지고 있음. <br>
복잡한 기하학적 구조와 고차원 문제에도 적용 가능, 다양한 물리 현상의 모델링이 가능함. <br>

<br><br>
### Features
***
자체개발한 Explicit DGLM 기반 시뮬레이션 코어를 사용하여 기본 시뮬레이션 사례(Advection, Compressible Euler)를 GPU, CPU 버전을 성공적으로 구현하여 실행하였고 해당 소스코드를 제공함. <br>
Mesh는 수동작업을 통해 Uniform mesh 생성과 Mesh 생성 툴인 gmsh를 사용하여 다수의 파티션 생성(gmsh 버전은 다음 공개버전에서 제공), 시뮬레이션 실행 시 프로세서(GPU, CPU) 선택, 프로세서 사용 개수, Polynomial order 선택과 실행이 가능하도록 하였음. <br>
앞으로 Equation 추가 및 핵심도구를 개발하여 다양한 시뮬레이션 사례가 ExaDGLM 프로젝트 내에서 실행되도록 추가하고, <br>
9개 이상의 GPU를 사용하여 시뮬레이션이 실행될 수 있도록 Multi node 지원 기능을 제공할 계획임. <br>

<br><br>
### Getting Started
***
##### Prerequisites
프로젝트에 필요한 소프트웨어 및 사전 구성 방법

> OS : Ubuntu 22.04 <br>
> NVIDIA GPU Driver : 550.54.14 (525.60.13 이상) <br>
> Docker container ( https://docs.docker.com/engine/install/ubuntu ) <br>
> NVIDIA Container Toolkit ( https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html )

<br>

##### Installing
개발 환경 구성 및 설치 절차

* 컨테이너 빌드 ([dockerfile](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/dockerfile) 참고) <br>
```bash
$ docker build --network host --no-cache --tag= container_image .
```

* 컨테이너 실행
```bash
$ docker run -it --gpus all --cap-add=SYS_ADMIN --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name container_name -v host_directory:container_directory container_image /bin/bash &
```

* 컨테이너 접속
```bash
$ docker exec -it container_name /bin/bash
```

* Jupyter lab 비밀번호 설정과 실행
```bash
$ python3 -m jupyter_server.auth password
Provide password:
Repeat password:

$ jupyter-lab --notebook-dir=jupyter_running_directory &
```

* 웹에서 Jupyter 접속
```
http://jupyter_server_ip:Jupyter_port
```

<br>

##### Running the tests
사용법 및 테스트 안내 ( 0.4.1/dim3/examples/_ex#.case_name_ ) <br>

* _ex#.case_name_.01.defind.problem.ipynb <br>
문제 정의 <br>

* [_ex#.case_name_.11.cpu.single.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-cpu.md) <br>
Single CPU core를 사용한 1개 파티션 시뮬레이션 실행 및 후처리 <br>

* [_ex#.case_name_.12.cpu.partition.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-cpu.md) <br>
Multi CPU core를 사용한 2~3개 분할 파티션 시뮬레이션 실행 및 후처리 <br>

* [_ex#.case_name_.21.gpu.single.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-gpu.md) <br>
Single GPU를 사용한 1개 파티션 시뮬레이션 실행 및 후처리 <br>

* [_ex#.case_name_.22.gpu.partition.ipynb](https://github.com/ExaDGLM/ExaDGLM/blob/master/UserGuide/UserGuide-gpu.md) <br>
Multi GPU를 사용한 2~3개 분할 파티션 시뮬레이션 실행 및 후처리 <br>

※ Uniform tetrahedral mesh는 1 ~ 3개 파티션 시뮬레이션 실행 가능 (mesh 수동생성 시, 4개 이상 분할 가능) <br>
※ Gmsh는 1~8개 파티션 시뮬레이션 실행 가능 (현재 gmsh 라이선스 이슈로 다음 공개버전에서 업데이트 예정) <br>

<br><br>
### Versioning
***
버전/릴리즈 안내(이력 및 현재 버전 안내)

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
(프로젝트 컨트리뷰션 및 요청, 의견이 있는 경우 E-MAIL 로 문의) <br>
> Ki-Hwan Kim ( kihwanwb.kim@samsung.com ) <br>
> Jae-Min Shin ( jaemin8.shin@samsung.com )<br>
> Jae-Young Choi ( jyng.choi@samsung.com ) <br>
> Jin Yoo ( ujin.yoo@samsung.com ) <br>
> Ki-Hyo Moon ( kihyo.moon@samsung.com ) <br>

<br><br>
***
Copyright (C) 2023 Samsung SDS Co., Ltd.