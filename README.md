# PCB_Operation_Inspection 

* A process of checking the operational status of PCB to identify functional errors or defects.
* It An AI system designed to monitor and verify the operational status of motors and LEDs on a PCB, detecting any functional errors or defects.

## High Level Design

graph TD
    A[Start] --> B[PCB 사진을 CVAT로 학습]
    B --> C[학습된 모델로 모터와 LED 상태 추적]
    C --> D[OpenCV로 카메라 영상 캡처]
    D --> E[모터 상태 모니터링]
    D --> F[LED 상태 모니터링]
    E --> G[모터 상태 분석]
    F --> H[LED 상태 분석]
    G --> I[상태 출력]
    H --> I[상태 출력]
    I --> J[End]


## Clone code

* (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)

```shell
git clone https://github.com/Regenmob/PCB_Operation_Inspection
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

* (프로젝트 실행 화면 캡쳐)

![./result.jpg](./result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
