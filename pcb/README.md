
# 🎥 OpenVINO 기반 영상 객체 감지 시스템

이 프로젝트는 OpenVINO 추론 엔진을 활용하여 영상에서 특정 클래스의 객체를 감지하고,
PASS / NONPASS 판단 및 감지 횟수를 시각화하는 시스템입니다.

---
## 📁 프로젝트 구성

### 실행 파일
- `main.py`: 시스템 실행을 위한 메인 스크립트입니다. 영상에서 객체를 감지하고 결과를 출력합니다.

### 주요 모듈
- `detector.py`: OpenVINO 모델을 로드하고 추론을 수행하는 객체 감지기 클래스입니다.
- `counter.py`: 각 클래스에 대한 감지 횟수를 관리하는 클래스입니다.
- `visualizer.py`: 감지 결과 및 상태 정보를 화면에 출력하는 시각화 함수들을 포함합니다.

### 기타 파일
- `openvino.xml` / `openvino.bin`: OpenVINO IR(Intemediate Representation) 포맷의 객체 감지 모델 파일입니다.
- `config.json`: 클래스별 confidence threshold 설정이 포함된 설정 파일입니다.
- `input_video.mp4`: 감지 대상이 포함된 입력 영상 파일입니다. 파일명은 필요시 교체 가능합니다.
---

## 📌 주요 기능

- 🎯 객체 감지 (m-on, m-off, l-on, l-off, not)
- ⏱️ 1초 이상 감지 시 카운트 1 증가
- ❄️ 동일 객체 쿨다운(5초) 중복 감지 방지
- ✅ PASS / NONPASS 상태 판단 및 3초 유지
- 👁️ 결과를 실시간으로 3개 창(`Detection`, `Status`, `Counts`)에 시각화 출력
- 🔁 영상 자동 반복 재생
- 📦 OpenVINO IR 모델을 사용한 고속 추론

---

## 📁 프로젝트 구조

```
.
├── main.py                 # 메인 실행 파일 (영상 감지 수행)
├── detector.py             # 모델 로드 및 추론 처리
├── counter.py              # 감지 클래스별 카운트 관리
├── visualizer.py           # 시각화 함수 모음
├── config.json             # 클래스별 confidence threshold 설정
├── openvino.xml            # 모델 구조 (OpenVINO IR)
├── openvino.bin            # 모델 가중치 (OpenVINO IR)
├── input_video.mp4         # 감지 대상 영상 파일
├── requirements.txt        # Python 패키지 목록
└── README.md               # 프로젝트 설명 문서
```

---

## 🚀 실행 방법

### 1. OpenVINO 설치 (필수)
```bash
# 설치 (Ubuntu 기준)
pip install openvino-dev[onnx]
```

> 또는 인텔 공식 홈페이지 참고: https://docs.openvino.ai/

### 2. 의존 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
python main.py
```

> 기본적으로 `input_video.mp4` 파일을 입력으로 사용합니다.  
> 필요 시 다른 파일명으로 변경하거나, 코드의 `VIDEO_PATH`를 수정하세요.

---

## 🖥️ 카메라 입력으로 사용하려면?

`main.py`에서 이 부분만 수정하세요:

```python
cap = cv2.VideoCapture(0)  # ← 0번 웹캠을 사용
```

> 또는 `1`, `2`로 다른 카메라 번호 지정 가능

---

## 🧠 클래스 정의

| ID | 클래스 이름 | 설명         |
|----|--------------|--------------|
| 0  | m-on         | M램프 켜짐   |
| 1  | m-off        | M램프 꺼짐   |
| 2  | l-on         | L램프 켜짐   |
| 3  | l-off        | L램프 꺼짐   |
| 4  | not          | 결함 또는 기타 감지 |

---

## 🖼️ 출력창 설명

- `Detection`: 감지된 객체 표시 (박스 및 라벨)
- `Status`: PASS / NONPASS / WORKING 상태 표시
- `Counts`: 클래스별 누적 감지 횟수 표시

---

## ⚙️ 환경

- Python >= 3.8
- OpenCV >= 4.5
- OpenVINO >= 2023.0
- OS: Windows / Ubuntu (x86_64)

---


