
import cv2
import time
import json
import numpy as np
from detector import ObjectDetector      # 모델 불러오는 모듈
from counter import DetectionCounter     # 감지된 객체 수 세는 모듈
from visualizer import draw_boxes, draw_status, draw_counts  # 화면에 보여주는 함수들

VIDEO_PATH = "input_video.mp4"  # 사용할 비디오 파일 이름

# 감지할 클래스 라벨 목록
class_labels = ["m-on", "m-off", "l-on", "l-off", "not"]

# 설정 파일에서 confidence threshold 값을 읽음
with open("config.json") as f:
    config = json.load(f)
CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.80)

# 모델과 카운터 초기화
detector = ObjectDetector("openvino.xml", device="GPU")
counter = DetectionCounter(class_labels)

# 처음에는 "작업중" 상태
status_text = "WORKING"
status_color = (255, 255, 0)  # 노란색
status_start_time = 0
status_hold_duration = 3  # 상태를 3초 동안 유지

# 비디오 파일 열기
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()

    # 만약 비디오가 끝났으면 처음으로 되돌아가기
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_time = time.time()

    # 객체 감지 수행
    boxes_raw, labels_raw, scale, pad_left, pad_top = detector.infer(frame)

    combined = []
    detected_classes_in_frame = set()
    detected_now = None  # 현재 PASS인지 NONPASS인지 저장

    for i in range(len(boxes_raw)):
        if np.all(boxes_raw[i] == 0):
            continue
        x1, y1, x2, y2, conf = boxes_raw[i]
        class_id = int(labels_raw[i])
        class_name = class_labels[class_id]

        # l-on일 때 신뢰도 높임
        if class_id == 2:
            conf *= 1.1

        # 클래스별 임계값 적용
        threshold = detector.custom_thresholds.get(class_id, CONFIDENCE_THRESHOLD)
        if conf < threshold:
            continue

        # 원래 비율로 좌표 되돌리기
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # 박스 크기, 위치 조건 확인
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 5 or box_height < 5:
            continue
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        if class_id == 4 and (box_width > frame.shape[1] * 0.5 or box_height > frame.shape[0] * 0.5):
            continue

        combined.append([x1, y1, x2, y2, conf, class_id])
        detected_classes_in_frame.add(class_name)

        # PASS 또는 NONPASS 판단
        if class_id in [0, 2]:  # m-on, l-on
            detected_now = "PASS"
        elif class_id in [1, 3, 4]:  # m-off, l-off, not
            detected_now = "NONPASS"

    # 상태 업데이트
    if detected_now:
        status_text = detected_now
        status_color = (0, 255, 0) if detected_now == "PASS" else (0, 0, 255)
        status_start_time = current_time
    elif current_time - status_start_time > status_hold_duration:
        status_text = "WORKING"
        status_color = (255, 255, 0)

    # 클래스별 카운트 업데이트
    counter.update(detected_classes_in_frame)

    # 화면에 보여줄 이미지 생성
    result_img = draw_boxes(frame.copy(), combined, class_labels, threshold=CONFIDENCE_THRESHOLD)
    decision_img = draw_status(status_text, status_color)
    count_img = draw_counts(counter.class_counts)

    # 각 이미지를 별도 창에 보여주기
    cv2.imshow("Detection", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))  # 감지 결과
    cv2.imshow("Status", decision_img)  # PASS / NONPASS 상태
    cv2.imshow("Counts", count_img)  # 클래스별 감지 횟수

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 비디오 종료 시 정리
cap.release()
cv2.destroyAllWindows()
