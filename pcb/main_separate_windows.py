import cv2
import os
import time
import json
import numpy as np
from detector import ObjectDetector  # 모델로 감지해주는 클래스
from counter import DetectionCounter  # 감지 횟수를 세는 클래스
from visualizer import draw_boxes, draw_status, draw_counts  # 화면에 그려주는 함수들

# 🎥 영상 파일 사용 (기본)
VIDEO_PATH = "input_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

#비디오 파일을 연다
cap = cv2.VideoCapture(VIDEO_PATH)

# # 📸 실시간 캠 사용하려면 아래 주석 해제하세요
# cap = cv2.VideoCapture(0)

# # 카메라 인풋 포멧을 MJPG 로 설정
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
#     print("Failed to set FOURCC to MJPG.")

# # 카메라 해상도 설정
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# print(f"Resolution = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)},FPS = {cap.get(cv2.CAP_PROP_FPS)}")

# 창 우선순위
cv2.imshow("Status", np.zeros((200, 300, 3), dtype=np.uint8))
time.sleep(0.3)
os.system("wmctrl -r 'Status' -b add,above")


cv2.imshow("Counts", np.zeros((200, 300, 3), dtype=np.uint8))
time.sleep(0.3)
os.system("wmctrl -r 'Counts' -b add,above")

# 감지할 라벨(종류)
class_labels = ["m-on", "m-off", "l-on", "l-off", "not"]

# config.json에서 기준값(신뢰도 임계값)을 읽기
with open("config.json") as f:
    config = json.load(f)
CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.7)

# 객체 감지기와 카운터를 만든다
detector = ObjectDetector("openvino.xml", device="GPU")
counter = DetectionCounter(class_labels)

# 처음엔 WORKING 상태 (아무 것도 감지 안 됨)
status_text = "WORKING"
status_color = (255, 255, 0)  # 노란색
status_start_time = 0
status_hold_duration = 3  # 상태 유지 시간 (초)


while cap.isOpened():
    ret, frame = cap.read()

    # 영상이 끝나면 처음으로 돌아감 (반복 재생)
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_time = time.time()

    # 모델로 감지를 실행한다
    boxes_raw, labels_raw, scale, pad_left, pad_top = detector.infer(frame)

    combined = []
    detected_classes_in_frame = set()  # 현재 감지된 클래스들
    detected_now = None  # PASS 또는 NONPASS 저장할 변수

    # 감지된 박스들 하나씩 확인
    for i in range(len(boxes_raw)):
        if np.all(boxes_raw[i] == 0):  # 박스가 없다면 건너뜀
            continue
        x1, y1, x2, y2, conf = boxes_raw[i]
        class_id = int(labels_raw[i])
        class_name = class_labels[class_id]

        # l-on은 조금 더 점수 올려줌
        if class_id == 2:
            conf *= 1.1

        # 클래스별 기준 확인
        threshold = detector.custom_thresholds.get(class_id, CONFIDENCE_THRESHOLD)
        if conf < threshold:
            continue

        # 박스 좌표를 원래 크기로 다시 맞춤
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # 너무 작거나 이상한 위치는 건너뜀
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 5 or box_height < 5:
            continue
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        if class_id == 4 and (
            box_width > frame.shape[1] * 0.5 or box_height > frame.shape[0] * 0.5
        ):
            continue

        # 사용할 박스에 추가
        combined.append([x1, y1, x2, y2, conf, class_id])
        detected_classes_in_frame.add(class_name)

        # 어떤 상태인지 저장 (PASS 또는 NONPASS)
        if class_id in [0, 2]:  # m-on, l-on
            detected_now = "PASS"
        elif class_id in [1, 3, 4]:  # m-off, l-off, not
            detected_now = "NONPASS"

    # 상태를 변경할지 확인
    if detected_now:
        status_text = detected_now
        status_color = (0, 255, 0) if detected_now == "PASS" else (0, 0, 255)
        status_start_time = current_time
    elif current_time - status_start_time > status_hold_duration:
        status_text = "WORKING"
        status_color = (255, 255, 0)

    # 감지된 것들을 바탕으로 카운트 업데이트
    counter.update(detected_classes_in_frame)

    # 결과 화면 그리기
    result_img = draw_boxes(
        frame.copy(), combined, class_labels, threshold=CONFIDENCE_THRESHOLD
    )
    decision_img = draw_status(status_text, status_color)
    count_img = draw_counts(counter.class_counts)

    # 창 배치 조정
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.moveWindow("Detection", 0, 0)

    cv2.namedWindow("Status", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("Status", 1600, 100)

    cv2.namedWindow("Counts", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("Counts", 1600, 380)

    # 각각의 창에 따로 보여줌
    cv2.imshow("Detection", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))  # 박스 보여줌
    cv2.imshow("Status", decision_img)  # PASS/NONPASS
    cv2.imshow("Counts", count_img)  # 몇 번 감지됐는지

    # 각각의 창 Size
    # WINDOW_W1 = 1800
    # WINDOW_H1 = 800
    WINDOW_W2 = 352
    WINDOW_H2 = 240

    # cv2.resizeWindow("Detection", WINDOW_W1, WINDOW_H1)
    cv2.resizeWindow("Status", WINDOW_W2, WINDOW_H2)
    cv2.resizeWindow("Counts", WINDOW_W2, WINDOW_H2)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# 영상 끝나면 창 닫기
cap.release()
cv2.destroyAllWindows()
