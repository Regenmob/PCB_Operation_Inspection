import cv2
import os
import time
import json
import numpy as np
import threading

from detector import ObjectDetector
from counter import DetectionCounter
from visualizer import draw_boxes, draw_status, draw_counts
from plc import handle_detection_result, check_safe_danger_signal  # 🔥 추가!

# 실시간 캠 사용
cap = cv2.VideoCapture(0)

# MJPG 포맷 설정
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"Resolution = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, FPS = {cap.get(cv2.CAP_PROP_FPS)}")

#cv2.imshow("Status", np.zeros((200, 300, 3), dtype=np.uint8))
#cv2.imshow("Counts", np.zeros((200, 300, 3), dtype=np.uint8))

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow("Status", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow("Status", 1600, 0)
cv2.namedWindow("Counts", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow("Counts", 1600, 340)
cv2.resizeWindow("Status", 352, 240)
cv2.resizeWindow("Counts", 352, 240)
time.sleep(0.3)
os.system("wmctrl -r 'Status' -b add,above")
os.system("wmctrl -r 'Counts' -b add,above")

class_labels = ["m-on", "m-off", "l-on", "l-off", "not"]

with open("config.json") as f:
    config = json.load(f)
CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.7)

detector = ObjectDetector("openvino.xml", device="GPU")
counter = DetectionCounter(class_labels)

status_text = "WORKING"
status_color = (255, 255, 0)
status_start_time = 0
status_hold_duration = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_time = time.time()

    # 🔥 SAFE 시스템 Danger 감지 여부 확인
    check_safe_danger_signal()

    boxes_raw, labels_raw, scale, pad_left, pad_top = detector.infer(frame)
    combined = []
    detected_classes_in_frame = set()
    detected_now = None
    camsignal = None
    max_conf = 0

    for i in range(len(boxes_raw)):
        if np.all(boxes_raw[i] == 0):
            continue
        x1, y1, x2, y2, conf = boxes_raw[i]
        class_id = int(labels_raw[i])
        class_name = class_labels[class_id]

        if class_id == 2:
            conf *= 1.1

        threshold = detector.custom_thresholds.get(class_id, CONFIDENCE_THRESHOLD)
        if conf < threshold:
            continue

        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

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

        if conf > max_conf:
            max_conf = conf
            camsignal = class_name

        if class_id in [0, 2]:
            detected_now = "PASS"
        elif class_id in [1, 3, 4]:
            detected_now = "NONPASS"

    if detected_now:
        status_text = detected_now
        status_color = (0, 255, 0) if detected_now == "PASS" else (0, 0, 255)
        status_start_time = current_time
    elif current_time - status_start_time > status_hold_duration:
        status_text = "WORKING"
        status_color = (255, 255, 0)

    counter.update(detected_classes_in_frame)

    result_img = draw_boxes(frame.copy(), combined, class_labels, threshold=CONFIDENCE_THRESHOLD)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    decision_img = draw_status(status_text, status_color)
    count_img = draw_counts(counter.class_counts)



    cv2.imshow("Detection", cv2.resize(result_img, dsize=(1280, 720)))
    cv2.imshow("Status", decision_img)
    cv2.imshow("Counts", count_img)


    # camsignal이 존재하면 PLC로 전달
    if camsignal:
        print("▶ 감지된 결과:", camsignal)
        threading.Thread(target=handle_detection_result, args=(camsignal,), daemon=True).start()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
