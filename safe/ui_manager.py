
import cv2
import numpy as np
import time
import os

class UIManager:
    def __init__(self, class_labels, count_size=(150, 100), status_size=(200, 100)):
        self.class_labels = class_labels
        self.INFO_COUNT_WIDTH, self.INFO_COUNT_HEIGHT = count_size
        self.INFO_STATUS_WIDTH, self.INFO_STATUS_HEIGHT = status_size
        self.danger_start_time = None
        self.is_danger = False
        self.is_fullscreen = False
        self.original_window_size = None

        # 창 미리 띄우고 waitKey로 시스템에 등록
        cv2.namedWindow("Detection Status", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow("Detection Count", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Detection Count", np.zeros((self.INFO_COUNT_HEIGHT, self.INFO_COUNT_WIDTH, 3), dtype=np.uint8))
        cv2.imshow("Detection Status", np.zeros((self.INFO_STATUS_HEIGHT, self.INFO_STATUS_WIDTH, 3), dtype=np.uint8))
        cv2.resizeWindow("Detection Status", 352, 240)
        cv2.resizeWindow("Detection Count", 352, 240)
        cv2.waitKey(1)
        os.system("wmctrl -r 'Detection Count' -b add,above")
        os.system("wmctrl -r 'Detection Status' -b add,above")

    def draw_boxes(self, frame, detections):
        colors = {0: (10, 240, 30), 1: (10, 30, 240)}
        for label, score, box in detections:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(label, (255,255,255)), 2)
            cv2.putText(frame, f"{self.class_labels[label]} {score:.2f}", (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(label, (255,255,255)), 1)
        return frame

    def display_count_info(self, safety, unsafety, results_rect):
        # 고정 좌표로 설정
        cv2.moveWindow("Detection Count", 1600, 525)

        img = np.zeros((self.INFO_COUNT_HEIGHT, self.INFO_COUNT_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, f"Safety: {safety}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Unsafety: {unsafety}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imshow("Detection Count", img)


    def display_status(self, safety, unsafety, total, results_rect):
        # 고정 좌표로 설정
        cv2.moveWindow("Detection Status", 1600, 900)

        # ... 이하 원래 코드 그대로 유지

        img = np.zeros((self.INFO_STATUS_HEIGHT, self.INFO_STATUS_WIDTH, 3), dtype=np.uint8)
        status_text = "Normal"
        status_color = (0, 255, 0)

        if total > 0:
            safety_pct = (safety / total) * 100
            unsafety_pct = (unsafety / total) * 100

            if 30 <= safety_pct <= 40:
                status_text, status_color = "Caution", (0, 255, 255)
            elif safety_pct < 30 or unsafety_pct > 50:
                if not self.is_danger:
                    self.danger_start_time = time.time()
                    self.is_danger = True
                elif time.time() - self.danger_start_time > 10:
                    status_text, status_color = "Danger", (0, 0, 255)
            else:
                self.is_danger = False
                self.danger_start_time = None

        cv2.putText(img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        cv2.imshow("Detection Status", img)
        return status_text
