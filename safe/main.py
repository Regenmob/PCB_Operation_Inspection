
# from model_manager import ModelManager
# from detection_utils import DetectionUtils
# from ui_manager import UIManager
# import cv2
# import numpy as np

# MODEL_PATH = "/home/yun/workspace/safe/openvino.xml"
# VIDEO_PATH = "/home/yun/workspace/safe/1743484949.mp4"
# CLASS_LABELS = ["Safety", "Unsafety"]
# MODEL_WIDTH, MODEL_HEIGHT = 992, 736

# def main():
#     model = ModelManager(MODEL_PATH, device="GPU")
#     utils = DetectionUtils(MODEL_WIDTH, MODEL_HEIGHT, CLASS_LABELS)
#     ui = UIManager(CLASS_LABELS)

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print("캡처 실패")
#         return

#     cv2.namedWindow("results", cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
#         input_tensor = np.expand_dims(resized.transpose(2, 0, 1), axis=0)
#         boxes, labels = model.infer(input_tensor)

#         detections, s_count, u_count, total = utils.process_results(frame, boxes, labels)
#         processed = ui.draw_boxes(frame, detections)

#         cv2.imshow("results", processed)

#         results_rect = cv2.getWindowImageRect("results")
#         ui.display_count_info(s_count, u_count, results_rect)
#         status = ui.display_status(s_count, u_count, total, results_rect)

#         if status == "Danger" and not ui.is_fullscreen:
#             ui.original_window_size = results_rect
#             cv2.setWindowProperty("results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#             ui.is_fullscreen = True

#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#         elif key == ord("y"):
#             if ui.is_fullscreen and ui.original_window_size:
#                 cv2.setWindowProperty("results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
#                 x, y, w, h = ui.original_window_size
#                 cv2.moveWindow("results", x, y)
#                 cv2.resizeWindow("results", w, h)
#                 ui.is_fullscreen = False
#                 ui.original_window_size = None
#                 # Danger 상태도 해제
#                 ui.is_danger = False
#                 ui.danger_start_time = None

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
from model_manager import ModelManager
from detection_utils import DetectionUtils
from ui_manager import UIManager
import cv2
import numpy as np
import os
import time

MODEL_PATH = "/home/yun/workspace/safe/openvino.xml"
VIDEO_PATH = "/home/yun/workspace/safe/1743484949.mp4"
CLASS_LABELS = ["Safety", "Unsafety"]
MODEL_WIDTH, MODEL_HEIGHT = 992, 736

# Danger 상태 파일 저장 경로
DANGER_IMAGE_PATH = "safe_logs/danger.jpg"
DANGER_SIGNAL_FILE = "safe_logs/danger_signal.txt"

def main():
    model = ModelManager(MODEL_PATH, device="GPU")
    utils = DetectionUtils(MODEL_WIDTH, MODEL_HEIGHT, CLASS_LABELS)
    ui = UIManager(CLASS_LABELS)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("캡처 실패")
        return

    # 로그 디렉토리 생성
    os.makedirs("safe_logs", exist_ok=True)

    cv2.namedWindow("results", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("results", 0, 900) 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), axis=0)
        boxes, labels = model.infer(input_tensor)

        detections, s_count, u_count, total = utils.process_results(frame, boxes, labels)
        processed = ui.draw_boxes(frame, detections)

        cv2.imshow("results", processed)

        results_rect = cv2.getWindowImageRect("results")
        ui.display_count_info(s_count, u_count, results_rect)
        status = ui.display_status(s_count, u_count, total, results_rect)

        # Danger 상태 감지되면 로그 저장 및 신호 생성
        if status == "Danger" and not ui.is_fullscreen:
            ui.original_window_size = results_rect

            # 전체화면 전환
            cv2.setWindowProperty("results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            ui.is_fullscreen = True

            # danger 이미지 저장
            cv2.imwrite(DANGER_IMAGE_PATH, processed)

            # danger 신호 파일 생성
            with open(DANGER_SIGNAL_FILE, "w") as f:
                f.write("1")

            print("⚠️ Danger 상태 감지 → 이미지 저장 및 신호 파일 생성")

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("y"):
            if ui.is_fullscreen and ui.original_window_size:
                cv2.setWindowProperty("results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                x, y, w, h = ui.original_window_size
                cv2.moveWindow("results", x, y)
                cv2.resizeWindow("results", w, h)
                ui.is_fullscreen = False
                ui.original_window_size = None
                ui.is_danger = False
                ui.danger_start_time = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
