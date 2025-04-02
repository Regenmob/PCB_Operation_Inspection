import cv2
import numpy as np
import openvino as ov

# ✅ 클래스 라벨 정의
class_labels = ["hole", "LED_OFF", "LED_ON", "MOTOR_ON", "MOTOR_OFF"]

# ✅ 모델 로딩 및 입력 설정
core = ov.Core()
model = core.read_model("/home/yun/workspace/project/pcb/otx-workspace-DETECTION/outputs/export/openvino.xml")
model.reshape({model.inputs[0].any_name: [1, 3, 736, 992]})
compiled_model = core.compile_model(model=model, device_name="GPU")
input_layer = compiled_model.input(0)
_, C, H, W = input_layer.shape

# ✅ NMS 함수 (겹치는 박스 제거)
def apply_nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes_array = np.array(boxes)
    coords = boxes_array[:, :4].tolist()
    scores = boxes_array[:, 4].tolist()
    indices = cv2.dnn.NMSBoxes(coords, scores, 0.5, iou_threshold)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

# ✅ 시각화 함수
def draw_boxes(image, boxes, threshold=0.5):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        if len(box) < 6:
            continue
        x1, y1, x2, y2, conf, class_id = box
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_id = int(class_id)
        label = f"{class_labels[class_id]} {int(conf * 100)}%"

        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return rgb

# ✅ 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (W, H))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

    result = compiled_model([input_tensor])
    boxes = result[compiled_model.output("boxes")].squeeze()
    labels = result[compiled_model.output("labels")].squeeze()

    # ✅ 박스 + 클래스 결합
    combined = []
    for i in range(len(boxes)):
        if np.all(boxes[i] == 0):
            continue
        x1, y1, x2, y2, conf = boxes[i]
        class_id = labels[i]
        combined.append([x1, y1, x2, y2, conf, class_id])

    # ✅ NMS 적용 (겹치는 박스 제거)
    combined = apply_nms(combined)

    # ✅ 시각화 및 출력
    output_img = draw_boxes(resized, combined, threshold=0.5)
    output_img = cv2.resize(output_img, (frame.shape[1], frame.shape[0]))
    cv2.imshow("Detection", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
