
import cv2
import numpy as np

def draw_boxes(image, boxes, class_labels, threshold=0.5):
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
        color = (0, 255, 255) if class_id == 2 else (0, 255, 0)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return rgb

def draw_status(text, color):
    img = np.zeros((120, 300, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
    x = (img.shape[1] - text_size[0]) // 2
    y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    return img

def draw_counts(class_counts):
    img = np.ones((200, 300, 3), dtype=np.uint8) * 50
    for idx, (label, count) in enumerate(class_counts.items()):
        cv2.putText(img, f"{label}: {count}", (10, 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img
