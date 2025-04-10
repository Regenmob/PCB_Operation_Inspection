import numpy as np

class DetectionUtils:
    def __init__(self, model_width, model_height, class_labels):
        self.model_width = model_width
        self.model_height = model_height
        self.class_labels = class_labels

    def process_results(self, frame, boxes, labels, thresh=0.3):
        frame_height, frame_width = frame.shape[:2]
        ratio_x = frame_width / self.model_width
        ratio_y = frame_height / self.model_height

        boxes_array = boxes.squeeze()
        labels_array = labels.squeeze()
        detections_by_box = {}

        safety_count = 0
        unsafety_count = 0

        for box_vals, label_vals in zip(boxes_array, labels_array):
            xmin, ymin, xmax, ymax, score = box_vals
            if score >= thresh:
                box_pixels = tuple(map(int, (
                    xmin * ratio_x,
                    ymin * ratio_y,
                    (xmax - xmin) * ratio_x,
                    (ymax - ymin) * ratio_y,
                )))
                label = int(label_vals)
                if box_pixels not in detections_by_box:
                    detections_by_box[box_pixels] = []
                detections_by_box[box_pixels].append((label, float(score)))

        final_detections = []
        for box, scores in detections_by_box.items():
            best_label, best_score = max(scores, key=lambda x: x[1])
            final_detections.append((best_label, best_score, box))
            if best_label == 0:
                safety_count += 1
            elif best_label == 1:
                unsafety_count += 1

        return final_detections, safety_count, unsafety_count, safety_count + unsafety_count
