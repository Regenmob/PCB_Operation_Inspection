
import cv2
import numpy as np
import openvino as ov
import json

class ObjectDetector:
    def __init__(self, model_path, device="CPU"):
        with open("config.json") as f:
            self.config = json.load(f)
        self.conf_threshold = self.config.get("confidence_threshold", 0.5)
        self.class_labels = ["m-on", "m-off", "l-on", "l-off", "not"]
        self.custom_thresholds = {4: 0.8, 2: 0.4}

        # 모델 불러오기
        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)

        # 입력 정보 설정
        self.input_layer = self.compiled_model.input(0)
        shape = self.input_layer.partial_shape
        _, self.C, self.H, self.W = [dim.get_length() if dim.is_static else 1 for dim in shape]

        outputs = self.compiled_model.outputs
        self.output_box = outputs[0].any_name
        self.output_label = outputs[1].any_name

    def letterbox_resize(self, image, target_size=(416, 416), pad_value=114):
        # 이미지 비율 유지하며 패딩 추가해서 resize
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (nw, nh))
        top = (target_size[0] - nh) // 2
        bottom = target_size[0] - nh - top
        left = (target_size[1] - nw) // 2
        right = target_size[1] - nw - left
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
        return image_padded, scale, left, top

    def infer(self, frame):
        # 추론을 위한 전처리 후 모델 실행
        resized, scale, pad_left, pad_top = self.letterbox_resize(frame, (self.H, self.W))
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
        result = self.compiled_model([input_tensor])
        boxes = result[self.output_box].squeeze()
        labels = result[self.output_label].squeeze()
        return boxes, labels, scale, pad_left, pad_top
