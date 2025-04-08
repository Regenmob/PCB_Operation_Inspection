
import time

class DetectionCounter:
    def __init__(self, class_labels):
        self.class_labels = class_labels
        self.counts = {label: 0 for label in class_labels}  # 클래스별 감지 횟수
        self.start_times = {label: None for label in class_labels}  # 감지 시작 시간
        self.cooldowns = {label: None for label in class_labels}  # 쿨다운 시작 시간

    @property
    def class_counts(self):
        return self.counts

    def update(self, detected_classes):
        current_time = time.time()
        for label in self.class_labels:
            if label in detected_classes:
                # 쿨타임 중이면 카운트하지 않음
                if self.cooldowns[label] is not None and current_time - self.cooldowns[label] < 5:
                    continue
                # 감지 시작 시간 저장
                if self.start_times[label] is None:
                    self.start_times[label] = current_time
                # 1초 이상 감지되었으면 카운트 증가
                elif current_time - self.start_times[label] >= 1:
                    self.counts[label] += 1
                    self.cooldowns[label] = current_time
                    self.start_times[label] = None
            else:
                self.start_times[label] = None
