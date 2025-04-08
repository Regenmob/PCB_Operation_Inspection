import time

class DetectionCounter:
    def __init__(self, class_labels):
        self.class_labels = class_labels
        self.counts = {label: 0 for label in class_labels}  # 각 클래스별 감지 횟수 저장
        self.start_times = {label: None for label in class_labels}  # 감지가 시작된 시간
        self.cooldowns = {label: None for label in class_labels}  # 쿨다운(잠시 멈춤) 시간

    @property
    def class_counts(self):
        return self.counts

    def update(self, detected_classes):
        current_time = time.time()  # 현재 시간 가져오기
        for label in self.class_labels:
            if label in detected_classes:
                # 쿨다운 중이면 카운트를 안 올림
                if self.cooldowns[label] is not None and current_time - self.cooldowns[label] < 5:
                    continue
                # 감지가 처음 시작된 경우
                if self.start_times[label] is None:
                    self.start_times[label] = current_time
                # 1초 이상 감지가 되었으면 카운트를 올림
                elif current_time - self.start_times[label] >= 1:
                    self.counts[label] += 1  # 감지 횟수 증가
                    self.cooldowns[label] = current_time  # 쿨다운 시작
                    self.start_times[label] = None  # 감지 시작 시간 초기화
            else:
                self.start_times[label] = None  # 감지되지 않으면 초기화
