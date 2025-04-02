현재 상황은 DETECTION은 되지만 정확성이 떨어짐.

(LED_OFF가 나와야 하는데 MOTOR_ON으로 인식중)

 train데이터 사진이 부족해서인지 인식했을때 %가 낮음

 



아래에 과정은 train방법 + export시 YOLOX 템플릿을 강제로 지정 하는 방법
#otx build Object_Detection_YOLO_X라고 명령어를 쳤지만,
#OTX가 자동으로 ATSS 기반 모델 템플릿을 선택했어.
#이유는 ./pcv_project/ 안의 데이터셋 구조가 COCO 형식이기 때문




# 1. 템플릿 찾기 (YOLOX 선택)
$ otx find --template --task DETECTION

# 2. YOLOX 빌드 (데이터셋 설정)
$ otx build Object_Detection_YOLO_X \
  --train-data-roots ../datumaro/export-coco/ \
  --val-data-roots ../datumaro/export-coco/

# 3. 학습
$ otx train params --learning_parameters.num_iters 8 \
  --learning_parameters.batch_size 1

# 4. HPO(하이퍼파라미터 튜닝)
$ otx train --enable-hpo

# 5. 성능 평가
$ otx eval --test-data-roots ../../datumaro/export-coco

# 6. 모델 내보내기 (.pth → export)
$ otx export --load-weights ./outputs/latest_trained_model/models/weights.pth \
  --output ./outputs/export

# 7. 최적화 (optional)
$ otx optimize --load-weights ./outputs/latest_trained_model/models/weights.pth \
  --output ./outputs/optimize/

# 8. 배포용 모델 생성 (OpenVINO IR 생성됨)
$ otx deploy --load-weights ./outputs/export/openvino.xml \
  --output ./outputs/deploy/


핵심: class_id 포함 여부는 어디에서 결정되냐?
 otx export 명령어에서 결정
네가 선택한 템플릿은 YOLOX가 아니라 "MobileNetV2-ATSS"야.
[*] Load Model Template ID: Custom_Object_Detection_Gen3_ATSS
[*] Load Model Name: MobileNetV2-ATSS
otx build Object_Detection_YOLO_X라고 명령어를 쳤지만,
OTX가 자동으로 ATSS 기반 모델 템플릿을 선택했어.
이유는 ./pcv_project/ 안의 데이터셋 구조가 COCO 형식이기 때문


이 문제 해결 방법: YOLOX 템플릿을 강제로 지정해주자
📌 1. 먼저 YOLOX 템플릿 ID 확인
otx find --template --task DETECTION
그럼 아래 같은 출력 중에 YOLOX 관련 템플릿 ID가 나올 거야:
ID:    Custom_Object_Detection_YOLOX
Name:  YOLOX

📌 2. YOLOX 템플릿을 명시적으로 사용해서 build 실행
otx build Custom_Object_Detection_YOLOX \
  --train-data-roots ./pcv_project/ \
  --val-data-roots ./pcv_project/


이후 단계는 동일
# 학습
otx train
# export (class_id 포함됨)
otx export --load-weights ./outputs/latest_trained_model/models/weights.pth \
  --output ./outputs/export

모델	    ./outputs/export/openvino.xml, openvino.bin ✅
class_id	[x1, y1, x2, y2, conf, class_id] 구조로 포함 ✅
라벨    	data.yaml에 정의되어 있음 ✅

위치 경로
"./otx-workspace-DETECTION/data.yaml"
"./outputs/export/openvino.xml"
