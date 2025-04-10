# 동작 메커님즘
# 컨베이어 동작 + 스톱센서 상승 + 이젝터 후진
# 스톱센서 감지 -> 이미지 처리를 통한 불량 반별(3가지 분류 : LED불량, Moter불량, 양품)
# LED불량 신호(이미지 처리) -> 서보모터(1번) 이동 후 이젝터 전진(PLC) -> 이젝터 전진 센터 &(and) LED불량신호면 -> 1번 창고 적재(Robot)
# Moter불량 신호(이미지 처리) -> 서보모터(2번) 이동 후 이젝터 전진(PLC) -> 이젝터 전진 센터 &(and) Moter불량신호면  -> 2번 창고 적재(Robot)
# 불량 공통 : 적재완료 신호(Robot) -> 다음 검사물품 푸쉬 -> 이하 반복
# 양품 : 컨베이어 동작
# 컨베이어 동작 + 스톱센서 상승 + 이젝터 후진 => PCL에서 동작 

#PLC 연결
# plc.py

import pymcprotocol
import time
import os

# PLC 연결을 모듈 레벨에서 한 번만!
PLC1 = pymcprotocol.Type3E()
PLC2 = pymcprotocol.Type3E()

PLC1.connect("192.168.3.150", 5010)
# PLC2.connect("192.168.3.160", 5010)

lock = None  # 필요하면 threading.Lock()

def handle_detection_result(camsignal):
    current_plc1 = PLC1
    current_plc2 = PLC2

    try:
        if camsignal == "l-off":
            print("LED_OFF")
            current_plc1.batchwrite_bitunits(headdevice="X24", values=[1])
            time.sleep(2)
            current_plc1.batchwrite_bitunits(headdevice="X24", values=[0])
        elif camsignal == "m-off":
            print("MOTOR_OFF")
            current_plc1.batchwrite_bitunits(headdevice="X22", values=[1])
            time.sleep(2)
            current_plc1.batchwrite_bitunits(headdevice="X22", values=[0])
        elif camsignal == "not":
            print("HOLE")
            current_plc1.batchwrite_bitunits(headdevice="X22", values=[1])  # 예시 주소
            time.sleep(2)
            current_plc1.batchwrite_bitunits(headdevice="X24", values=[0])
        elif camsignal == "l-on" or camsignal == "m-on":
            print("양품")
            current_plc1.batchwrite_bitunits(headdevice="X20", values=[1])
            time.sleep(2)
            current_plc1.batchwrite_bitunits(headdevice="X20", values=[0])
        else:
            print("Unknown 상태")
    except Exception as e:
        print(f"[PLC 통신 에러] {e}")

last_check_time = 0  # 마지막 Danger 체크 시간

def check_safe_danger_signal():
    global last_check_time
    current_plc1 = PLC1

    now = time.time()
    if now - last_check_time < 3:
        return  # 3초 안 지났으면 무시

    signal_file = "/home/yun/workspace/safe/safe_logs/danger_signal.txt"
    if os.path.exists(signal_file):
        print("[PLC] ⚠️ SAFE Danger 감지 → Y125 ON")

        try:
            current_plc1.batchwrite_bitunits(headdevice="X25", values=[1])
            time.sleep(3)
            current_plc1.batchwrite_bitunits(headdevice="X25", values=[0])
            print("[PLC] 🟢 Y125 OFF 처리 완료")
        except Exception as e:
            print("[PLC] 🚨 Y125 제어 중 오류:", e)

        os.remove(signal_file)
        last_check_time = now  # 마지막 실행 시간 갱신

    # PLC1.close()
    # PLC2.close()
