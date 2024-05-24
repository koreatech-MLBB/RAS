import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANFORMS"] = "0"

import sys
openpose_dir = "D:\RAS\openpose\Release"
model_dir = "D:\RAS\models"
sys.path.append(openpose_dir)

try:
    import pyopenpose as op
except ImportError as e:
    print(e)
    raise e

pose_path = "D:\\RAS\\openpose\\Release"
model_path = "D:\\RAS\\models"

from multiprocessing import Process, Pipe, shared_memory
import cv2

from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.event import RotateEvent
from pymysqlreplication.row_event import WriteRowsEvent, DeleteRowsEvent

from scipy.signal import find_peaks

from pose_functions import *


def run_api_server():
    try:
        os.system("python ./ras/manage.py runserver 0.0.0.0:8080")
    except BaseException as e:
        print(f"run_api_server.except: {e.__str__()}")
        print("run_api_server terminated")


def running_process(my_name, read_handle):

    # cap = cv2.VideoCapture("./examples/run_example_4.mp4")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, test_frame = cap.read()
        if ret:
            break
    print(test_frame.shape)
    buf1 = shared_memory.SharedMemory(name=f"{my_name}_frame",
                                   create=True,
                                   size=test_frame.shape[0] * test_frame.shape[1] * test_frame.shape[2])
    # print(_.name)
    b = np.array(["a" * 40], dtype=np.dtype('U40'))
    buf2 = shared_memory.SharedMemory(name=f"{my_name}_feedback",
                                   create=True,
                                   size=b.nbytes)
    # print(_.name)
    op_wrapper = op.WrapperPython()
    op_wrapper.configure({"model_folder": model_dir})
    op_wrapper.start()

    body_coordinates = dict()
    body = {"Nose": (0,), "Neck": (1,), "Mid-hip": (8,),
            "Eye": (15, 16), "Ear": (17, 18), "Shoulder": (2, 5), "Elbow": (3, 6), "Wrist": (4, 7),
            "Hip": (9, 12), "Knee": (10, 13), "Ankle": (11, 14), "Big-toe": (22, 19)}

    for key, values in body.items():
        if len(values) == 1:
            body_coordinates[key] = [[]]
        else:
            body_coordinates[key] = [[], []]

    frames = [0 for _ in range(100)]
    frames.clear()

    while True:
        if read_handle.poll(timeout=0.01):
            data = read_handle.recv()
            if data == "kill":
                cap.release()
                print(f"kill this({my_name}) process")
                break
        # cam에서 frame정보 읽어오기
        ret, frame = cap.read()

        if not ret:
            continue
        try:
            datum = op.Datum()
            imageToProcess = frame
            datum.cvInputData = imageToProcess
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

            # shared_memory 저장 부분
            my_frame = shared_memory.SharedMemory(name=f"{my_name}_frame")
            frame = np.array(datum.cvOutputData)
            my_frame_buf = np.ndarray(frame.shape, dtype=np.uint8, buffer=my_frame.buf)
            np.copyto(my_frame_buf, frame)
            my_frame.close()

            for k, idx in body.items():
                for i in range(len(idx)):
                    body_coordinates[k][i].append(tuple(datum.poseKeypoints[0][idx[i]][:2]))

            frames.append(datum.cvOutputData)

            nose_position = -1 * np.array([y for _, y in body_coordinates['Neck'][0]])
            peaks, _ = find_peaks(nose_position)
            if len(peaks) == 2:
                target = 1 if body_coordinates['Ankle'][0][0] > body_coordinates['Ankle'][1][0] else 0

                knee_function = lambda x: (
                        (1.3148 * (0.1 ** 8)) * (x ** 5)
                        - (3.3236 * (0.1 ** 6)) * (x ** 4)
                        + 0.0002891 * (x ** 3)
                        - 0.010055 * (x ** 2)
                        + 0.12522 * x
                )

                hip_function = lambda x: (
                        (-4.1847 * (0.1 ** 14)) * (x ** 8)
                        + (1.6414 * (0.1 ** 11)) * (x ** 7)
                        - (2.484 * (0.1 ** 9)) * (x ** 6)
                        + (1.8194 * (0.1 ** 7)) * (x ** 5)
                        - (6.8429 * (0.1 ** 6)) * (x ** 4)
                        + 0.0001503 * (x ** 3)
                        - 0.0024657 * (x ** 2)
                        + 0.0020494 * x
                        + 0.91
                )

                ankle_function = lambda x: (
                        1.8553 * (0.1 ** 14) * (x ** 8)
                        - 1.4506 * (0.1 ** 11) * (x ** 7)
                        + 3.916 * (0.1 ** 9) * (x ** 6)
                        - 5.0626 * (0.1 ** 7) * (x ** 5)
                        + 3.4058 * (0.1 ** 5) * (x ** 4)
                        - 0.0011533 * (x ** 3)
                        + 0.016508 * (x ** 2)
                        - 0.059354 * x
                        + 0.68168
                )

                knee_angles = []
                hip_angles = []
                ankle_angles = []
                gaze_scores = []
                upper_body_scores = []

                elbow_scores = elbow_angle_calc_scores(shoulder=body_coordinates['Shoulder'],
                                                       elbow=body_coordinates['Elbow'],
                                                       wrist=body_coordinates['Wrist'],
                                                       length=peaks[-1] + 1)

                for i in range(peaks[-1] + 1):
                    # 무릎 각도
                    knee_angle = calculate_angle(a=body_coordinates['Hip'][target][i],
                                                 b=body_coordinates['Knee'][target][i],
                                                 c=body_coordinates['Ankle'][target][i])
                    knee_angles.append(180 - knee_angle)

                    # 골반 각도
                    hip_angle = calculate_angle(a=body_coordinates['Neck'][0][i],
                                                b=body_coordinates['Hip'][target][i],
                                                c=body_coordinates['Knee'][target][i])
                    hip_angles.append(180 - hip_angle)

                    # 발목 각도
                    ankle_angle = calculate_angle(a=body_coordinates['Big-toe'][target][i],
                                                  b=body_coordinates['Ankle'][target][i],
                                                  c=body_coordinates['Knee'][target][i])
                    ankle_angles.append(90 - ankle_angle)

                    # 시선
                    gaze_score = inclination_to_degree(calc_inclination(dot_a=body_coordinates['Eye'][1][i],
                                                                        dot_b=body_coordinates['Ear'][1][i]))
                    gaze_scores.append(100 if -15 <= gaze_score <= 15 else 0)

                    # 상체
                    upper_body_socre = inclination_to_degree(calc_inclination(dot_a=body_coordinates['Neck'][0][i],
                                                                              dot_b=body_coordinates['Mid-hip'][0][i],
                                                                              reversed=True))
                    upper_body_scores.append(100 if -10 <= upper_body_socre <= 10 else 0)

                knee_angles = normalize_list(knee_angles)
                hip_angles = normalize_list(hip_angles)
                ankle_angles = normalize_list(ankle_angles)

                x = np.linspace(0, 100, peaks[-1] + 1)
                target_knee_angles = [knee_function(val) for val in x]
                target_hip_angles = [hip_function(val) for val in x]
                target_ankle_angles = [ankle_function(val) for val in x]

                knee_score = round(cosine_similarity(target_knee_angles, knee_angles) * 100, 1)
                hip_score = round(cosine_similarity(target_hip_angles, hip_angles) * 100, 1)
                ankle_score = round(cosine_similarity(target_ankle_angles, ankle_angles) * 100, 1)
                gaze_score = round(sum(gaze_scores) / (peaks[-1] + 1), 1)
                elbow_score = round(sum(elbow_scores) / (peaks[-1] + 1), 1)
                upper_body_score = round(sum(upper_body_scores) / (peaks[-1] + 1), 1)
                target = 'left' if target else 'right'
                print(f"{target}_knee_score: {knee_score}, "
                      f"{target}_hip_score: {hip_score}, "
                      f"{target}_ankle_score: {ankle_score}, "
                      f" gaze: {gaze_score}, "
                      f" elbow: {elbow_score}, "
                      f" upper_body: {upper_body_score}")

                for k, idx in body.items():
                    for i in range(len(idx)):
                        body_coordinates[k][i] = body_coordinates[k][i][peaks[0] + 1:]
                frames = frames[peaks[0] + 1:]



                my_audio = shared_memory.SharedMemory(name=f"{my_name}_feedback")
                feedback = np.array(["tteesstt" + my_name])
                my_audio_buf = np.ndarray(shape=feedback.shape, dtype=np.dtype('U40'), buffer=my_audio.buf)
                np.copyto(my_audio_buf, feedback)
                my_audio.close()

        except BaseException as e:
            print("in main running process")
            print(e)
            continue


if __name__ == "__main__":

    # 모든 프로세스가 담길 리스트
    processes = []

    # Django 서버 실행 프로세스
    pipe_list = {}

    api_server = Process(target=run_api_server, name="api_server")
    api_server.start()

    processes.append(api_server)
    print("run_api_server appended in processes(list)")

    # DB 연결 세팅
    MYSQL_SETTINGS = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '1234',
        'db': 'ras'
    }

    # 변화를 감지하는 BinLogStreamReader 객체 생성
    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=1,
        blocking=True,
        only_events=[WriteRowsEvent, DeleteRowsEvent],
        only_tables=['running_state'],
        resume_stream=True,
        skip_to_timestamp=0
    )

    while True:
        try:
            for event in stream:
                # RotateEvent는 binlog 파일이 변경될 때 발생하므로 무시합니다.
                if isinstance(event, RotateEvent):
                    continue
                elif isinstance(event, WriteRowsEvent):
                    # Write event: 사용자가 달리기를 시작한 경우 발생함
                    print("Write event detected")
                    for row in event.rows:
                        # values를 list로 변환하여 val[1]로 process_name을 생성
                        val = list(row['values'].values())
                        process_name = f"running_{val[1]}"
                        # process_name을 사용하여 pipe 생성, duplex=False로 하여 단방향 통신 설정
                        pipe_list[process_name] = tuple(Pipe(duplex=False))
                        # 프로세스 생성 및 시작
                        new_running_process = Process(target=running_process,
                                                      args=(process_name, pipe_list[process_name][0]),
                                                      name=process_name)
                        new_running_process.start()
                        # 프로세스 리스트에 추가
                        processes.append(new_running_process)
                elif isinstance(event, DeleteRowsEvent):
                    # Delete event: 사용자가 달리기를 종료한 경우 발생함
                    print("Delete event detected")
                    for row in event.rows:
                        # values를 list로 변환하여 val[1]로 process_name을 생성
                        val = list(row['values'].values())
                        process_name = f"running_{val[1]}"
                        for idx in range(len(processes)):
                            # process_name과 같은 name을 가진 process를 리스트에서 검색
                            if processes[idx].name == process_name:
                                # pipe 통신을 활용하여 kill 명령 전달
                                # .kill()을 사용해도 되지만, 안전한 종료 조건 및 코드를 작성하기 위함
                                pipe_list[process_name][1].send("kill")
                                processes = processes[:idx] + processes[idx + 1:]
                                break
        except KeyboardInterrupt as ke:
            break

    # while문이 끝난 후 모든 프로세스 종료
    for idx in range(len(processes)):
        print(f"terminate {processes[idx].name}")
        if processes[idx].name == "api_server":
            processes[idx].terminate()
        else:
            pipe_list[processes[idx].name][1].send("kill")
