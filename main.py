import os
import time

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANFORMS"] = "0"

# openpose와 models 경로 추가하기
import sys
openpose_dir = "D:\\RAS\\openpose\\Release"
model_dir = "D:\\RAS\\models"
sys.path.append(openpose_dir)

# openpose: 제대로 빌드되지 않았을 경우 대비하여 try-exception으로 감싸기
try:
    import pyopenpose as op
except ImportError as e:
    raise e

# 필요한 모듈들 import하기
from multiprocessing import Process, Pipe, shared_memory
import cv2

from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.event import RotateEvent
from pymysqlreplication.row_event import WriteRowsEvent, DeleteRowsEvent

from scipy.signal import find_peaks

import pathlib
import datetime

import imageio

# pose 점수 알고리즘에 필요한 함수들 import하기
from pose_functions import *


# Django server 실행하는 함수 작성
def run_api_server():
    try:
        os.system("python ./ras/manage.py runserver 0.0.0.0:8080")
    except BaseException as run_api_server_exception:
        print(f"run_api_server.except: {run_api_server_exception.__str__()}")
        print("run_api_server terminated")


# 달리기 알고리즘 실행하는 함수 작성
def running_process(my_name, running_id, read_handle):
    # 웹캠 설정
    # 해상도: 720 * 1280
    # cap = cv2.VideoCapture("./examples/run_example_4.mp4")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 첫 테스트 프레임 읽어오기
    # while True:
    #     ret, test_frame = cap.read()
    #     if ret:
    #         break

    # # print(test_frame.shape)
    # os.system("pause")
    # 테스트 프레임 크기로 공유 메모리 생성
    # buf1 = shared_memory.SharedMemory(name=f"{my_name}_frame",
    #                                   create=True,
    #                                   size=480 * 640 * 3 * 8)
    #
    # # 피드백 문자열 전달할 공유 메모리 생성
    # b = np.ndarray(shape=(1, ), dtype=np.int32)
    # buf2 = shared_memory.SharedMemory(name=f"{my_name}_feedback",
    #                                   create=True,
    #                                   size=b.nbytes)
    #
    # # 점수 전달할 공유 메모리 생성
    # info = np.ndarray(shape=(11, ), dtype=np.float64)
    # buf3 = shared_memory.SharedMemory(name=f"{my_name}_info",
    #                                   create=True,
    #                                   size=info.nbytes)
    #
    # # 피드백 등장 횟수 전달할 공유 메모리 생성
    # feed_cnt = np.ndarray(shape=(3, ), dtype=np.int32)
    # buf4 = shared_memory.SharedMemory(name=f"{my_name}_feed_cnt",
    #                                   create=True,
    #                                   size=feed_cnt.nbytes)

    # openpose 모델에 필요한 객체 생성
    op_wrapper = op.WrapperPython()
    op_wrapper.configure({"model_folder": model_dir})
    op_wrapper.start()

    # 필요한 관절 좌표 인덱스 생성
    body_coordinates = dict()
    body = {"Nose": (0,), "Neck": (1,), "Mid-hip": (8,),
            "Eye": (15, 16), "Ear": (17, 18), "Shoulder": (2, 5), "Elbow": (3, 6), "Wrist": (4, 7),
            "Hip": (9, 12), "Knee": (10, 13), "Ankle": (11, 14), "Big-toe": (22, 19)}

    # 관절 좌표 초기화
    for key, values in body.items():
        if len(values) == 1:
            body_coordinates[key] = [[]]
        else:
            body_coordinates[key] = [[], []]

    # best pose를 저장할 변수 선언
    best_frames = (None, 0)

    # worst pose를 저장할 변수 선언
    worst_frames = (None, 100)

    # frame을 저장할 리스트 생성 및 메모리 초기화
    frames = [0 for _ in range(100)]
    frames.clear()

    total_score = {
        'knee_score': [[], []],
        'hip_score': [[], []],
        'ankle_score': [[], []],
        'gaze_score': [],
        'elbow_score': [],
        'upper_body_score': []
    }

    # 피드백 등장 횟수 카운트
    cnt_feedback = {}

    steps = 0

    # 시작!
    while True:
        # pipe: 부모 프로세스로부터 메시지 수신
        if read_handle.poll(timeout=0.01):
            data = read_handle.recv()

            if data == "kill":
                # print("kill??")
                w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적
                cap.release()
                # print(f"kill this({my_name}) process")
                if best_frames[0] is not None:
                    # # print(f"best_frames is not None : {best_frames[0]}")

                    # best_frames를 mp4형식으로 저장
                    _my_name = my_name.split('_')
                    save_path = f"ras/ras_db/static/media/best_pose/{_my_name[1]}/{running_id}"
                    _directory = pathlib.Path(save_path)
                    if not _directory.exists():
                        os.makedirs(save_path)

                    # 비디오 파일 이름 및 코덱 설정
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 코덱 설정
                    # fps = 10  # 초당 프레임 수 (FPS)
                    # frame_size = (1280, 720)  # 프레임 크기 (width, height)

                    # 비디오 라이터 객체 생성
                    # out = cv2.VideoWriter(save_path + '/best_pose.avi', fourcc, fps, frame_size)
                    out = cv2.VideoWriter(save_path + '/best_pose.mp4', fourcc, 10.0, (640, 480))

                    # 프레임 리스트를 순회하며 비디오 파일에 프레임 추가
                    best_f = []
                    for frame in best_frames[0]:
                        frame = np.reshape(frame, (480, 640, 3))
                        # best_f.append(frame)
                        out.write(frame)
                    # imageio.mimsave(save_path + '/best_pose.gif', best_f, fps=10)
                    # print("this is the end of video save...")
                    # 비디오 라이터 객체 해제
                    out.release()

                    score = {}
                    # 총점 계산
                    for key in ['ankle_score', 'knee_score', 'hip_score']:
                        for index, direct in [(0, 'right'), (1, 'left')]:
                            score[f"{direct}_{key}"] = round(sum(total_score[key][index]) / len(total_score[key][index]), 2) if len(total_score[key][index]) else 0

                    for key in ['gaze_score', 'elbow_score', 'upper_body_score']:
                        score[key] = round(sum(total_score[key]) / len(total_score[key]), 2) if len(total_score[key]) else 0

                    total_score_buf = 0
                    # print("this is the total score calcul")

                    # 가중치를 곱하여 계산
                    for key, _val in score.items():
                        if key not in ['gaze_score', 'elbow_score', 'upper_body_score']:
                            total_score_buf += _val * 6
                        else:
                            total_score_buf += _val

                    score['total_score'] = round(total_score_buf / 39, 2)

                    # print(f"score : {score}")

                    # 점수 및 걸음 수 저장
                    # 순서
                    # [0] = right_ankle_score
                    # [1] = left_ankle_score
                    # [2] = right_knee_score
                    # [3] = left_knee_score
                    # [4] = right_hip_score
                    # [5] = left_hip_score
                    # [6] = gaze_score
                    # [7] = elbow_score
                    # [8] = upper_body_score
                    # [9] = total_score
                    # [10] = steps

                    # print("let's start the score!~~~!")
                    my_score = shared_memory.SharedMemory(name=f"{my_name}_info")
                    my_scores = np.array([x for x in score.values()] + [steps])
                    print(f"my_scores: {my_scores}")
                    my_scores_buf = np.ndarray(shape=(11, ), dtype=np.float64, buffer=my_score.buf)
                    np.copyto(my_scores_buf, my_scores)
                    my_score.close()
                    # print("the end of score~~~!!")

                    # 피드백 등장 횟수 저장
                    my_feed_cnt = shared_memory.SharedMemory(name=f"{my_name}_feed_cnt")
                    f = list(cnt_feedback.items()) + [(-1, 0), (-1, 0), (-1, 0)]
                    f = list(sorted(f, key=lambda x: x[1], reverse=True))
                    feed_cnts = np.array([f[i][0] for i in range(3)])
                    my_feed_cnt_buf = np.ndarray(shape=(3, ), dtype=np.int32, buffer=my_feed_cnt.buf)
                    np.copyto(my_feed_cnt_buf, feed_cnts)
                    my_feed_cnt.close()
                    # print("feed back is end~~~!!!")
                    time.sleep(5)

                # print("i am break")

                break

        # 프레임 읽어오기
        ret, frame = cap.read()
        # # print("after break")
        if not ret:
            # print('no target frame')
            continue
        # try:
        time1 = time.time()
        # openpose 이미지 처리
        datum = op.Datum()
        image_to_process = frame
        datum.cvInputData = image_to_process
        op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        time2 = time.time()

        # 필요한 관절 좌표 저장
        if datum.poseKeypoints is not None:
            try:
                for k, _idx in body.items():
                    for i in range(len(_idx)):
                        body_coordinates[k][i].append(tuple(datum.poseKeypoints[0][_idx[i]][:2]))
            except BaseException as e:
                # print(f"{my_name} there is no poseKeypoints")
                raise e
        else:
            continue
            # # print(datum.poseKeypoints)
            # # print(f"{e} : 관절 정보가 없습니다.")

        try:
            # 공유메모리에 openpose 이미지 처리 결과 frame 저장
            my_frame = shared_memory.SharedMemory(name=f"{my_name}_frame")
            frame = np.array(datum.cvOutputData)
            my_frame_buf = np.ndarray(frame.shape, dtype=np.uint8, buffer=my_frame.buf)
            np.copyto(my_frame_buf, frame)
            my_frame.close()
        except FileNotFoundError as e:
            # print(f"{my_name}_frame does not found in running process (line 242)")
            raise e
        except BaseException as e:
            # print(f"{my_name}: unexpected error (line 245)")
            raise e

        # frames에 openpose 이미지 처리 결과 frame 저장
        # # print("frame")
        frames.append(datum.cvOutputData)
        # # print("frame save")

        # Neck좌표에 -1을 곱하여 np.array로 변환
        neck_position = -1 * np.array([round(y, -1) for _, y in body_coordinates['Neck'][0]])

        # Neck좌표를 find_peaks로 전달 -> peak값 뽑기
        peaks, _ = find_peaks(neck_position)

        # peaks의 길이가 2라면: 한 달리기 주기이므로 점수 계산
        if len(peaks) == 2:
            # # print("peak?")
            # 걸음수 +1
            steps += 1

            # target: 앞으로 가고 있는 발 (이번 주기의 점수 측정 대상이 되는 발)
            target = 1 if body_coordinates['Ankle'][0][0] > body_coordinates['Ankle'][1][0] else 0

            # 각 관절의 각도를 저장할 리스트 생성
            knee_angles = []
            hip_angles = []
            ankle_angles = []

            # 시선, 상체의 점수를 저장할 리스트 생성
            gaze_scores = []
            upper_body_scores = []
            elbow_scores = []

            # 팔꿈치 각도 점수 계산
            # elbow_scores = elbow_angle_calc_scores(shoulder=body_coordinates['Shoulder'],
            #                                        elbow=body_coordinates['Elbow'],
            #                                        wrist=body_coordinates['Wrist'],
            #                                        length=peaks[-1] + 1)

            # 한 달리기 주기의 각 관절 각도 계산

            for i in range(peaks[-1] + 1):
                # 무릎 각도
                knee_angle = calculate_angle(a=body_coordinates['Hip'][target][i],
                                             b=body_coordinates['Knee'][target][i],
                                             c=body_coordinates['Ankle'][target][i])
                if knee_angle is None:
                    if knee_angles:
                        knee_angles.append(knee_angles[-1])
                    else:
                        knee_angles.append(0)
                else:
                    knee_angles.append(180 - knee_angle)
                # knee_angles.append(180 - knee_angle if knee_angle is not None else knee_angles[-1])

                # 골반 각도
                hip_angle = calculate_angle(a=body_coordinates['Neck'][0][i],
                                            b=body_coordinates['Hip'][target][i],
                                            c=body_coordinates['Knee'][target][i])
                if hip_angle is None:
                    if hip_angles:
                        hip_angles.append(hip_angles[-1])
                    else:
                        hip_angles.append(0)
                else:
                    hip_angles.append(180 - hip_angle)
                # hip_angles.append(180 - hip_angle if hip_angle is not None else hip_angles[-1])

                # 발목 각도
                ankle_angle = calculate_angle(a=body_coordinates['Big-toe'][target][i],
                                              b=body_coordinates['Ankle'][target][i],
                                              c=body_coordinates['Knee'][target][i])
                if ankle_angle is None:
                    if ankle_angles:
                        ankle_angles.append(ankle_angles[-1])
                    else:
                        ankle_angles.append(0)
                else:
                    ankle_angles.append(180 - ankle_angle)
                # ankle_angles.append(90 - ankle_angle if ankle_angle is not None else ankle_angles[-1])

                # 팔꿈치
                elbow_angle = calculate_angle(a=body_coordinates['Shoulder'][1][i],
                                              b=body_coordinates['Elbow'][1][i],
                                              c=body_coordinates['Wrist'][1][i])
                if elbow_angle is None:
                    if elbow_scores:
                        elbow_scores.append(elbow_scores[-1])
                    else:
                        elbow_scores.append(0)
                else:
                    elbow_scores.append(100 if 70 <= elbow_angle <= 100 else 0)

                # 시선
                gaze_score = inclination_to_degree(calc_inclination(dot_a=body_coordinates['Eye'][1][i],
                                                                    dot_b=body_coordinates['Ear'][1][i]))
                gaze_scores.append(100 if -15 <= gaze_score <= 15 else 0)

                # 상체
                upper_body_score = inclination_to_degree(calc_inclination(dot_a=body_coordinates['Neck'][0][i],
                                                                          dot_b=body_coordinates['Mid-hip'][0][i],
                                                                          reversed=True))
                upper_body_scores.append(100 if -15 <= upper_body_score <= 15 else 0)

            # 각 관절 각도를 0~1로 정규화
            knee_angles = normalize_list(knee_angles)
            hip_angles = normalize_list(hip_angles)
            ankle_angles = normalize_list(ankle_angles)

            # 0~100사이의 peaks[-1] + 1개의 포인트 생성
            x = np.linspace(0, 100, peaks[-1] + 1)

            # x를 논문에서 추출한 함수에 대입하여 정답 리스트 작성
            norm_target_knee_angles = [knee_function(_val) for _val in x]
            norm_target_hip_angles = [hip_function(_val) for _val in x]
            norm_target_ankle_angles = [ankle_function(_val) for _val in x]

            # 사용자의 관절 각도와 논문에서 추출한 정답 리스트 사이의 cosine_similarity 계산
            knee_score = round(cosine_similarity(norm_target_knee_angles, knee_angles) * 100, 1)
            hip_score = round(cosine_similarity(norm_target_hip_angles, hip_angles) * 100, 1)
            ankle_score = round(cosine_similarity(norm_target_ankle_angles, ankle_angles) * 100, 1)

            # 시선, 팔꿈치, 상체의 경우 평균값으로 계산
            gaze_score = round(sum(gaze_scores) / (peaks[-1] + 1), 1)
            elbow_score = round(sum(elbow_scores) / (peaks[-1] + 1), 1)
            upper_body_score = round(sum(upper_body_scores) / (peaks[-1] + 1), 1)

            best_score = sum([x * 3 for x in [knee_score, hip_score, ankle_score]] + [gaze_score, elbow_score,
                                                                                      upper_body_score]) / 12
            if best_frames[1] < best_score:
                best_frames = (frames[:peaks[-1] + 1], best_score)

            # 나중에 총점을 도출할 때 사용
            total_score['knee_score'][target].append(knee_score)
            total_score['hip_score'][target].append(hip_score)
            total_score['ankle_score'][target].append(ankle_score)
            total_score['gaze_score'].append(gaze_score)
            total_score['elbow_score'].append(elbow_score)
            total_score['upper_body_score'].append(upper_body_score)

            # 피드백 생성을 위한 target 재정규화
            target_ankle_angles = np.array(normalize_list(norm_target_ankle_angles, a=-20, b=30))
            target_knee_angles = np.array(normalize_list(norm_target_knee_angles, a=20, b=90))
            target_hip_angles = np.array(normalize_list(norm_target_hip_angles, a=-5, b=56))

            # np.array로 변환
            ankle_angles = np.array(ankle_angles)
            knee_angles = np.array(knee_angles)
            hip_angles = np.array(hip_angles)

            # 여기서 피드백 발생
            feedbacks = []
            for target_angles, user_angles, _score, _idx in [(target_ankle_angles, ankle_angles, ankle_score, 0),
                                                             (target_knee_angles, knee_angles, knee_score, 1),
                                                             (target_hip_angles, hip_angles, hip_score, 2)]:
                v = 80 if target else 70

                if _score < v:
                    feedback_list = feedback(user_angles=user_angles,
                                             target_angles=target_angles,
                                             mid=int((peaks[-1] + 1) * 0.3),
                                             body_point=_idx,
                                             direction="오른" if target else "왼",
                                             threshold=20)

                    for _val in feedback_list:
                        feedbacks.append(_val)

            if upper_body_score <= 20:
                feedbacks.append(25)
            if gaze_score <= 20:
                feedbacks.append(26)
            if elbow_score <= 20:
                feedbacks.append(27)

            # 피드백 등장 횟수 카운트
            for f in feedbacks:
                try:
                    cnt_feedback[f] += 1
                except KeyError:
                    # print(f"warning: there is no key({f}) in cnt_feedback. Auto add.")
                    cnt_feedback[f] = 1

            # 앞서 계산한 한 주기의 절반만큼의 정보를 삭제
            for k, __idx in body.items():
                for i in range(len(__idx)):
                    body_coordinates[k][i] = body_coordinates[k][i][peaks[0] + 1:]
            frames = frames[peaks[0] + 1:]

            # 생성된 피드백을 공유메모리에 id로 저장
            try:
                my_audio = shared_memory.SharedMemory(name=f"{my_name}_feedback")
                if feedbacks:
                    feedback_str = np.array([feedbacks[0]])
                else:
                    feedback_str = np.array([-1])
                my_audio_buf = np.ndarray(shape=(1, ), dtype=np.int32, buffer=my_audio.buf)
                np.copyto(my_audio_buf, feedback_str)
                my_audio.close()
            except FileNotFoundError as e:
                # print(f"{my_name}_feedback does not found in running process (line 430)")
                raise e
            except BaseException as e:
                # print(f"{my_name}: unexpected error (line 433)")
                raise e

        time3 = time.time()

        # print(f"model running time: {time2 - time1}")
        # print(f"pose algorithm running time: {time3 - time2}")
        # except BaseException as running_process_exception:
        #     # print("in main running process")
        #     # print(running_process_exception)
        #     continue
    return


if __name__ == "__main__":

    # 모든 프로세스가 담길 리스트
    processes = []

    # Django 서버 실행 프로세스
    pipe_list = {}


    api_server = Process(target=run_api_server, name="api_server")
    api_server.start()

    processes.append(api_server)
    # print("run_api_server appended in processes(list)")

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
                    # print("Write event detected")
                    for row in event.rows:
                        # values를 list로 변환하여 val[1]로 process_name을 생성
                        val = list(row['values'].values())
                        # print("value :", val)
                        # print("val[1]: ", val[1], type(val[1]))
                        process_name = f"running_{val[1]}"
                        # process_name을 사용하여 pipe 생성, duplex=False로 하여 단방향 통신 설정
                        pipe_list[process_name] = tuple(Pipe(duplex=False))
                        # print("pipe list")
                        # 프로세스 생성 및 시작
                        print(process_name)
                        try:
                            buf1 = shared_memory.SharedMemory(name=f"{process_name}_frame",
                                                              create=True,
                                                              size=480 * 640 * 3 * 8)
                        except FileExistsError as e:
                            # print(f"{process_name}_frame is already exists")
                            raise e
                        # buf1.close()

                        # 피드백 문자열 전달할 공유 메모리 생성
                        b = np.ndarray(shape=(1,), dtype=np.int32)
                        try:
                            buf2 = shared_memory.SharedMemory(name=f"{process_name}_feedback",
                                                              create=True,
                                                              size=b.nbytes)
                        except FileExistsError as e:
                            # print(f"{process_name}_feedback is already exists")
                            raise e
                        # buf2.close()

                        # 점수 전달할 공유 메모리 생성
                        info = np.ndarray(shape=(11,), dtype=np.float64)
                        try:
                            buf3 = shared_memory.SharedMemory(name=f"{process_name}_info",
                                                              create=True,
                                                              size=info.nbytes)
                        except FileExistsError as e:
                            # print(f"{process_name}_info is already exists")
                            raise e
                        # buf3.close()

                        # 피드백 등장 횟수 전달할 공유 메모리 생성
                        feed_cnt = np.ndarray(shape=(3,), dtype=np.int32)
                        try:
                            buf4 = shared_memory.SharedMemory(name=f"{process_name}_feed_cnt",
                                                              create=True,
                                                              size=feed_cnt.nbytes)
                        except FileExistsError as e:
                            # print(f"{process_name}_feed_cnt is already exists")
                            raise e
                        # buf4.close()

                        new_running_process = Process(target=running_process,
                                                      args=(process_name, datetime.datetime.now().date(), pipe_list[process_name][0]),
                                                      name=process_name)
                        new_running_process.start()
                        # 프로세스 리스트에 추가
                        processes.append(new_running_process)

                        # print("process start")
                elif isinstance(event, DeleteRowsEvent):
                    # Delete event: 사용자가 달리기를 종료한 경우 발생함
                    # print("Delete event detected")
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
                                time.sleep(5)
                                try:
                                    buf1 = shared_memory.SharedMemory(name=f"{process_name}_frame")
                                    buf1.close()
                                    buf1.unlink()
                                except FileNotFoundError as e:
                                    # print(f"{process_name}_frame is not found")
                                    raise e

                                try:
                                    buf2 = shared_memory.SharedMemory(name=f"{process_name}_feedback")
                                    buf2.close()
                                    buf2.unlink()
                                except FileNotFoundError as e:
                                    # print(f"{process_name}_feedback is not found")
                                    raise e

                                try:
                                    buf3 = shared_memory.SharedMemory(name=f"{process_name}_info")
                                    buf3.close()
                                    buf3.unlink()
                                except FileNotFoundError as e:
                                    # print(f"{process_name}_info is not found")
                                    raise e

                                try:
                                    buf4 = shared_memory.SharedMemory(name=f"{process_name}_feed_cnt")
                                    buf4.close()
                                    buf4.unlink()
                                except FileNotFoundError as e:
                                    # print(f"{process_name}_feed_cnt is not found")
                                    raise e
                                processes = processes[:idx] + processes[idx + 1:]
                                break
        except KeyboardInterrupt as ke:
            # print("main process is terminated")
            for idx in range(len(processes)):
                # print(f"terminate {processes[idx].name}")
                if processes[idx].name == "api_server":
                    processes[idx].terminate()
                else:
                    pipe_list[processes[idx].name][1].send("kill")

            raise ke

            # break

    # while문이 끝난 후 모든 프로세스 종료
    # for idx in range(len(processes)):
    #     # print(f"terminate {processes[idx].name}")
    #     if processes[idx].name == "api_server":
    #         processes[idx].terminate()
    #     else:
    #         pipe_list[processes[idx].name][1].send("kill")
