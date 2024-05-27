import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANFORMS"] = "0"

from multiprocessing import Process, Pipe, shared_memory
import cv2

import numpy as np
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.event import RotateEvent
from pymysqlreplication.row_event import WriteRowsEvent, DeleteRowsEvent

from scipy.signal import find_peaks


def run_api_server():
    try:
        os.system("python ./ras/manage.py runserver 0.0.0.0:8080")
    except KeyboardInterrupt as e:
        print(f"run_api_server.except: {e.__str__()}")
        print("run_api_server terminated")


def pose_scoring(pose: dict):
    # find cycle
    hip_position = -1 * np.array([pose['LEFT_HIP']])
    peak, _ = find_peaks(hip_position)

    # peak값이 두개가 아니다 -> 아직 한 주기가 나타나지 않았다라고 판단
    # None을 return하여 주기가 없음을 전달
    if len(peak) != 2:
        return False, (None, None)

    cycle_score = 0
    cycle_idx = {'start': 0, 'end': peak[-1]}
    print(hip_position[cycle_idx[0]:cycle_idx[1] + 1])
    hip_position = np.array(hip_position[cycle_idx[1] + 1:])
    print(len(hip_position))

    # cycle_idx에 해당하는 점수 계산

    return True, cycle_idx, cycle_score


def best_pose(start_idx, end_idx):
    return start_idx, end_idx


def running_process(my_name, read_handle):
    print(f"{my_name}: streaming")
    new_shm = shared_memory.SharedMemory(name=my_name, create=True, size=480 * 680 * 3)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture("D:\\WorkSpace\\Graduation_Project\\openpose_custom\\example\\run_example_4.mp4")
    data = b""

    while True:
        if read_handle.poll(timeout=0.01):
            data = read_handle.recv()
            if data == "kill":
                cap.release()
                print(f"kill this({my_name}) process")
                break
        ret, frame = cap.read()
        my_shm = shared_memory.SharedMemory(name=my_name)
        frame = np.array(frame)
        my_shm_buf = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=my_shm.buf)
        np.copyto(my_shm_buf, frame)
        my_shm.close()


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
        'passwd': '1212',
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
