import os
import time
import math
from multiprocessing import Process
from collections import deque

import numpy as np
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.event import RotateEvent
from pymysqlreplication.row_event import WriteRowsEvent, DeleteRowsEvent

from db_connection import *


def run_db_server():
    os.system("python ./ras/manage.py runserver 0.0.0.0:8080")


def calc_cycle_sim(left_foot_position=None, right_foot_position=None):
    """
    :parameter: left_foot_position, right_foot_position
    :return: left_sim, right_sim
    """
    assert len(left_foot_position) == len(right_foot_position), "length must be same"
    assert len(left_foot_position) > 1, "left_foot_position length must larger than 1"
    assert len(right_foot_position) > 1, "right_foot_position length must larger than 1"

    length = len(left_foot_position)
    left_min = min(left_foot_position.values(), key=lambda x: x >= 0)
    right_min = min(right_foot_position.values(), key=lambda x: x >= 0)
    left_foot_positions = np.array([val if val >= 0 else left_min for val in left_foot_position.values()])
    right_foot_positions = np.array([val if val >= 0 else right_min for val in right_foot_position.values()])

    target = np.array(np.sin([i for i in range(length)], len(left_foot_position)))

    left_sim = (length - sum([abs(x) for x in target - left_foot_positions])) / length
    right_sim = (length - sum([abs(x) for x in target - right_foot_positions])) / length

    return left_sim, right_sim


def pose_scoring(pose: dict,  cycle_info: list):
    pass

def running_process():

    ras_db = DBConnection("root", "1234", "RAS")
    print(ras_db.get_tables())


def db_replication():

    # MySQL Replication 연결 설정
    MYSQL_SETTINGS = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '1234',
        'db': 'ras'
    }

    def process_binlog_event(event):
        if isinstance(event, WriteRowsEvent):
            print("Write event detected")
            for row in event.rows:
                print("Inserted row:", row)
                # 여기에서 삽입된 데이터 처리
        elif isinstance(event, DeleteRowsEvent):
            print("Delete event detected")
            for row in event.rows:
                print("Deleted row:", row)
                # 여기에서 삭제된 데이터 처리

    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=1,
        blocking=True,
        only_events=[WriteRowsEvent, DeleteRowsEvent],
        only_tables=['running_state'],
        resume_stream=True,
        skip_to_timestamp=0
    )

    for binlogevent in stream:
        if isinstance(binlogevent, RotateEvent):
            continue  # RotateEvent는 binlog 파일이 변경될 때 발생하므로 무시합니다.
        process_binlog_event(binlogevent)

    # 공유메모리에 user_id저장.

    # stream.close()


if __name__ == "__main__":
    # Django 서버 실행 프로세스
    p1 = Process(target=run_db_server)

    p2 = Process(target=running_process, name=f"{time.time()}")
    print(p2.name)
    p3 = Process(target=db_replication)

    # 공유메모리에 user_id 가져와.
    # Process(target=run_test, name=user_id)
    # run_test_arr.append(p)
    # run_test_arr.find(p.name=user_id).kill()

    processes = [p1, p2, p3]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
