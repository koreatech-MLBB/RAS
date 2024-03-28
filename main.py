from db_connection import *
from multiprocessing import Process
import os


def run_db_server():
    os.system("python ./ras/manage.py runserver 0.0.0.0:8000")


def run_test():
    ras_db = DBConnection("root", "1234", "RAS")
    print(ras_db.get_tables())


if __name__ == "__main__":
    p1 = Process(target=run_db_server)
    p2 = Process(target=run_test)

    processes = [p1, p2]

    for p in processes:
        p.start()

    for p in processes:
        p.join()