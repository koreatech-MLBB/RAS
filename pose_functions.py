import numpy as np


def cosine_similarity(list1, list2):
    # 리스트를 넘파이 배열로 변환합니다.
    vec1 = np.array(list1)
    vec2 = np.array(list2)

    # 벡터의 내적을 계산합니다.
    dot_product = np.dot(vec1, vec2)

    # 각 벡터의 크기를 계산합니다.
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # 코사인 유사도를 계산합니다.
    if magnitude1 == 0 or magnitude2 == 0:
        # 만약 두 벡터 중 하나라도 크기가 0이라면, 코사인 유사도를 정의할 수 없습니다.
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def normalize_list(input_list, a=0, b=1):
    min_val = min(input_list)
    max_val = max(input_list)

    if min_val == max_val:
        # 모든 값이 동일한 경우, 모든 값을 0으로 설정
        return [0.0] * len(input_list)

    normalized_list = [(((x - min_val) / (max_val - min_val)) * (b - a)) + a for x in input_list]
    return normalized_list


def calculate_angle(a, b, c):
    # 벡터 BA와 BC를 numpy 배열로 계산합니다.
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    # 벡터 BA와 BC의 내적을 구합니다.
    dot_product = np.dot(ba, bc)

    # 벡터 BA와 BC의 크기를 구합니다.
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # 벡터 크기가 0인 경우를 처리합니다.
    if magnitude_ba == 0 or magnitude_bc == 0:
        return None  # 각도를 0도로 설정하거나 적절히 처리

    # 내적과 크기를 이용해 각도(라디안 단위)를 구합니다.
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 라디안 단위의 각도를 도 단위로 변환합니다.
    angle_degrees = np.degrees(angle_radians)

    # 더 작은 각도를 선택합니다.
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees

    return angle_degrees


# NOTE: elbow 점수 기준 변경 (+ 오른쪽 팔에 대한 대안...?)
# def elbow_angle_calc_scores(shoulder, elbow, wrist, length, tar=(1,)):
#     scores = []
#     angle_buf = None
#     angle_res = [[], []]
#
#     for t in tar:
#         for _i in range(length):
#             _angle = calculate_angle(shoulder[t][_i], elbow[t][_i], wrist[t][_i])
#             if _angle is None:
#                 _angle = angle_buf
#             else:
#                 angle_buf = _angle
#             angle_res[t].append(_angle)
#
#         if len(tar) == 2:
#             if (angle_res[0] > 110 or angle_res[1] > 110) or (angle_res[0] < 70 or angle_res[1] < 70):
#                 scores.append(0)
#             else:
#                 scores.append(100)
#         else:
#             if angle_res[0] > 110 or angle_res[0] < 70:
#                 scores.append(0)
#             else:
#                 scores.append(100)
#     return scores


def calc_inclination(dot_a, dot_b, reversed=False):
    if dot_a[0] > dot_b[0]:
        dot_a, dot_b = dot_b, dot_a

    dx = 0 if not reversed else 1
    dy = 1 if not reversed else 0

    if dot_b[dx] - dot_a[dx] == 0:
        return float('inf')

    return (dot_b[dy] - dot_a[dy]) / (dot_b[dx] - dot_a[dx])


def inclination_to_degree(inclination):
    return (np.arctan(inclination) * 180) / np.pi


# 논문 그래프에서 추출한 시간에 따른 무릎 각도 함수
def knee_function(_x):
    return (
            (1.3148 * (0.1 ** 8)) * (_x ** 5)
            - (3.3236 * (0.1 ** 6)) * (_x ** 4)
            + 0.0002891 * (_x ** 3)
            - 0.010055 * (_x ** 2)
            + 0.12522 * _x
    )


# 논문 그래프에서 추출한 시간에 따른 골반 각도 함수
def hip_function(_x):
    return (
            (-4.1847 * (0.1 ** 14)) * (_x ** 8)
            + (1.6414 * (0.1 ** 11)) * (_x ** 7)
            - (2.484 * (0.1 ** 9)) * (_x ** 6)
            + (1.8194 * (0.1 ** 7)) * (_x ** 5)
            - (6.8429 * (0.1 ** 6)) * (_x ** 4)
            + 0.0001503 * (_x ** 3)
            - 0.0024657 * (_x ** 2)
            + 0.0020494 * _x
            + 0.91
    )


# 논문 그래프에서 추출한 시간에 따른 발목 각도 함수
def ankle_function(_x):
    return (
            1.8553 * (0.1 ** 14) * (_x ** 8)
            - 1.4506 * (0.1 ** 11) * (_x ** 7)
            + 3.916 * (0.1 ** 9) * (_x ** 6)
            - 5.0626 * (0.1 ** 7) * (_x ** 5)
            + 3.4058 * (0.1 ** 5) * (_x ** 4)
            - 0.0011533 * (_x ** 3)
            + 0.016508 * (_x ** 2)
            - 0.059354 * _x
            + 0.68168
    )


def feedback(user_angles, target_angles, mid, body_point, direction, threshold=20):
    """
    피드백을 생성하는 함수

    :param user_angles:
    :param target_angles:
    :param mid: 입각기와 유각기를 나누는 인덱스
    :param body_point: (0~5) 0: 발목, 1: 무릎, 2: 골반, 3: 상체, 4: 시선, 5: 팔꿈치
    :param direction: 왼 or 오른
    :param threshold: 임계값
    :return: 피드백
    """

    # 입각기, 유각기에서 임계치를 넘어서 선넘게 틀린 값들의 개수
    # [0]: 사용자의 각도가 더 작거나 같을 때
    # [1]: 사용자의 각도가 더 클 때
    penalty_1, penalty_2 = [0, 0], [0, 0]
    # 입각기
    for x in range(0, mid):
        diff = abs(user_angles[x] - target_angles[x])
        if diff > threshold:
            penalty_1[0 if user_angles[x] <= target_angles[x] else 1] += 1
    # 유각기
    for x in range(mid, user_angles.size):
        diff = abs(user_angles[x] - target_angles[x])
        if diff > threshold:
            penalty_2[0 if user_angles[x] <= target_angles[x] else 1] += 1

    feedbacks = []

    # 입각기에서 사용자의 관절 각도가 더 작은 경우
    if penalty_1[0] > mid / 2:
        if body_point == 0:
            # 발이 땅을 디디고 있을 때 발목을 조금 더 굽혀주세요.
            feedbacks.append(1 if direction == "왼" else 2)
        elif body_point == 1:
            # 발이 땅을 디디고 있을 대 무릎을 조금 더 굽혀주세요.
            feedbacks.append(9 if direction == "왼" else 10)
        elif body_point == 2:
            feedbacks.append(17 if direction == "왼" else 18)

    # 유각기에서 사용자의 관절 각도가 더 작은 경우
    if penalty_2[0] > (user_angles.size - mid) / 2:
        if body_point == 0:
            feedbacks.append(3 if direction == "왼" else 4)
        elif body_point == 1:
            feedbacks.append(11 if direction == "왼" else 12)
        elif body_point == 2:
            feedbacks.append(19 if direction == "왼" else 20)

    # 입각기에서 사용자의 관절 각도가 더 큰 경우
    if penalty_1[1] > mid / 2:
        if body_point == 0:
            feedbacks.append(5 if direction == "왼" else 6)
        elif body_point == 1:
            feedbacks.append(13 if direction == "왼" else 14)
        elif body_point == 2:
            feedbacks.append(21 if direction == "왼" else 22)

    # 유각기에서 사용자의 관절 각도가 더 큰 경우
    if penalty_2[1] > (user_angles.size - mid) / 2:
        if body_point == 0:
            feedbacks.append(7 if direction == "왼" else 8)
        elif body_point == 1:
            feedbacks.append(15 if direction == "왼" else 16)
        elif body_point == 2:
            feedbacks.append(23 if direction == "왼" else 24)

    return feedbacks
