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
        return 0.0  # 각도를 0도로 설정하거나 적절히 처리

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
def elbow_angle_calc_scores(shoulder, elbow, wrist, length, tar=(1, )):
    scores = []
    for i in range(length):
        angle = [calculate_angle(shoulder[t][i], elbow[t][i], wrist[t][i]) for t in tar]

        if len(tar) == 2:
            if (angle[0] > 120 or angle[1] > 120) or (angle[0] < 60 or angle[1] < 60):
                scores.append(0)
            else:
                scores.append(100)
        else:
            if angle[0] > 120 or angle[0] < 60:
                scores.append(0)
            else:
                scores.append(100)
    return scores


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


def feedback(user_angles, target_angles, mid, body_point, direction, threshold=20):
    '''
    피드백을 생성하는 함수

    :param user_angles:
    :param target_angles:
    :param mid: 입각기와 유각기를 나누는 인덱스
    :param body_point: (0~5) 0: 발목, 1: 무릎, 2: 골반, 3: 상체, 4: 시선, 5: 팔꿈치
    :param threshold: 임계값
    :return: 피드백
    '''

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
            feedbacks.append(f"{direction} 발이 지면에서 떨어질 때 발목을 너무 펴지 마세요.")
        elif body_point == 1:
            feedbacks.append(f"{direction} 발이 땅을 디디고 있을 때 무릎을 너무 펴고 달리지 마세요.")
        elif body_point == 2:
            feedbacks.append(f"{direction} 뒤쪽 발을 조금 더 느긋하게 떼 주세요.")

    # 유각기에서 사용자의 관절 각도가 더 작은 경우
    if penalty_2[0] > (user_angles.size - mid) / 2:
        if body_point == 0:
            feedbacks.append(f"{direction} 발이 앞으로 나아갈 때 발목을 너무 펴지 마세요.")
        elif body_point == 1:
            feedbacks.append(f"{direction} 발이 앞으로 나아갈 때 무릎을 너무 펴고 달리지 마세요.")
        elif body_point == 2:
            feedbacks.append(f"{direction} 무릎을 조금 더 낮게 들어주세요.")

    # 입각기에서 사용자의 관절 각도가 더 큰 경우
    if penalty_1[1] > mid / 2:
        if body_point == 0:
            feedbacks.append(f"{direction} 발이 땅을 디디고 있을 때 발등이 너무 접히지 않도록 주의해 주세요.")
        elif body_point == 1:
            feedbacks.append(f"{direction} 발이 땅을 디디고 있을 때 무릎을 너무 접고 달리지 마세요.")
        elif body_point == 2:
            feedbacks.append(f"{direction} 뒤쪽 발을 조금 더 빠르게 떼 주세요.")

    # 유각기에서 사용자의 관절 각도가 더 큰 경우
    if penalty_2[1] > (user_angles.size - mid) / 2:
        if body_point == 0:
            feedbacks.append(f"{direction} 발이 앞으로 나아갈 때 발목을 너무 들지 마세요.")
        elif body_point == 1:
            feedbacks.append(f"{direction} 발이 앞으로 나아갈 때 무릎을 너무 접고 달리지 마세요.")
        elif body_point == 2:
            feedbacks.append(f"{direction} 무릎을 조금 더 높게 들어주세요.")

    return feedbacks
