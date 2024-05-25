import json
import re
from datetime import datetime, timedelta
import calendar

from multiprocessing import shared_memory

import random
from datetime import datetime

import cv2
import numpy as np
from django.contrib.auth import authenticate
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import *


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password']


class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = '__all__'


class RunningSerializer(serializers.ModelSerializer):
    class Meta:
        model = Running
        fields = ['running_id', 'running_date']


class RunningInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Running
        fields = '__all__'


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'


class MainView(APIView):
    def get(self, request):
        return render(request, 'main.html')


class SignUpView(APIView):
    def get(self, request):
        return render(request, 'signup.html')

    def post(self, request):
        data = json.loads(request.body.decode('utf-8'))
        try:
            id_ = User.objects.get(username=data['user_id'])
        except:
            id_ = None

        if id_ is not None:
            return Response({"result": False, "Error": "이미 존재 하는 아이디 입니다."})

        new_user = User.objects.create_user(
            username=data['user_id'],
            password=data['password'],
        )

        new_profile = Profile(
            id=new_user,
            name=data['name'],
            age=data['age'],
            weight=data['weight'],
            height=data['height'],
            gender=data['gender']
        )
        new_user.save()
        new_profile.save()

        Token.objects.create(user=new_user)
        return Response({"result": True})


class SignInView(APIView):
    def get(self, request):
        return render(request, 'login.html')

    def post(self, request):
        print(request)
        data = json.loads(request.body.decode('utf-8'))
        user = authenticate(username=data['user_id'], password=data['password'])
        if user is not None:
            token = Token.objects.get(user=user)
            response = render(request, 'ready_to_run.html')
            response.set_cookie(key='authToken', value=token.key)
            return response
            # return Response({"result": True, "Token": token.key})
        else:
            return Response({"result": False, "Error": "가입되지 않은 사용자입니다."})


class ProfileView(APIView):
    def get(self, request, user_id):
        user_ = User.objects.get(username=user_id)
        myprofile = Profile.objects.get(id=user_)

        context = {}
        if myprofile:
            context = {
                "name": myprofile.name,
                "user_id": user_id,
                # TODO: "password":
                "age": myprofile.age,
                "weight": myprofile.weight,
                "height": myprofile.height,
                "gender": 'Female' if myprofile.gender else 'Male'
            }
        print("hi this is profile")
        return render(request, 'profile.html', context)


# 달리기 메인 화면
class RunningMainView(APIView):
    def get(self, request, user_id):
        print("get running main view ++++++++++++++++")
        print(request.META)
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if 'HTTP_AUTHORIZATION' in request.META:
                if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                    print(user_)
                    return render(request, 'ready_to_run.html', {'user_id': user_id})
                    # return Response({"result": True, "runnings": runnings})
                else:
                    return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
            else:
                return Response({"result": False, "error": "토큰이 없습니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


def classify_time(total_time):
    if total_time >= timedelta(hours=1):
        return 5
    elif total_time >= timedelta(minutes=30):
        return 4
    elif total_time >= timedelta(minutes=5):
        return 3
    elif total_time > timedelta(seconds=0):
        return 2
    else:
        return 1


def create_calendar(runnings):
    log_times = []
    total_result = []
    for running in runnings:
        year = running['running_date'].year
        month = running['running_date'].month
        if (year, month) not in log_times:
            log_times.append((year, month))

    for (year, month) in log_times:
        cal = calendar.monthcalendar(year, month)
        cal_data = []

        for week in cal:
            week_data = []
            for day in week:
                if day == 0:
                    week_data.append((None, None, None))
                else:
                    time = timedelta()
                    for running in runnings:
                        if running['running_date'].day == day:
                            start_time = running['start_time']
                            end_time = running['end_time']
                            elapsed_time = end_time - start_time
                            print(elapsed_time)
                            time += elapsed_time
                    level = classify_time(time)
                    week_data.append((day, str(time), level))
            cal_data.append(week_data)
        total_result.append((month, cal_data))
    return total_result


# 캘린더 러닝 데이터
class RunningView(APIView):
    def get(self, request, user_id):
        print(request.META)
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if 'HTTP_AUTHORIZATION' in request.META:
                if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                    runnings = (Running.objects.filter(user=user_)
                                .values('running_id', 'running_date', 'start_time', 'end_time'))
                    result = create_calendar(runnings)
                    return render(request, 'running_log.html', {"result": result, "uesr": user_})
                    # return Response({"result": True, "runnings": runnings})
                else:
                    return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
            else:
                return Response({"result": False, "error": "토큰이 없습니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})

    # 달리기 시작 버튼, 종료 버튼
    def post(self, request, user_id):
        print("post")
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                try:
                    state = RunningState.objects.get(user=user_)
                except:
                    state = None
                # 멈춤
                if state is not None:
                    print("state is not None")
                    state.delete()
                    print("delete")
                    # NOTE: RunningSaveView 호출해서 DB 저장..!!!
                    data = json.loads(request.body.decode('utf-8'))
                    print("data : ", data)
                    running_id = save_running_state(user_id, data)
                    print("run id : ", running_id)
                    if running_id != -1:
                        return Response({"result": True, "state": False, "running_id": state.id})
                    return Response({"result": False, "error": "데이터 저장에 실패했습니다.", "running_id": -1})
                # 시작
                else:
                    new_state = RunningState(user=user_, username=user_id, state=True)
                    new_state.save()
                    return Response({"result": True, "state": True})
            else:
                return Response({"result": True, "error": "인증 되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


def save_running_state(user_id, data):
    user_ = User.objects.filter(username=user_id).values('id')[0]['id']
    try:
        print(user_)
        new_running = Running(
            running_date=data['running_date'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            heart_rate=data['heart_rate'],
            steps=data['steps'],
            user_id=user_
        )

        new_running.save()
        return new_running.running_id
    except Exception as e:
        return -1


# 달리기 종료 버튼 누르면 실행
# class RunningSaveView(APIView):
#     def post(self, request, user_id):
#         user_ = User.objects.filter(username=user_id).values('id')[0]['id']
#         data = json.loads(request.body.decode('utf-8'))
#         if user_ is not None:
#             if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
#                 print(user_)
#                 new_running = Running(
#                     running_date=data['running_date'],
#                     start_time=data['start_time'],
#                     end_time=data['end_time'],
#                     heart_rate=data['heart_rate'],
#                     steps=data['steps'],
#                     user_id=user_
#                 )
#
#                 new_running.save()
#                 return Response({"result": True})
#             else:
#                 return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
#         else:
#             return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


# 러닝 데이터 전부 불러오기
class RunningInfoView(APIView):
    def get(self, request, user_id, running_id):
        user_ = User.objects.get(username=user_id)
        print(user_)
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                running_info = Running.objects.get(running_id=running_id)
                running_info = RunningInfoSerializer(running_info).data

                context = {}
                print(running_info)
                print(type(running_info['end_time']))
                if running_info:
                    start_time = datetime.strptime(running_info['start_time'], '%Y-%m-%dT%H:%M:%SZ')
                    end_time = datetime.strptime(running_info['end_time'], '%Y-%m-%dT%H:%M:%SZ')
                    elapsed_time = start_time - end_time
                    context = {
                        'user': user_,
                        'running_date': running_info['running_date'],
                        'running_time': elapsed_time,
                        'heart_rate': running_info['heart_rate'],
                        'steps': running_info['steps']
                    }
                return render(request, 'detail.html', context)
                # return Response({"result": True, "running_info": running_info})
            else:
                return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


class FeedbackView(APIView):
    def get(self, request):
        feedbacks = Feedback.objects.all()
        context = {"발목": [], "무릎": [], "골반": [], "상체": [], "시선": [], "팔꿈치": []}
        for feedback in feedbacks:
            context[feedback.type].append((feedback.id, feedback.name))
        return render(request, 'feedback.html', {'feedbacks': context})


class FeedbackDetailView(APIView):
    def get(self, request, feed_id):
        feedback = Feedback.objects.get(id=feed_id)
        return render(request, 'feedback_detail.html', {"feedback": feedback})


# class RunningLogView(APIView):
#     def get(self, request, user_id):
#         return render(request, 'running_log.html')


def video_stream(user_id):
    while True:
        # 공유 메모리 열기
        try:
            my_shm = shared_memory.SharedMemory(name=f"running_{user_id}_frame")
        except:
            break

        # 공유 메모리에서 데이터를 NumPy 배열로 읽기
        shm_array = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=my_shm.buf)
        # NumPy 배열을 이미지로 변환
        image = np.reshape(shm_array, (480, 640, 3))  # 예시 이미지 크기 (높이, 너비, 채널)
      
        # JPEG 인코딩
        success, jpeg = cv2.imencode(".jpg", image)
        # 공유 메모리 닫기
        my_shm.close()
        if not success:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # time.sleep(0.05)  # 0.1초 대기


def audio_view(request):
    return render(request, 'app1/audio_template.html')


def get_next_audio(request):
    # 오디오 파일 목록
    audio_files = [
        'audio/audio1.mp3',
        'audio/audio2.mp3',
        'audio/audio3.mp3',
        'None'
    ]
    # 랜덤하게 오디오 파일 선택
    next_audio = random.choice(audio_files)
    return JsonResponse({'next_audio': next_audio})



class AudioStreamingView(APIView):
    def get(self, request, user_id):
        return render(request, 'audio_streaming.html')
        # return StreamingHttpResponse(audio_stream(user_id)(), content_type="audio/mpeg")


class StreamingView(APIView):
    def get(self, request, user_id):
        return StreamingHttpResponse(video_stream(user_id), content_type="multipart/x-mixed-replace; boundary=frame")


class StreamingRunning(APIView):
    def get(self, request, user_id):
        # 원하는 컨텍스트 데이터 설정
        context = {
            'user_id': user_id,
            # 다른 필요한 컨텍스트 데이터 추가 가능
        }
        # 템플릿을 렌더링하여 HTML 생성
        return render(request, 'streaming.html', context)
