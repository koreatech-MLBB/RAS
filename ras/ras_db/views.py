import json
from multiprocessing import shared_memory

import random

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

# Create your views here.


class SignUpView(APIView):
    def post(self, request):
        data = json.loads(request.body.decode('utf-8'))
        try:
            id_ = User.objects.get(username=data['user_id'])
        except:
            id_ = None

        if id_ is not None:
            return Response({"result": False, "Error": "이미 존재 하는 아이디 입니다."})
        else:
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
    def post(self, request):
        data = json.loads(request.body.decode('utf-8'))
        user = authenticate(username=data['user_id'], password=data['password'])
        if user is not None:
            token = Token.objects.get(user=user)
            return Response({"result": True, "Token": token.key})
        else:
            return Response({"result": False})


class RunningView(APIView):
    def post(self, request, user_id):
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                try:
                    state = RunningState.objects.get(user=user_)
                except:
                    state = None
                if state is not None:
                    state.delete()
                    return Response({"result": True, "state": False})
                else:
                    new_state = RunningState(user=user_, username=user_id, state=True)
                    new_state.save()
                    return Response({"result": True, "state": True})
            else:
                return Response({"result": True, "error": "인증 되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})

    def get(self, request, user_id):
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if 'HTTP_AUTHORIZATION' in request.META:
                if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                    runnings = Running.objects.filter(user=user_).values('running_id', 'running_date')
                    return Response({"result": True, "runnings": runnings})
                else:
                    return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
            else:
                return Response({"result": False, "error": "토큰이 없습니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


class RunningSaveView(APIView):
    def post(self, request, user_id):
        user_ = User.objects.filter(username=user_id).values('id')[0]['id']
        data = json.loads(request.body.decode('utf-8'))
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
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
                return Response({"result": True})
            else:
                return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})


class RunningInfoView(APIView):
    def get(self, request, user_id, running_id):
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                running_info = Running.objects.get(running_id=running_id)
                running_info = RunningInfoSerializer(running_info).data
                return Response({"result": True, "running_info": running_info})
            else:
                return Response({"result": False, "error": "인증 되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재 하지 않는 사용자 입니다."})

def video_stream(user_id):


    while True:
        # 공유 메모리 열기
        try:
            my_shm = shared_memory.SharedMemory(name=f"running_{user_id}_frame")
        except:
            break

        # 공유 메모리에서 데이터를 NumPy 배열로 읽기
        shm_array = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=my_shm.buf)

        # NumPy 배열을 이미지로 변환
        image = np.reshape(shm_array, (720, 1280, 3))  # 예시 이미지 크기 (높이, 너비, 채널)

        # JPEG 인코딩
        success, jpeg = cv2.imencode(".jpg", image)

        # 공유 메모리 닫기
        my_shm.close()
        if not success:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # time.sleep(0.05)  # 0.1초 대기

def audio_view(request):
    return render(request, 'audio_template.html')

class next_audio(APIView):
    def get(self, request, user_id):
        try:
            my_shm = shared_memory.SharedMemory(name=f"running_{user_id}_feedback")
        except:
            return JsonResponse({'next_audio': 'None'})
    # 공유 메모리에서 데이터를 NumPy 배열로 읽기
        b = np.array(["a" * 40], dtype=np.dtype('U40'))
        next_audio = np.ndarray(b.shape, dtype=np.dtype('U40'), buffer=my_shm.buf)
        res = str(next_audio)
        # print("in view", next_audio)
        # 공유 메모리 닫기
        my_shm.close()
        # 랜덤하게 오디오 파일 선택
        return JsonResponse({'next_audio': res})

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
