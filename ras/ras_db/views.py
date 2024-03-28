import json

from django.contrib.auth import authenticate
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
            return Response({"result": False, "Error": "이미 존재하는 아이디 입니다."})
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

            token = Token.objects.create(user=new_user)
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
                    new_state = RunningState(user=user_, state=True)
                    new_state.save()
                    return Response({"result": True, "state": True})
            else:
                return Response({"result": True, "error": "인증되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재하지 않는 사용자 입니다."})

    def get(self, request, user_id):
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if 'HTTP_AUTHORIZATION' in request.META:
                if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                    runnings = Running.objects.filter(user=user_).values('running_id', 'running_date')
                    return Response({"result": True, "runnings": runnings})
                else:
                    return Response({"result": False, "error": "인증되지 않은 토큰 입니다."})
            else:
                return Response({"result": False, "error": "토큰이 없습니다."})
        else:
            return Response({"result": False, "error": "존재하지 않는 사용자 입니다."})

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
                return Response({"result": False, "error": "인증되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재하지 않는 사용자 입니다."})


class RunningInfoView(APIView):
    def get(self, request, user_id, running_id):
        user_ = User.objects.get(username=user_id)
        if user_ is not None:
            if request.META['HTTP_AUTHORIZATION'].split()[1] == Token.objects.get(user=user_).key:
                running_info = Running.objects.get(running_id=running_id)
                running_info = RunningInfoSerializer(running_info).data
                return Response({"result": True, "running_info": running_info})
            else:
                return Response({"result": False, "error": "인증되지 않은 토큰 입니다."})
        else:
            return Response({"result": False, "error": "존재하지 않는 사용자 입니다."})