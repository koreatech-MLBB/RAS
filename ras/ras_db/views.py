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

# Create your views here.


class SignUp(APIView):
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


class SignIn(APIView):
    def post(self, request):
        data = json.loads(request.body.decode('utf-8'))
        user = authenticate(username=data['user_id'], password=data['password'])
        if user is not None:
            token = Token.objects.get(user=user)
            return Response({"result": True, "Token": token.key})
        else:
            return Response({"result": False})

