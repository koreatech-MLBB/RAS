# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.contrib.auth.models import User
from django.db import models
from rest_framework import serializers


class Pose(models.Model):
    running = models.ForeignKey('Running', models.CASCADE)
    pose_score = models.FloatField()
    pose_photo = models.CharField(max_length=255)

    class Meta:
        managed = True
        db_table = 'pose'


class Running(models.Model):
    running_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE)
    running_date = models.DateField()
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    heart_rate = models.IntegerField()
    steps = models.IntegerField()

    class Meta:
        managed = True
        db_table = 'running'


class RunningState(models.Model):
    user = models.OneToOneField(User, models.CASCADE)
    username = models.CharField(max_length=255)
    state = models.IntegerField()

    class Meta:
        managed = True
        db_table = 'running_state'


class Profile(models.Model):
    id = models.OneToOneField(User, models.CASCADE, primary_key=True)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    weight = models.FloatField()
    height = models.FloatField()
    gender = models.IntegerField()

    class Meta:
        managed = True
        db_table = 'profile'


class Feedback(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    explain = models.TextField(max_length=1024)
    type = models.CharField(max_length=25)
    video_path = models.FileField(upload_to='videos/', null=True, verbose_name="")

    class Meta:
        managed = True
        db_table = 'feedback'

