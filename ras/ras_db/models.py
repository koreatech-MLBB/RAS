# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Pose(models.Model):
    running = models.ForeignKey('Running', models.CASCADE)
    pose_score = models.FloatField()
    pose_photo = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'pose'


class Running(models.Model):
    running_id = models.IntegerField(primary_key=True)
    user = models.ForeignKey('User', models.CASCADE)
    running_date = models.DateField()
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    average_speed = models.FloatField()

    class Meta:
        managed = False
        db_table = 'running'


class User(models.Model):
    user_id = models.CharField(primary_key=True, max_length=255)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    weight = models.FloatField()
    height = models.FloatField()
    gender = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'user'
