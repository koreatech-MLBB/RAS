from django.contrib import admin
# from django.contrib.auth.models import User
from . import models

# admin.site.register(User)
admin.site.register(models.Profile)
admin.site.register(models.RunningState)
admin.site.register(models.Running)
admin.site.register(models.Pose)

# Register your models here.
