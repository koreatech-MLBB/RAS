from django.contrib import admin
from . import models

admin.site.register(models.Profile)
admin.site.register(models.RunningState)
admin.site.register(models.Running)
admin.site.register(models.Pose)
admin.site.register(models.Feedback)
