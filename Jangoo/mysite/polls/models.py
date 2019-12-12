from django.db import models


class UserMessage(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')


class BotMessage(models.Model):
    Answer_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')