from django.apps import AppConfig
import html
import pathlib
import os


class PollsConfig(AppConfig):
    name = 'polls'
    MODEL_PATH = Path("model")

