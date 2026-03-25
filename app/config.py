import os
from pathlib import Path

basedir = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get('HF TOKEN') 
    UPLOAD_FOLDER = basedir / 'uploads'
    ALLOWED_EXTENSIONS = {'mp4'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  
    DEBUG = False

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    pass


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}