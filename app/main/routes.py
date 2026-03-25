from flask import Blueprint, render_template, current_app

main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['GET'])
def index():
    video = "sample_video.mp4"
    return render_template('index.html', video=video, is_video_display=False)

