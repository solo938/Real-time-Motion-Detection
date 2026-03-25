from flask import Blueprint, render_template, request, flash, send_from_directory, current_app
from werkzeug.utils import secure_filename
import os

upload_bp = Blueprint('upload', __name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@upload_bp.route('/', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file part')
        return render_template('index.html', video="", is_video_display=False), 400

    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return render_template('index.html', video="", is_video_display=False), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', video=filename, is_video_display=True)

    flash('Invalid file type. Only .mp4 allowed.')
    return render_template('index.html', video="", is_video_display=False), 400


@upload_bp.route('/sample', methods=['POST'])
def use_sample():
    return render_template('index.html', video="sample_video.mp4", is_video_display=True)


@upload_bp.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)