from flask import Flask, render_template, Response, request, session, redirect, url_for, send_file, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
import numpy as np
from YOLO_Video import video_detection
from YOLO_Image import image_detection

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sksiraj'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Form for uploading video or image
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

# Video frame generation for detection
def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Webcam feed generation
def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Image detection route
@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection_route():
    if request.method == 'POST':
        image_file = request.files['image_file']
        if image_file:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            # Perform object detection on the image
            detected_image = image_detection(image_path)
            
            # Convert detected image to a format that can be displayed
            _, buffer = cv2.imencode('.jpg', detected_image)
            image_bytes = buffer.tobytes()

            # Send the image data as JSON response
            return jsonify({'image_data': image_bytes.decode('latin1')})

    return render_template('image_detection.html')

@app.route('/image_detection/<path:image_path>')
def display_image_route(image_path):
    detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], image_path)
    if os.path.exists(detected_image_path):
        return send_file(detected_image_path, mimetype='image/jpeg')
    else:
        return "File not found", 404

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        session['file_path'] = filepath
        session['file_type'] = 'video' if file.filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image'
    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    if session.get('file_type') == 'video':
        return Response(generate_frames(path_x=session.get('file_path', None)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Error: Not a video file"

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
