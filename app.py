from flask import Flask, render_template, request, Response, jsonify, url_for
import cv2
import cvlib
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.backend import clear_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=2)

# Dictionary of available models with their file names
models = {
    'MYCNN': 'mycnn.h5',
    'InceptionV3': 'Inceptionv3.h5',
    'ResNet50': 'ResNet50.h5',
    'ResNet152v2': 'resnet152v2.h5',
    'VGG16': 'VGG16.h5',
    'VGG19': 'VGG19.h5',
    'DenseNet201': 'densenet201.h5',
    'EthnicityModel': 'ethnicity_model.h5'
    
}

# Cache for loaded models to avoid reloading
model_cache = {}

def get_models(model_name):
    clear_session()
    """Load and return main model and ethnicity model from cache or disk."""
    if model_name in model_cache:
        main_model = model_cache[model_name]
    else:
        model_path = models[model_name]
        main_model = load_model(model_path)
        model_cache[model_name] = main_model
        print(f"{model_name} Model loaded")
    
    # Load the ethnicity model
    if 'EthnicityModel' in model_cache:
        ethnicity_model = model_cache['EthnicityModel']
    else:
        ethnicity_model_path = models['EthnicityModel']
        ethnicity_model = load_model(ethnicity_model_path)
        model_cache['EthnicityModel'] = ethnicity_model
        print("Ethnicity Model loaded")
    
    return main_model, ethnicity_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    model_name = request.form.get('model', 'MYCNN')
    main_model, ethnicity_model = get_models(model_name)
    
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 422
    
    faces = cvlib.detect_face(img)[0]  
    if not faces:
        return jsonify({'error': 'No faces detected'}), 404

    img = draw_labels(img, faces, main_model, ethnicity_model)

    output_filename = 'detected.jpg'
    cv2.imwrite(os.path.join('static', output_filename), img)
    return render_template('result.html', image_file=output_filename, model_name=model_name)


@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.files['video']
    if not video_file:
        return jsonify({'error': 'No video provided'}), 400

    model_name = request.form.get('model', 'MYCNN')
    main_model, ethnicity_model = get_models(model_name)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    video_file.save(temp_path)

    try:
        future = executor.submit(process_video_file, temp_path, main_model, ethnicity_model)
        result = future.result()
        os.remove(temp_path)
        return jsonify({'redirect': url_for('result_video', filename=result, model_name=model_name)})
    except Exception as e:
        os.remove(temp_path)
        print(f"Error processing video: {str(e)}")
        return jsonify({'error': 'Failed to process video'}), 500

@app.route('/resultvideo')
def result_video():
    video_file = request.args.get('filename', 'processed_video.mp4')
    model_name = request.args.get('model_name', 'Unknown Model')
    return render_template('resultvideo.html', video_file=video_file, model_name=model_name)

def process_video_file(video_path, main_model, ethnicity_model):
    """Process video file to detect faces and label them using cvlib."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('static/processed_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = cvlib.detect_face(frame)[0]  
            frame = draw_labels(frame, faces, main_model, ethnicity_model)
            out.write(frame)
    finally:
        cap.release()
        out.release()

def draw_labels(img, faces, main_model, ethnicity_model):
    color_male = (255, 0, 0)  # Blue color in BGR
    color_female = (147, 20, 255)  # Rose color in BGR
    ethnicity_dict = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

    for (startx, starty, endx, endy) in faces:
        face_crop = img[starty:endy, startx:endx]
        if face_crop.size == 0:
            continue
        face_crop_resized = cv2.resize(face_crop, (128, 128))
        face_crop_resized = np.expand_dims(face_crop_resized, axis=0) / 255.0
        
        # Predict age, gender, and ethnicity
        pred = main_model.predict(face_crop_resized)
        ethnicity_pred = ethnicity_model.predict(face_crop_resized)
        
        age = int(np.round(pred[1][0]))
        sex = int(np.round(pred[0][0]))
        ethnicity = np.argmax(ethnicity_pred, axis=1)[0]
        
        gender_label = "Male" if sex == 0 else "Female"
        ethnicity_label = ethnicity_dict[ethnicity]
        
        color = color_male if sex == 0 else color_female
        label = f"{gender_label}, {age} years, {ethnicity_label}"
        
        y = starty - 10 if starty - 10 > 10 else starty + 10
        cv2.rectangle(img, (startx, starty), (endx, endy), color, 2)
        cv2.putText(img, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img


@app.route('/video_feed')
def video_feed():
    model_name = request.args.get('model', 'MYCNN')  # Get the selected model name
    return Response(video_stream(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')


def video_stream(model_name):
    """Video streaming generator function with face detection and labeling."""
    cap = cv2.VideoCapture(0)  # Use 0 for web camera
    main_model, ethnicity_model = get_models(model_name)  # Load both models
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = cvlib.detect_face(frame)[0]  
        frame = draw_labels(frame, faces, main_model, ethnicity_model)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
