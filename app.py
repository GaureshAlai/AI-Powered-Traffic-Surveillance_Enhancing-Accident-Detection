import os
import torch
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import tempfile
import uuid
import threading
import logging
import time

# Import your model class from the existing script
from accident_detection_model import AccidentDetectionModel, DEVICE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (lazy initialization)
model = None
transform = None

# Dictionary to track processing status
processing_tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the model only once when needed"""
    global model, transform
    if model is None:
        logger.info("Loading model...")
        model = AccidentDetectionModel().to(DEVICE)
        model.load_state_dict(torch.load('best_accident_detection_model.pth', map_location=DEVICE))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"Model loaded successfully on {DEVICE}")

def analyze_video_task(video_path, output_path, task_id, window_size=5, threshold=0.6):
    """Process video in a separate thread and update task status"""
    global model, transform
    
    # Ensure model is loaded
    if model is None:
        load_model()
    
    try:
        processing_tasks[task_id]['status'] = 'processing'
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        frame_count = 0
        accident_frames = []
        recent_predictions = []  # For temporal smoothing
        
        logger.info(f"Processing video: {video_path}, total frames: {total_frames}")
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if frame_count % 10 == 0:
                progress = int((frame_count / total_frames) * 100)
                processing_tasks[task_id]['progress'] = progress
                
            # Convert frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Preprocess and get prediction
            img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                accident_prob = probs[0][1].item()  # Probability of accident class
                recent_predictions.append(accident_prob)
            
            # Temporal smoothing (moving average)
            if len(recent_predictions) > window_size:
                recent_predictions.pop(0)
            smoothed_prob = sum(recent_predictions) / len(recent_predictions)
            
            # Detect accident
            is_accident = smoothed_prob > threshold
            
            if is_accident:
                accident_frames.append(frame_count)
                # Add visual indicator (red border)
                cv2.putText(frame, f"ACCIDENT DETECTED! {smoothed_prob:.2f}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
            else:
                # Add confidence score
                cv2.putText(frame, f"No accident: {smoothed_prob:.2f}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add frame number and timestamp
            timestamp = frame_count / fps
            cv2.putText(frame, f"Frame: {frame_count} | Time: {timestamp:.2f}s", 
                        (50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        total_time = total_frames / fps
        accident_seconds = len(set([int(frame / fps) for frame in accident_frames]))
        
        # Update task with results
        processing_tasks[task_id].update({
            'status': 'completed',
            'progress': 100,
            'total_frames': total_frames,
            'processing_time': processing_time,
            'total_duration': total_time,
            'accident_seconds': accident_seconds,
            'accident_percentage': (accident_seconds / total_time * 100) if total_time > 0 else 0,
            'processed_fps': frame_count / processing_time
        })
        
        logger.info(f"Video processing complete: {task_id}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this task
        task_id = str(uuid.uuid4())
        
        # Save original file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # Define output path
        output_filename = f"{task_id}_result_{filename}"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Initialize task status
        processing_tasks[task_id] = {
            'status': 'queued',
            'progress': 0,
            'original_file': filename,
            'file_path': file_path,
            'output_path': output_path,
            'output_filename': output_filename,
            'threshold': float(request.form.get('threshold', 0.6))
        }
        
        # Start processing in a separate thread
        threading.Thread(
            target=analyze_video_task,
            args=(file_path, output_path, task_id, 5, float(request.form.get('threshold', 0.6)))
        ).start()
        
        return redirect(url_for('result', task_id=task_id))
    
    return redirect(request.url)

@app.route('/result/<task_id>')
def result(task_id):
    if task_id not in processing_tasks:
        return redirect(url_for('index'))
    
    return render_template('result.html', task_id=task_id)

@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in processing_tasks:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify(processing_tasks[task_id])

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/clear/<task_id>', methods=['POST'])
def clear_task(task_id):
    if task_id in processing_tasks:
        # Remove files to free space
        try:
            if os.path.exists(processing_tasks[task_id]['file_path']):
                os.remove(processing_tasks[task_id]['file_path'])
            if os.path.exists(processing_tasks[task_id]['output_path']):
                os.remove(processing_tasks[task_id]['output_path'])
        except Exception as e:
            logger.error(f"Error removing files: {str(e)}")
            
        # Remove task from dict
        del processing_tasks[task_id]
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Pre-load model in the main thread
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
