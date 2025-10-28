"""
Flask API cho Heart Sound Classification
Sử dụng model 1DCNN đã train
"""

import os
import sys
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scipy.signal import butter, filtfilt
from kymatio.numpy import Scattering1D
import pandas as pd

# Import Keras (Keras 3 compatibility)
try:
    # Keras 3.x (standalone)
    from keras.models import load_model
except ImportError:
    # Keras 2.x (bundled with TensorFlow)
    from tensorflow.keras.models import load_model

# Add src to path
sys.path.append('../../src')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
MODEL_PATH = '../../results/models/1dcnn_method_best.h5'

# Model parameters
SR = 4000
MAX_LEN = 3000
CUTOFF_FREQ = 500
SCATTERING_J = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Initialize Scattering Transform
scattering = Scattering1D(J=SCATTERING_J, shape=MAX_LEN)

# Class names
CLASS_NAMES = ['Absent', 'Present']


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def butter_lowpass_filter(data, cutoff_freq, sr, order=5):
    """Apply Butterworth low-pass filter"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def normalize_signal(sig):
    """Normalize signal (zero mean, unit variance)"""
    sig = sig - np.mean(sig)
    std = np.std(sig)
    return sig / std if std > 0 else sig


def extract_cycles_from_audio(wav_path, tsv_path=None):
    """
    Extract heart sound cycles from audio file
    If TSV file is provided, use segmentation. Otherwise, use sliding window.
    """
    # Load signal
    signal, _ = librosa.load(wav_path, sr=SR)
    
    # Apply Butterworth low-pass filter
    signal = butter_lowpass_filter(signal, CUTOFF_FREQ, SR, order=5)
    
    cycles = []
    
    if tsv_path and os.path.exists(tsv_path):
        # Use segmentation from TSV
        df_seg = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'label'])
        
        for i in range(len(df_seg) - 3):
            labels_seq = df_seg.iloc[i:i+4]['label'].tolist()
            if labels_seq == [1, 2, 3, 4]:
                start = int(df_seg.iloc[i]['start'] * SR)
                end = int(df_seg.iloc[i+3]['end'] * SR)
                cycle = signal[start:end]
                
                if len(cycle) > 100:
                    cycle = normalize_signal(cycle)
                    cycles.append(cycle)
    else:
        # Use sliding window approach (fallback)
        # Typical heart cycle is 0.5-1.5 seconds
        window_size = int(1.0 * SR)  # 1 second window
        hop_size = int(0.5 * SR)  # 50% overlap
        
        for i in range(0, len(signal) - window_size, hop_size):
            cycle = signal[i:i + window_size]
            cycle = normalize_signal(cycle)
            cycles.append(cycle)
    
    return cycles


def extract_scattering_features(signal):
    """Extract Scattering features from signal"""
    # Pad or truncate to MAX_LEN
    if len(signal) < MAX_LEN:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
    else:
        signal = signal[:MAX_LEN]
    
    features = scattering(signal)
    return features


def predict_from_audio(wav_path, tsv_path=None):
    """
    Predict heart murmur from audio file
    Returns prediction and confidence
    """
    # Extract cycles
    cycles = extract_cycles_from_audio(wav_path, tsv_path)
    
    if len(cycles) == 0:
        return None, None, "No valid heart cycles detected"
    
    # Extract features for all cycles
    features_list = []
    for cycle in cycles:
        features = extract_scattering_features(cycle)
        features_list.append(features)
    
    X = np.array(features_list)
    
    # Predict
    predictions = model.predict(X, verbose=0)
    
    # Average predictions across all cycles
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    confidence = float(avg_prediction[predicted_class])
    
    return CLASS_NAMES[predicted_class], confidence, None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict heart murmur from uploaded audio file
    Accepts: WAV file (required), TSV file (optional)
    """
    # Check if audio file is present
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type. Only WAV files are allowed'}), 400
    
    try:
        # Save audio file
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Check for optional TSV file
        tsv_path = None
        if 'segmentation' in request.files:
            tsv_file = request.files['segmentation']
            if tsv_file.filename != '':
                tsv_filename = secure_filename(tsv_file.filename)
                tsv_path = os.path.join(app.config['UPLOAD_FOLDER'], tsv_filename)
                tsv_file.save(tsv_path)
        
        # Make prediction
        prediction, confidence, error = predict_from_audio(audio_path, tsv_path)
        
        # Clean up uploaded files
        os.remove(audio_path)
        if tsv_path and os.path.exists(tsv_path):
            os.remove(tsv_path)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'Absent': float(1 - confidence) if prediction == 'Present' else float(confidence),
                'Present': float(confidence) if prediction == 'Present' else float(1 - confidence)
            }
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if tsv_path and os.path.exists(tsv_path):
            os.remove(tsv_path)
        
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': '1D CNN Heart Sound Classifier',
        'input_shape': [MAX_LEN],
        'sampling_rate': SR,
        'classes': CLASS_NAMES,
        'accuracy': 0.8346,  # From training results
        'architecture': '2 Conv blocks + Dense layers',
        'features': 'Scattering Transform (J=6)'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Heart Sound Classification API")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Sampling Rate: {SR} Hz")
    print(f"Max Length: {MAX_LEN} samples")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
