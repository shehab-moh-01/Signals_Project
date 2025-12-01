import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Global state to store processing results
state = {
    'audio_path': None,
    'sample_rate': None,
    'audio_data': None,
    'time': None,
    'freq': None,
    'fft_data': None,
    'filtered_data': None
}

def generate_noisy_audio():
    """Generate a sample noisy audio file for testing."""
    t = np.linspace(0, 5, 5 * 44100, False)  # 5 seconds at 44.1kHz
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = 0.2 * np.random.normal(0, 1, t.size)  # White noise
    noisy_signal = signal + noise
    noisy_signal = (noisy_signal / np.max(np.abs(noisy_signal)) * 32767).astype(np.int16)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_noisy.wav')
    static_path = os.path.join(app.config['STATIC_FOLDER'], 'sample_noisy.wav')
    wavfile.write(upload_path, 44100, noisy_signal)
    shutil.copy(upload_path, static_path)  # Copy to static for web access
    return static_path

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        # Generate sample noisy audio if no file uploaded
        state['audio_path'] = generate_noisy_audio()
    else:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        state['audio_path'] = os.path.join(app.config['STATIC_FOLDER'], filename)
        file.save(upload_path)
        shutil.copy(upload_path, state['audio_path'])  # Copy to static
    
    # Read and process audio
    state['sample_rate'], state['audio_data'] = wavfile.read(state['audio_path'])
    if len(state['audio_data'].shape) > 1:
        state['audio_data'] = state['audio_data'][:, 0]  # Use first channel for stereo
    state['audio_data'] = state['audio_data'].astype(float) / 32767.0
    state['time'] = np.arange(len(state['audio_data'])) / state['sample_rate']
    
    # Plot original audio
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(state['time'], state['audio_data'], color='#60a5fa')
    ax.set_title('Original Audio', color='white')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.grid(True, color='gray')
    ax.set_facecolor('#1f2937')
    fig.set_facecolor('#1f2937')
    ax.tick_params(colors='white')
    original_plot = plot_to_base64(fig)
    plt.close(fig)
    
    return render_template('step1.html', audio_path=state['audio_path'].split(os.sep)[-1], plot=original_plot)

@app.route('/step2')
def step2():
    # Downsample audio (e.g., take every 2nd sample)
    downsampled_data = state['audio_data'][::2]
    downsampled_time = state['time'][::2]
    
    # Plot downsampled audio
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(downsampled_time, downsampled_data, color='#60a5fa')
    ax.set_title('Downsampled Audio', color='white')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.grid(True, color='gray')
    ax.set_facecolor('#1f2937')
    fig.set_facecolor('#1f2937')
    ax.tick_params(colors='white')
    downsampled_plot = plot_to_base64(fig)
    plt.close(fig)
    
    return render_template('step2.html', plot=downsampled_plot)

@app.route('/step3')
def step3():
    # Compute FFT
    n = len(state['audio_data'])
    state['fft_data'] = fft(state['audio_data'])
    state['freq'] = np.fft.fftfreq(n, 1/state['sample_rate'])
    fft_magnitude = np.abs(state['fft_data'])[:n//2]
    freq_positive = state['freq'][:n//2]
    
    # Plot FFT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_positive, fft_magnitude, color='#60a5fa')
    ax.set_title('Fourier Transform', color='white')
    ax.set_xlabel('Frequency (Hz)', color='white')
    ax.set_ylabel('Magnitude', color='white')
    ax.grid(True, color='gray')
    ax.set_facecolor('#1f2937')
    fig.set_facecolor('#1f2937')
    ax.tick_params(colors='white')
    fft_plot = plot_to_base64(fig)
    plt.close(fig)
    
    return render_template('step3.html', plot=fft_plot)

@app.route('/step4')
def step4():
    # Apply low-pass Butterworth filter
    cutoff = 1000  # Hz
    nyquist = state['sample_rate'] / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    state['filtered_data'] = filtfilt(b, a, state['audio_data'])
    
    # Save filtered audio
    filtered_filename = 'filtered_audio.wav'
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)
    static_path = os.path.join(app.config['STATIC_FOLDER'], filtered_filename)
    wavfile.write(upload_path, state['sample_rate'], (state['filtered_data'] * 32767 *2).astype(np.int16))
    shutil.copy(upload_path, static_path)
    
    # Plot filtered audio
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(state['time'], state['filtered_data'], color='#60a5fa')
    ax.set_title('Filtered Audio (Noise Reduced)', color='white')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.grid(True, color='gray')
    ax.set_facecolor('#1f2937')
    fig.set_facecolor('#1f2937')
    ax.tick_params(colors='white')
    filtered_plot = plot_to_base64(fig)
    plt.close(fig)
    
    return render_template('step4.html', audio_path=filtered_filename, plot=filtered_plot)

if __name__ == '__main__':
    app.run(debug=True)