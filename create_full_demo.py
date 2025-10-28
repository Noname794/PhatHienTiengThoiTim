#!/usr/bin/env python3
"""
Script tạo notebook demo đầy đủ cho data_preprocessing.py
"""

import json

def create_cell(cell_type, content, execution_count=None):
    """Helper function để tạo cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

# Tạo notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# ===== CELLS =====

# Title
notebook["cells"].append(create_cell("markdown", [
    "# 🎵 Demo: HeartSoundPreprocessor\n",
    "\n",
    "Notebook này demo từng bước xử lý trong `data_preprocessing.py`\n",
    "\n",
    "## 📋 Nội dung:\n",
    "1. ✅ Khởi tạo HeartSoundPreprocessor\n",
    "2. 🎵 Load và visualize 1 file WAV\n",
    "3. 🔊 Apply Butterworth filter\n",
    "4. 💓 Extract cardiac cycles từ TSV\n",
    "5. 📊 Normalize cycles\n",
    "6. 🌊 Extract Scattering features\n",
    "7. 📈 Visualize toàn bộ pipeline\n",
    "\n",
    "---"
]))

# Imports
notebook["cells"].append(create_cell("markdown", ["## 1. Import Libraries"]))
notebook["cells"].append(create_cell("code", [
    "import os\n",
    "import sys\n",
    "import glob\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import librosa\n",
    "import librosa.display\n",
    "from scipy import signal as scipy_signal\n",
    "\n",
    "# Import HeartSoundPreprocessor\n",
    "sys.path.append('../src')\n",
    "from data_preprocessing import HeartSoundPreprocessor\n",
    "\n",
    "# Setup plotting\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('viridis')\n",
    "plt.rcParams['figure.figsize'] = (15, 5)\n",
    "%matplotlib inline\n",
    "\n",
    "print(\"✅ Đã import thành công!\")"
]))

# Config
notebook["cells"].append(create_cell("markdown", ["## 2. Cấu hình Tham Số"]))
notebook["cells"].append(create_cell("code", [
    "# Tham số (giống 04_training_cnn_method và 05_training_lstm_method)\n",
    "SR = 4000           # Sampling rate\n",
    "MAX_LEN = 3000      # Độ dài cycle tối đa\n",
    "CUTOFF_FREQ = 500   # Butterworth filter cutoff\n",
    "J = 6               # Scattering depth\n",
    "\n",
    "# Đường dẫn\n",
    "RAW_DATA_DIR = '../data/raw/training_data/'\n",
    "METADATA_FILE = '../data/raw/training_data.csv'\n",
    "\n",
    "print(f\"⚙️ Cấu hình:\")\n",
    "print(f\"   - SR = {SR} Hz\")\n",
    "print(f\"   - MAX_LEN = {MAX_LEN} samples ({MAX_LEN/SR:.2f}s)\")\n",
    "print(f\"   - CUTOFF_FREQ = {CUTOFF_FREQ} Hz\")\n",
    "print(f\"   - Scattering J = {J}\")\n",
    "print(f\"\\n📁 Data directory: {os.path.abspath(RAW_DATA_DIR)}\")"
]))

# Initialize
notebook["cells"].append(create_cell("markdown", ["## 3. Khởi tạo HeartSoundPreprocessor"]))
notebook["cells"].append(create_cell("code", [
    "# Khởi tạo preprocessor\n",
    "preprocessor = HeartSoundPreprocessor(\n",
    "    sr=SR,\n",
    "    max_len=MAX_LEN,\n",
    "    cutoff_freq=CUTOFF_FREQ,\n",
    "    scattering_j=J\n",
    ")\n",
    "\n",
    "print(\"✅ Đã khởi tạo HeartSoundPreprocessor!\")\n",
    "print(f\"\\n📊 Thông tin:\")\n",
    "print(f\"   - Sampling rate: {preprocessor.sr} Hz\")\n",
    "print(f\"   - Max length: {preprocessor.max_len} samples\")\n",
    "print(f\"   - Cutoff frequency: {preprocessor.cutoff_freq} Hz\")\n",
    "print(f\"   - Scattering J: {preprocessor.scattering_j}\")"
]))

# Find sample file
notebook["cells"].append(create_cell("markdown", ["## 4. Tìm File Mẫu để Demo"]))
notebook["cells"].append(create_cell("code", [
    "# Tìm 1 file WAV và TSV để demo\n",
    "wav_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.wav'))\n",
    "\n",
    "if len(wav_files) == 0:\n",
    "    print(\"❌ Không tìm thấy file WAV!\")\n",
    "    print(f\"   Kiểm tra đường dẫn: {RAW_DATA_DIR}\")\n",
    "else:\n",
    "    # Chọn file đầu tiên\n",
    "    sample_wav = wav_files[0]\n",
    "    sample_tsv = sample_wav.replace('.wav', '.tsv')\n",
    "    \n",
    "    print(f\"✅ Tìm thấy {len(wav_files)} files WAV\")\n",
    "    print(f\"\\n📄 File demo:\")\n",
    "    print(f\"   WAV: {os.path.basename(sample_wav)}\")\n",
    "    print(f\"   TSV: {os.path.basename(sample_tsv)}\")\n",
    "    print(f\"   Exists: WAV={os.path.exists(sample_wav)}, TSV={os.path.exists(sample_tsv)}\")"
]))

# Load raw signal
notebook["cells"].append(create_cell("markdown", ["## 5. Load Raw Signal"]))
notebook["cells"].append(create_cell("code", [
    "# Load signal gốc\n",
    "signal_raw, sr_original = librosa.load(sample_wav, sr=SR)\n",
    "\n",
    "print(f\"📊 Thông tin signal:\")\n",
    "print(f\"   - Duration: {len(signal_raw)/SR:.2f}s\")\n",
    "print(f\"   - Samples: {len(signal_raw):,}\")\n",
    "print(f\"   - Sampling rate: {SR} Hz\")\n",
    "print(f\"   - Min: {signal_raw.min():.4f}\")\n",
    "print(f\"   - Max: {signal_raw.max():.4f}\")\n",
    "print(f\"   - Mean: {signal_raw.mean():.4f}\")\n",
    "print(f\"   - Std: {signal_raw.std():.4f}\")"
]))

# Visualize raw
notebook["cells"].append(create_cell("markdown", ["### 5.1. Visualize Raw Signal"]))
notebook["cells"].append(create_cell("code", [
    "fig, axes = plt.subplots(2, 1, figsize=(15, 8))\n",
    "\n",
    "# Time domain\n",
    "time = np.arange(len(signal_raw)) / SR\n",
    "axes[0].plot(time, signal_raw, linewidth=0.5)\n",
    "axes[0].set_title('Raw Signal - Time Domain', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Time (s)')\n",
    "axes[0].set_ylabel('Amplitude')\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Frequency domain\n",
    "freqs, psd = scipy_signal.welch(signal_raw, SR, nperseg=1024)\n",
    "axes[1].semilogy(freqs, psd)\n",
    "axes[1].set_title('Raw Signal - Frequency Domain (PSD)', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Frequency (Hz)')\n",
    "axes[1].set_ylabel('Power Spectral Density')\n",
    "axes[1].set_xlim([0, 1000])\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "axes[1].axvline(CUTOFF_FREQ, color='r', linestyle='--', label=f'Cutoff={CUTOFF_FREQ}Hz')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"✅ Đã visualize raw signal!\")"
]))

# Apply filter
notebook["cells"].append(create_cell("markdown", ["## 6. Apply Butterworth Low-Pass Filter"]))
notebook["cells"].append(create_cell("code", [
    "# Apply Butterworth filter\n",
    "signal_filtered = preprocessor.butter_lowpass_filter(signal_raw, order=5)\n",
    "\n",
    "print(f\"🔊 Đã apply Butterworth filter (cutoff={CUTOFF_FREQ}Hz)\")\n",
    "print(f\"\\n📊 So sánh:\")\n",
    "print(f\"   Raw    - Min: {signal_raw.min():.4f}, Max: {signal_raw.max():.4f}\")\n",
    "print(f\"   Filtered - Min: {signal_filtered.min():.4f}, Max: {signal_filtered.max():.4f}\")"
]))

# Visualize filtered
notebook["cells"].append(create_cell("markdown", ["### 6.1. So Sánh Raw vs Filtered"]))
notebook["cells"].append(create_cell("code", [
    "fig, axes = plt.subplots(3, 1, figsize=(15, 10))\n",
    "\n",
    "# Raw\n",
    "axes[0].plot(time, signal_raw, linewidth=0.5, color='blue', alpha=0.7)\n",
    "axes[0].set_title('Raw Signal', fontsize=12, fontweight='bold')\n",
    "axes[0].set_ylabel('Amplitude')\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Filtered\n",
    "axes[1].plot(time, signal_filtered, linewidth=0.5, color='green', alpha=0.7)\n",
    "axes[1].set_title('Filtered Signal (Butterworth Low-Pass)', fontsize=12, fontweight='bold')\n",
    "axes[1].set_ylabel('Amplitude')\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "# Overlay\n",
    "axes[2].plot(time, signal_raw, linewidth=0.5, color='blue', alpha=0.5, label='Raw')\n",
    "axes[2].plot(time, signal_filtered, linewidth=0.5, color='green', alpha=0.7, label='Filtered')\n",
    "axes[2].set_title('Overlay Comparison', fontsize=12, fontweight='bold')\n",
    "axes[2].set_xlabel('Time (s)')\n",
    "axes[2].set_ylabel('Amplitude')\n",
    "axes[2].legend()\n",
    "axes[2].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"✅ Filter đã loại bỏ high-frequency noise!\")"
]))

# Extract cycles
notebook["cells"].append(create_cell("markdown", ["## 7. Extract Cardiac Cycles từ TSV"]))
notebook["cells"].append(create_cell("code", [
    "# Extract cycles\n",
    "cycles = preprocessor.extract_cycles(sample_wav, sample_tsv)\n",
    "\n",
    "print(f\"💓 Đã extract {len(cycles)} cardiac cycles\")\n",
    "\n",
    "if len(cycles) > 0:\n",
    "    cycle_lengths = [len(c) for c in cycles]\n",
    "    print(f\"\\n📊 Thống kê độ dài cycles:\")\n",
    "    print(f\"   - Min: {min(cycle_lengths)} samples ({min(cycle_lengths)/SR:.3f}s)\")\n",
    "    print(f\"   - Max: {max(cycle_lengths)} samples ({max(cycle_lengths)/SR:.3f}s)\")\n",
    "    print(f\"   - Mean: {np.mean(cycle_lengths):.0f} samples ({np.mean(cycle_lengths)/SR:.3f}s)\")\n",
    "    print(f\"   - Median: {np.median(cycle_lengths):.0f} samples ({np.median(cycle_lengths)/SR:.3f}s)\")\n",
    "else:\n",
    "    print(\"❌ Không extract được cycles!\")"
]))

# Visualize cycles
notebook["cells"].append(create_cell("markdown", ["### 7.1. Visualize Extracted Cycles"]))
notebook["cells"].append(create_cell("code", [
    "if len(cycles) > 0:\n",
    "    # Hiển thị 5 cycles đầu tiên\n",
    "    n_show = min(5, len(cycles))\n",
    "    \n",
    "    fig, axes = plt.subplots(n_show, 1, figsize=(15, 3*n_show))\n",
    "    if n_show == 1:\n",
    "        axes = [axes]\n",
    "    \n",
    "    for i in range(n_show):\n",
    "        cycle = cycles[i]\n",
    "        time_cycle = np.arange(len(cycle)) / SR\n",
    "        \n",
    "        axes[i].plot(time_cycle, cycle, linewidth=1)\n",
    "        axes[i].set_title(f'Cycle {i+1} - Length: {len(cycle)} samples ({len(cycle)/SR:.3f}s)', \n",
    "                         fontsize=12, fontweight='bold')\n",
    "        axes[i].set_xlabel('Time (s)')\n",
    "        axes[i].set_ylabel('Amplitude')\n",
    "        axes[i].grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(f\"✅ Đã visualize {n_show} cycles đầu tiên!\")\n",
    "else:\n",
    "    print(\"❌ Không có cycles để visualize!\")"
]))

# Normalize
notebook["cells"].append(create_cell("markdown", ["## 8. Normalize Signal"]))
notebook["cells"].append(create_cell("code", [
    "if len(cycles) > 0:\n",
    "    # Lấy 1 cycle để demo\n",
    "    cycle_raw = cycles[0]\n",
    "    \n",
    "    # Normalize\n",
    "    cycle_normalized = preprocessor.normalize_signal(cycle_raw)\n",
    "    \n",
    "    print(f\"📊 So sánh trước/sau normalize:\")\n",
    "    print(f\"   Raw        - Mean: {cycle_raw.mean():.4f}, Std: {cycle_raw.std():.4f}\")\n",
    "    print(f\"   Normalized - Mean: {cycle_normalized.mean():.4f}, Std: {cycle_normalized.std():.4f}\")\n",
    "    \n",
    "    # Visualize\n",
    "    fig, axes = plt.subplots(2, 1, figsize=(15, 8))\n",
    "    \n",
    "    time_cycle = np.arange(len(cycle_raw)) / SR\n",
    "    \n",
    "    axes[0].plot(time_cycle, cycle_raw, linewidth=1, color='blue')\n",
    "    axes[0].set_title('Before Normalization', fontsize=12, fontweight='bold')\n",
    "    axes[0].set_ylabel('Amplitude')\n",
    "    axes[0].grid(True, alpha=0.3)\n",
    "    \n",
    "    axes[1].plot(time_cycle, cycle_normalized, linewidth=1, color='green')\n",
    "    axes[1].set_title('After Normalization (Zero Mean, Unit Variance)', fontsize=12, fontweight='bold')\n",
    "    axes[1].set_xlabel('Time (s)')\n",
    "    axes[1].set_ylabel('Amplitude')\n",
    "    axes[1].grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"✅ Normalization đưa signal về mean=0, std=1!\")\n",
    "else:\n",
    "    print(\"❌ Không có cycles để normalize!\")"
]))

# Scattering features
notebook["cells"].append(create_cell("markdown", ["## 9. Extract Scattering Features"]))
notebook["cells"].append(create_cell("code", [
    "if len(cycles) > 0:\n",
    "    # Extract scattering features từ cycle đầu tiên\n",
    "    cycle = cycles[0]\n",
    "    features = preprocessor.extract_scattering_features(cycle)\n",
    "    \n",
    "    print(f\"🌊 Scattering Features:\")\n",
    "    print(f\"   - Input shape: {cycle.shape}\")\n",
    "    print(f\"   - Output shape: {features.shape}\")\n",
    "    print(f\"   - Feature dimension: {features.shape[0] * features.shape[1]}\")\n",
    "    print(f\"   - Min: {features.min():.4f}\")\n",
    "    print(f\"   - Max: {features.max():.4f}\")\n",
    "    print(f\"   - Mean: {features.mean():.4f}\")\n",
    "else:\n",
    "    print(\"❌ Không có cycles để extract features!\")"
]))

# Visualize scattering
notebook["cells"].append(create_cell("markdown", ["### 9.1. Visualize Scattering Features"]))
notebook["cells"].append(create_cell("code", [
    "if len(cycles) > 0:\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "    \n",
    "    # Heatmap\n",
    "    im = axes[0].imshow(features, aspect='auto', cmap='viridis', interpolation='nearest')\n",
    "    axes[0].set_title('Scattering Features Heatmap', fontsize=12, fontweight='bold')\n",
    "    axes[0].set_xlabel('Time Frames')\n",
    "    axes[0].set_ylabel('Scattering Coefficients')\n",
    "    plt.colorbar(im, ax=axes[0])\n",
    "    \n",
    "    # Flatten và histogram\n",
    "    features_flat = features.flatten()\n",
    "    axes[1].hist(features_flat, bins=50, edgecolor='black', alpha=0.7)\n",
    "    axes[1].set_title('Distribution of Scattering Coefficients', fontsize=12, fontweight='bold')\n",
    "    axes[1].set_xlabel('Coefficient Value')\n",
    "    axes[1].set_ylabel('Frequency')\n",
    "    axes[1].grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"✅ Scattering features capture multi-scale time-frequency information!\")\n",
    "else:\n",
    "    print(\"❌ Không có features để visualize!\")"
]))

# Summary
notebook["cells"].append(create_cell("markdown", [
    "## 10. Tóm Tắt Pipeline\n",
    "\n",
    "### Workflow đã demo:\n",
    "```\n",
    "Raw WAV File (4000 Hz)\n",
    "    ↓\n",
    "Butterworth Low-Pass Filter (500 Hz cutoff)\n",
    "    ↓\n",
    "Extract Cardiac Cycles (TSV annotations: 1→2→3→4)\n",
    "    ↓\n",
    "Normalize Each Cycle (Zero mean, Unit variance)\n",
    "    ↓\n",
    "Pad/Crop to MAX_LEN (3000 samples)\n",
    "    ↓\n",
    "Scattering Transform (J=6, Q=8)\n",
    "    ↓\n",
    "Features Ready for Training!\n",
    "```\n",
    "\n",
    "### Key Points:\n",
    "- ✅ **Butterworth filter** loại bỏ high-frequency noise\n",
    "- ✅ **Cycle extraction** tận dụng TSV annotations\n",
    "- ✅ **Normalization** standardize amplitude\n",
    "- ✅ **Scattering** extract multi-scale features\n",
    "- ✅ **Fixed length** (3000 samples) cho CNN/LSTM\n",
    "\n",
    "### Sử dụng trong training:\n",
    "- `04_training_cnn_method.ipynb` - 1D CNN\n",
    "- `05_training_lstm_method.ipynb` - Bidirectional LSTM\n",
    "\n",
    "---\n",
    "\n",
    "**🎉 Demo hoàn tất!**"
]))

# Save
output_file = 'notebooks/demo_data_preprocessing.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Đã tạo notebook: {output_file}")
print(f"📊 Tổng số cells: {len(notebook['cells'])}")
print(f"\n🚀 Để chạy notebook:")
print(f"   jupyter notebook {output_file}")
