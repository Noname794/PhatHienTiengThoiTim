#!/usr/bin/env python3
"""
Script để tạo demo notebook cho data_preprocessing.py
"""

import json

# Tạo notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: Title
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Demo: HeartSoundPreprocessor - Xem Cách Hoạt Động\n",
        "\n",
        "Notebook này demo các functions trong `data_preprocessing.py` để hiểu cách chúng hoạt động.\n",
        "\n",
        "## Nội dung:\n",
        "1. Khởi tạo HeartSoundPreprocessor\n",
        "2. Load và visualize 1 file WAV\n",
        "3. Apply Butterworth filter\n",
        "4. Extract cardiac cycles từ TSV\n",
        "5. Normalize cycles\n",
        "6. Extract Scattering features\n",
        "7. Visualize toàn bộ pipeline"
    ]
})

# Cell 2: Imports
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import sys\n",
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
        "%matplotlib inline\n",
        "\n",
        "print(\"✅ Đã import thành công!\")"
    ]
})

# Cell 3: Config
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Cấu hình tham số"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Tham số (giống như trong 04_training_cnn_method và 05_training_lstm_method)\n",
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
        "print(f\"   - MAX_LEN = {MAX_LEN} samples\")\n",
        "print(f\"   - CUTOFF_FREQ = {CUTOFF_FREQ} Hz\")\n",
        "print(f\"   - Scattering J = {J}\")"
    ]
})

# Cell 4: Initialize Preprocessor
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Khởi tạo HeartSoundPreprocessor"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
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
    ]
})

# Save notebook
with open('notebooks/demo_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Đã tạo phần 1 của notebook!")
print("Chạy script tiếp theo để thêm các cells còn lại...")
