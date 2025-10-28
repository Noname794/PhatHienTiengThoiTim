# Data Preprocessing và Deep Learning Models

## Tổng quan

Project này bao gồm:
1. **Module tiền xử lý dữ liệu** (`data_preprocessing.py`)
2. **CNN Model** (file gốc: `04_training_cnn_method1 - Copy.ipynb`)
3. **LSTM/GRU Model** (file mới: `05_training_lstm_method.ipynb`)

---

## 1. Module Tiền Xử Lý (`src/data_preprocessing.py`)

### Mục đích
Tách riêng các bước tiền xử lý dữ liệu ra một module độc lập để:
- Tái sử dụng cho nhiều models khác nhau
- Dễ bảo trì và cập nhật
- Code sạch hơn, tránh duplicate

### Class: `HeartSoundPreprocessor`

#### Khởi tạo
```python
from data_preprocessing import HeartSoundPreprocessor

preprocessor = HeartSoundPreprocessor(
    sr=4000,           # Sampling rate (Hz)
    max_len=3000,      # Độ dài cycle (samples)
    cutoff_freq=500,   # Butterworth filter cutoff (Hz)
    scattering_j=6     # Scattering Transform depth
)
```

#### Các phương thức chính

##### 1. `process_dataset()`
Xử lý toàn bộ dataset từ raw files.

```python
X, y = preprocessor.process_dataset(
    raw_data_dir='../data/raw/training_data/',
    metadata_file='../data/raw/training_data.csv',
    use_scattering=True,  # True: Scattering features, False: raw signal
    verbose=True
)
```

**Output:**
- `X`: Features array (Scattering hoặc raw signal)
- `y`: Labels array (0=Absent, 1=Present)

##### 2. `balance_dataset()`
Balance dataset bằng downsampling.

```python
X_balanced, y_balanced = preprocessor.balance_dataset(X, y, random_state=42)
```

##### 3. `save_processed_data()` / `load_processed_data()`
Lưu và load dữ liệu đã xử lý.

```python
# Lưu
preprocessor.save_processed_data(X, y, output_dir='../data/processed/', prefix='balanced')

# Load
X, y = preprocessor.load_processed_data(output_dir='../data/processed/', prefix='balanced')
```

##### 4. Các helper methods
- `butter_lowpass_filter()`: Butterworth low-pass filter
- `normalize_signal()`: Normalize signal (zero mean, unit variance)
- `extract_cycles()`: Trích xuất cycles từ WAV + TSV
- `extract_scattering_features()`: Trích xuất Scattering features
- `load_labels()`: Load labels từ metadata CSV

---

## 2. CNN Model (File gốc)

### File: `04_training_cnn_method1 - Copy.ipynb`

### Architecture
```
Input (26, 47)
    ↓
Conv1D(128) + LeakyReLU + MaxPooling + BatchNorm
    ↓
Conv1D(64) + LeakyReLU + MaxPooling + BatchNorm
    ↓
Flatten
    ↓
Dense(128) + Dropout(0.3) + LeakyReLU
    ↓
Dense(64) + Dropout(0.3) + LeakyReLU
    ↓
Dense(32) + ReLU
    ↓
Dense(2, softmax)
```

### Kết quả
- **Cross Validation Mean Accuracy**: ~80.69%
- **Test Accuracy**: ~83.46%
- **Precision (Present)**: 88%
- **Recall (Present)**: 78%

### Đặc điểm
- ✅ Sử dụng Scattering Transform features
- ✅ 2 Conv blocks đơn giản
- ✅ LeakyReLU activation
- ✅ 5-fold Cross Validation

---

## 3. LSTM/GRU Model (File mới)

### File: `05_training_lstm_method.ipynb`

### Architecture Options

#### Option 1: Bidirectional LSTM
```
Input (26, 47)
    ↓
Bidirectional LSTM(128) + BatchNorm + Dropout(0.3)
    ↓
Bidirectional LSTM(64) + BatchNorm + Dropout(0.3)
    ↓
GlobalAveragePooling1D
    ↓
Dense(128) + Dropout(0.4) + BatchNorm
    ↓
Dense(64) + Dropout(0.3)
    ↓
Dense(2, softmax)
```

#### Option 2: Bidirectional GRU
```
Input (26, 47)
    ↓
Bidirectional GRU(128) + BatchNorm + Dropout(0.3)
    ↓
Bidirectional GRU(64) + BatchNorm + Dropout(0.3)
    ↓
GlobalAveragePooling1D
    ↓
Dense(128) + Dropout(0.4) + BatchNorm
    ↓
Dense(64) + Dropout(0.3)
    ↓
Dense(2, softmax)
```

### Đặc điểm
- ✅ **Bidirectional**: Học features từ cả 2 hướng (forward + backward)
- ✅ **Global Average Pooling**: Thay vì Flatten
- ✅ **ReduceLROnPlateau**: Tự động giảm learning rate
- ✅ Có thể chọn LSTM hoặc GRU (GRU nhanh hơn)
- ✅ Sử dụng preprocessing module

### Ưu điểm so với CNN
- **LSTM/GRU** tốt hơn cho sequential data (time series)
- Có khả năng học **long-term dependencies**
- **Bidirectional** giúp context từ cả quá khứ và tương lai

---

## 4. So sánh Models

| Model | Architecture | Params | Speed | Accuracy (dự kiến) |
|-------|-------------|--------|-------|-------------------|
| **CNN** | 2 Conv blocks | ~Medium | Fast | ~83-84% |
| **LSTM** | 2 Bi-LSTM layers | ~High | Slower | ~84-86% |
| **GRU** | 2 Bi-GRU layers | ~Medium | Medium | ~83-85% |

### Khi nào dùng model nào?

#### CNN
- ✅ Cần inference nhanh
- ✅ Dữ liệu có local patterns rõ ràng
- ✅ Tài nguyên hạn chế

#### LSTM
- ✅ Dữ liệu sequential phức tạp
- ✅ Cần học long-term dependencies
- ✅ Có đủ tài nguyên training

#### GRU
- ✅ Cân bằng giữa LSTM và CNN
- ✅ Training nhanh hơn LSTM
- ✅ Performance tương đương LSTM

---

## 5. Workflow Sử Dụng

### Bước 1: Preprocessing (chỉ chạy 1 lần)
```python
from data_preprocessing import HeartSoundPreprocessor

# Khởi tạo
preprocessor = HeartSoundPreprocessor(sr=4000, max_len=3000)

# Xử lý data
X, y = preprocessor.process_dataset(raw_data_dir, metadata_file)

# Balance
X_balanced, y_balanced = preprocessor.balance_dataset(X, y)

# Lưu
preprocessor.save_processed_data(X_balanced, y_balanced, output_dir)
```

### Bước 2: Train Model
```python
# Load data
X, y = preprocessor.load_processed_data(output_dir)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model (CNN hoặc LSTM)
model = build_lstm_attention_model(X_train.shape[1:])

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), ...)
```

---

## 6. Files Structure

```
VIP/
├── src/
│   └── data_preprocessing.py          # Module tiền xử lý
├── notebooks/
│   ├── 04_training_cnn_method1 - Copy.ipynb   # CNN model (gốc)
│   ├── 05_training_lstm_method.ipynb          # LSTM/GRU model (mới)
│   └── example_using_preprocessor.ipynb       # Example usage
├── data/
│   ├── raw/
│   │   └── training_data/             # Raw WAV + TSV files
│   ├── processed_cnn_method/          # Processed data cho CNN
│   └── processed_lstm_method/         # Processed data cho LSTM
└── results/
    └── models/
        ├── 1dcnn_method_best.h5       # CNN model
        └── lstm_model_best.h5         # LSTM model
```

---

## 7. Next Steps

### Cải tiến có thể thực hiện:

1. **Ensemble Model**: Kết hợp CNN + LSTM
2. **Attention Mechanism**: Thêm attention layer cho LSTM
3. **Data Augmentation**: Tăng cường dữ liệu
4. **Hyperparameter Tuning**: Tối ưu tham số
5. **Transfer Learning**: Sử dụng pretrained models
6. **Multi-task Learning**: Predict cả Murmur và Outcome

---

## 8. Requirements

```bash
# Core
numpy
pandas
librosa
scipy

# Deep Learning
tensorflow>=2.0
keras

# Scattering Transform
kymatio

# Visualization
matplotlib
seaborn

# Utils
scikit-learn
tqdm
```

---

## 9. Troubleshooting

### Lỗi: "Module not found: data_preprocessing"
```python
import sys
sys.path.append('../src')
from data_preprocessing import HeartSoundPreprocessor
```

### Lỗi: "Out of memory"
- Giảm `BATCH_SIZE`
- Giảm số units trong LSTM/GRU
- Sử dụng GRU thay vì LSTM

### Model không converge
- Tăng `EARLY_STOPPING_PATIENCE`
- Thử learning rate khác
- Check data balance

---

## 10. Contact & Support

Nếu có vấn đề hoặc câu hỏi, vui lòng:
1. Check documentation này
2. Xem example notebook
3. Review code comments trong module

**Happy Coding! 🚀**
