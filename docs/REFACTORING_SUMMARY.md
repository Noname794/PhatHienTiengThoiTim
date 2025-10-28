# Refactoring Summary - CNN Notebook

## ✅ Đã Hoàn Thành

File `04_training_cnn_method1.ipynb` đã được refactor để sử dụng `data_preprocessing.py` module.

---

## 📝 Những Thay Đổi

### 1. **Cell 1 - Import**
**Trước:**
```python
import librosa
from tqdm.notebook import tqdm
from scipy.signal import butter, filtfilt
from kymatio.numpy import Scattering1D
# ... nhiều imports khác
```

**Sau:**
```python
# Import preprocessing module
sys.path.append('../src')
from data_preprocessing import HeartSoundPreprocessor
# ... chỉ giữ lại imports cần thiết cho model
```

**Lợi ích:** Giảm dependencies, code sạch hơn

---

### 2. **Cell 4-5 - Loại bỏ Helper Functions**
**Trước:**
```python
def butter_lowpass_filter(data, cutoff, sr, order=5):
    # ... 30+ lines code
    
def normalize_signal(sig):
    # ... code
    
def extract_cycles(wav_path, tsv_path):
    # ... 40+ lines code
    
def extract_scattering_features(signal, scattering):
    # ... code
```

**Sau:**
```python
# Khởi tạo preprocessor
preprocessor = HeartSoundPreprocessor(
    sr=SR,
    max_len=MAX_LEN,
    cutoff_freq=CUTOFF_FREQ,
    scattering_j=J
)
```

**Lợi ích:** Loại bỏ ~100 lines duplicate code

---

### 3. **Cell 8-9 - Load Labels & Process Data**
**Trước:**
```python
# Đọc metadata
df_info = pd.read_csv(METADATA_FILE)
df_info.columns = [col.replace(' ', '_') for col in df_info.columns]

label_dict = {}
for _, row in df_info.iterrows():
    # ... 10+ lines code
    
# Khởi tạo Scattering Transform
scattering = Scattering1D(J=J, shape=MAX_LEN)

X = []
y = []

for fname in tqdm(wav_files, desc="Processing"):
    # ... 30+ lines code
    cycles = extract_cycles(wav_path, tsv_path)
    for cycle in cycles:
        features = extract_scattering_features(cycle, scattering)
        X.append(features)
        y.append(label)
```

**Sau:**
```python
# Xử lý dataset bằng preprocessing module
X, y = preprocessor.process_dataset(
    raw_data_dir=RAW_DATA_DIR,
    metadata_file=METADATA_FILE,
    use_scattering=True,
    verbose=True
)
```

**Lợi ích:** Từ ~50 lines → 6 lines

---

### 4. **Cell 12-13 - Balance Dataset**
**Trước:**
```python
# Filter indices by label
idx_present = np.where(y == 1)[0]
idx_absent = np.where(y == 0)[0]

n_present = len(idx_present)

np.random.seed(42)
idx_absent_reduced = np.random.choice(idx_absent, size=n_present, replace=False)

idx_final = np.concatenate([idx_present, idx_absent_reduced])
idx_final = shuffle(idx_final, random_state=42)

X_balanced = X[idx_final]
y_balanced = y[idx_final]
```

**Sau:**
```python
# Balance dataset bằng preprocessing module
X_balanced, y_balanced = preprocessor.balance_dataset(X, y, random_state=42)
```

**Lợi ích:** Từ ~12 lines → 1 line

---

### 5. **Bonus - Save/Load Data (Optional)**
**Thêm mới:**
```python
# (Optional) Lưu dữ liệu đã xử lý
preprocessor.save_processed_data(X_balanced, y_balanced, PROCESSED_DATA_DIR)

# (Optional) Load dữ liệu đã xử lý
X_balanced, y_balanced = preprocessor.load_processed_data(PROCESSED_DATA_DIR)
```

**Lợi ích:** Có thể skip preprocessing khi chạy lại

---

## 📊 Kết Quả

### Code Reduction
| Metric | Trước | Sau | Giảm |
|--------|-------|-----|------|
| **Total cells** | 24 | 22 | -2 |
| **Preprocessing code** | ~150 lines | ~10 lines | **-93%** |
| **Dependencies** | 15+ imports | 8 imports | -47% |

### Maintainability
- ✅ **DRY Principle**: Không duplicate code giữa CNN và LSTM notebooks
- ✅ **Single Source of Truth**: Preprocessing logic chỉ ở 1 nơi
- ✅ **Easier Updates**: Sửa preprocessing → tự động apply cho cả 2 models
- ✅ **Reusability**: Dễ dàng tạo models mới

---

## 🎯 Cấu Trúc Mới

```
VIP/
├── src/
│   └── data_preprocessing.py          # ⭐ Single source of truth
│
├── notebooks/
│   ├── 04_training_cnn_method1.ipynb  # ✅ Refactored - uses module
│   └── 05_training_lstm_method.ipynb  # ✅ Already uses module
│
└── data/
    ├── processed_cnn_method/          # Có thể save/load
    └── processed_lstm_method/
```

---

## 🚀 Workflow Mới

### Lần đầu chạy:
```python
# 1. Khởi tạo preprocessor
preprocessor = HeartSoundPreprocessor(sr=4000, max_len=3000, ...)

# 2. Process data
X, y = preprocessor.process_dataset(raw_data_dir, metadata_file)

# 3. Balance
X_balanced, y_balanced = preprocessor.balance_dataset(X, y)

# 4. (Optional) Save để tái sử dụng
preprocessor.save_processed_data(X_balanced, y_balanced, output_dir)

# 5. Train model
# ...
```

### Lần sau chạy (nhanh hơn):
```python
# 1. Khởi tạo preprocessor
preprocessor = HeartSoundPreprocessor(sr=4000, max_len=3000, ...)

# 2. Load data đã xử lý
X_balanced, y_balanced = preprocessor.load_processed_data(output_dir)

# 3. Train model ngay
# ...
```

---

## ✨ Lợi Ích Chính

### 1. **Code Quality**
- Giảm 93% preprocessing code trong notebook
- Dễ đọc, dễ hiểu hơn
- Follow best practices (separation of concerns)

### 2. **Maintainability**
- Sửa bug ở 1 nơi → apply cho tất cả
- Dễ thêm features mới
- Dễ test preprocessing logic riêng

### 3. **Reusability**
- Dùng cho CNN, LSTM, GRU, ...
- Dùng cho các experiments khác
- Dùng cho production code

### 4. **Performance**
- Có thể save/load processed data
- Không cần reprocess mỗi lần chạy
- Tiết kiệm thời gian development

---

## 🔄 Migration Guide

Nếu bạn có notebooks khác muốn refactor:

### Bước 1: Import module
```python
import sys
sys.path.append('../src')
from data_preprocessing import HeartSoundPreprocessor
```

### Bước 2: Thay thế preprocessing code
```python
# Thay vì:
# - butter_lowpass_filter()
# - normalize_signal()
# - extract_cycles()
# - extract_scattering_features()
# - load_labels()
# - balance dataset code

# Dùng:
preprocessor = HeartSoundPreprocessor(...)
X, y = preprocessor.process_dataset(...)
X_balanced, y_balanced = preprocessor.balance_dataset(X, y)
```

### Bước 3: Giữ nguyên model code
```python
# Model architecture và training code không đổi
model = build_model(...)
model.fit(X_train, y_train, ...)
```

---

## 📚 Files Liên Quan

1. **`src/data_preprocessing.py`** - Preprocessing module
2. **`notebooks/04_training_cnn_method1.ipynb`** - CNN notebook (refactored)
3. **`notebooks/05_training_lstm_method.ipynb`** - LSTM notebook (already clean)
4. **`docs/PREPROCESSING_AND_MODELS.md`** - Full documentation

---

## ✅ Checklist

- [x] Refactor CNN notebook
- [x] Test preprocessing module
- [x] Add save/load functionality
- [x] Update documentation
- [x] Verify both notebooks work with module

---

## 🎉 Kết Luận

File `04_training_cnn_method1.ipynb` giờ đây:
- ✅ **Sạch hơn**: Giảm 93% preprocessing code
- ✅ **Dễ maintain**: Single source of truth
- ✅ **Consistent**: Cùng preprocessing với LSTM notebook
- ✅ **Professional**: Follow software engineering best practices

**Bạn có thể XÓA file `example_using_preprocessor.ipynb`** vì cả 2 notebooks chính đã sử dụng module rồi! 🗑️
