# Heart Sound Classifier - Web Application

Ứng dụng web phân loại âm thanh tim sử dụng Deep Learning (1D-CNN).

##  Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Node.js**: 14.0 trở lên
- **npm**: 6.0 trở lên
- **Git**: Để clone repository

##  Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd VIP
```

### Bước 2: Tải dữ liệu và model (BẮT BUỘC)

 **Quan trọng**: Bạn cần tải data và model từ Google Drive trước khi chạy ứng dụng.

1. Truy cập link Google Drive: https://drive.google.com/drive/folders/17Q5LUaqsQkGMusgG__-sJJ4UBAX3c-4f?usp=sharing

2. Tải về 2 thư mục:
   - `data/` - Chứa dữ liệu âm thanh tim
   - `results/` - Chứa model đã train (file .h5)

3. Giải nén và đặt vào thư mục gốc của project:
   ```
   VIP/
   ├── data/           # Thư mục data vừa tải
   ├── results/        # Thư mục results vừa tải
   ├── webapp/
   └── ...
   ```

4. Kiểm tra file model tồn tại tại: `results/models/1dcnn_method_best.h5`

### Bước 3: Cài đặt tự động (Windows)

Chạy script setup tự động:

```bash
cd webapp
setup.bat
```

Script sẽ tự động:
- Tạo môi trường ảo Python
- Cài đặt các thư viện Python cần thiết
- Cài đặt các dependencies cho React frontend
- Kiểm tra file model

### Bước 4: Cài đặt thủ công (nếu cần)

#### Backend (Python/Flask)

```bash
cd webapp/backend

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Frontend (React)

```bash
cd webapp/frontend

# Cài đặt dependencies
npm install
```

## Chạy ứng dụng

### Cách 1: Chạy tự động (Windows)

```bash
cd webapp
start.bat
```

Script sẽ tự động mở 2 cửa sổ terminal:
- Backend server: http://localhost:5000
- Frontend server: http://localhost:3000

### Cách 2: Chạy thủ công

#### Terminal 1 - Backend:

```bash
cd webapp/backend
python app.py
```

Backend sẽ chạy tại: http://localhost:5000

#### Terminal 2 - Frontend:

```bash
cd webapp/frontend
npm start
```

Frontend sẽ tự động mở trình duyệt tại: http://localhost:3000

##  Cấu trúc thư mục

```
VIP/
├── webapp/
│   ├── backend/          # Flask API server
│   │   ├── app.py       # Main application
│   │   ├── requirements.txt
│   │   └── uploads/     # Thư mục lưu file upload
│   ├── frontend/        # React application
│   │   ├── src/
│   │   ├── public/
│   │   └── package.json
│   ├── setup.bat        # Script cài đặt (Windows)
│   ├── setup.sh         # Script cài đặt (Linux/Mac)
│   ├── start.bat        # Script khởi chạy (Windows)
│   └── start.sh         # Script khởi chạy (Linux/Mac)
├── src/                 # Source code training model
├── notebooks/           # Jupyter notebooks
└── requirements_cnn.txt # Dependencies cho training
```

##  Sử dụng ứng dụng

1. Mở trình duyệt tại http://localhost:3000
2. Upload file âm thanh tim (.wav)
3. Nhấn nút "Phân tích"
4. Xem kết quả phân loại và độ tin cậy

##  Xử lý sự cố

### Backend không khởi động

- Kiểm tra Python đã cài đặt: `python --version`
- Kiểm tra môi trường ảo đã được kích hoạt
- Kiểm tra tất cả dependencies đã được cài: `pip list`

### Frontend không khởi động

- Kiểm tra Node.js đã cài đặt: `node --version`
- Xóa thư mục `node_modules` và chạy lại `npm install`
- Kiểm tra port 3000 có bị chiếm dụng không

### Model không tìm thấy

- Đảm bảo đã tải thư mục `results/` từ Google Drive
- Kiểm tra file model tồn tại tại: `results/models/1dcnn_method_best.h5`
- Link tải: https://drive.google.com/drive/folders/17Q5LUaqsQkGMusgG__-sJJ4UBAX3c-4f?usp=sharing

### Thiếu dữ liệu

- Đảm bảo đã tải thư mục `data/` từ Google Drive
- Kiểm tra cấu trúc thư mục data đúng như hướng dẫn

## Ghi chú

- Backend mặc định chạy trên port 5000
- Frontend mặc định chạy trên port 3000
- File upload được lưu tạm trong `webapp/backend/uploads/`
- Hỗ trợ định dạng file: .wav

##  Development

### Training model mới

```bash
# Cài đặt dependencies cho training
pip install -r requirements_cnn.txt

# Chạy training script
python src/train.py
```

### Chạy tests

```bash
# Backend tests
cd webapp/backend
python test_api.py

# Frontend tests
cd webapp/frontend
npm test
```
