"""
Module tiền xử lý dữ liệu cho heart sound classification
Bao gồm: cycle extraction, filtering, normalization, feature extraction
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt
from kymatio.numpy import Scattering1D
from tqdm import tqdm
from sklearn.utils import shuffle


class HeartSoundPreprocessor:
    """
    Class xử lý dữ liệu heart sound
    """
    
    def __init__(self, sr=4000, max_len=3000, cutoff_freq=500, scattering_j=6):
        """
        Args:
            sr: Sampling rate (Hz)
            max_len: Độ dài tối đa của cycle (samples)
            cutoff_freq: Tần số cắt cho Butterworth filter (Hz)
            scattering_j: Depth cho Scattering Transform
        """
        self.sr = sr
        self.max_len = max_len
        self.cutoff_freq = cutoff_freq
        self.scattering_j = scattering_j
        
        # Khởi tạo Scattering Transform
        self.scattering = Scattering1D(J=scattering_j, shape=max_len)
        
    def butter_lowpass_filter(self, data, order=5):
        """Butterworth low-pass filter"""
        nyquist = 0.5 * self.sr
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def normalize_signal(self, sig):
        """Normalize signal (zero mean, unit variance)"""
        sig = sig - np.mean(sig)
        std = np.std(sig)
        return sig / std if std > 0 else sig
    
    def extract_cycles(self, wav_path, tsv_path):
        """
        Trích xuất các chu kỳ tim (1→2→3→4) từ file WAV và TSV
        
        Args:
            wav_path: Đường dẫn file WAV
            tsv_path: Đường dẫn file TSV (segmentation)
            
        Returns:
            List các cycles đã được normalize
        """
        try:
            # Load signal
            signal, _ = librosa.load(wav_path, sr=self.sr)
            
            # Apply Butterworth low-pass filter
            signal = self.butter_lowpass_filter(signal, order=5)
            
            # Load segmentation
            df_seg = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'label'])
            
            cycles = []
            for i in range(len(df_seg) - 3):
                labels_seq = df_seg.iloc[i:i+4]['label'].tolist()
                if labels_seq == [1, 2, 3, 4]:
                    start = int(df_seg.iloc[i]['start'] * self.sr)
                    end = int(df_seg.iloc[i+3]['end'] * self.sr)
                    cycle = signal[start:end]
                    
                    # Lọc cycles quá ngắn
                    if len(cycle) > 100:
                        cycle = self.normalize_signal(cycle)
                        cycles.append(cycle)
            
            return cycles
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            return []
    
    def extract_scattering_features(self, signal):
        """
        Trích xuất Scattering features từ signal
        
        Args:
            signal: Input signal (1D array)
            
        Returns:
            Scattering features (2D array)
        """
        # Pad hoặc cắt về MAX_LEN
        if len(signal) < self.max_len:
            signal = np.pad(signal, (0, self.max_len - len(signal)))
        else:
            signal = signal[:self.max_len]
        
        features = self.scattering(signal)
        return features
    
    def load_labels(self, metadata_file):
        """
        Load labels từ metadata file
        
        Args:
            metadata_file: Đường dẫn file CSV metadata
            
        Returns:
            Dictionary {patient_id: label}
            - Absent=0, Present=1, Unknown bị loại bỏ
        """
        df_info = pd.read_csv(metadata_file)
        df_info.columns = [col.replace(' ', '_') for col in df_info.columns]
        
        label_dict = {}
        for _, row in df_info.iterrows():
            patient_id = str(row['Patient_ID'])
            murmur = str(row['Murmur']).strip().capitalize()
            
            if murmur == 'Absent':
                label_dict[patient_id] = 0
            elif murmur == 'Present':
                label_dict[patient_id] = 1
            # Unknown sẽ không được thêm vào label_dict (bỏ qua)
        
        return label_dict
    
    def process_dataset(self, raw_data_dir, metadata_file, use_scattering=True, verbose=True):
        """
        Xử lý toàn bộ dataset
        
        Args:
            raw_data_dir: Thư mục chứa files WAV và TSV
            metadata_file: File CSV chứa labels
            use_scattering: Có sử dụng Scattering Transform không
            verbose: Hiển thị progress bar
            
        Returns:
            X: Features array
            y: Labels array
        """
        # Load labels
        label_dict = self.load_labels(metadata_file)
        
        if verbose:
            print(f"✅ Đã tải labels cho {len(label_dict)} bệnh nhân")
            print(f"   - Absent (0): {sum(1 for v in label_dict.values() if v == 0)}")
            print(f"   - Present (1): {sum(1 for v in label_dict.values() if v == 1)}")
        
        X = []
        y = []
        
        # Lấy danh sách files
        wav_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.wav')]
        
        iterator = tqdm(wav_files, desc="Processing") if verbose else wav_files
        
        for fname in iterator:
            # Parse filename
            patient_id = fname.split('_')[0]
            valve = fname.split('_')[1].split('.')[0]
            
            tsv_name = f"{patient_id}_{valve}.tsv"
            wav_path = os.path.join(raw_data_dir, fname)
            tsv_path = os.path.join(raw_data_dir, tsv_name)
            
            # Check files tồn tại
            if not os.path.exists(tsv_path):
                continue
            if patient_id not in label_dict:
                continue
            
            label = label_dict[patient_id]
            
            # Extract cycles
            cycles = self.extract_cycles(wav_path, tsv_path)
            
            # Extract features từ mỗi cycle
            for cycle in cycles:
                if use_scattering:
                    features = self.extract_scattering_features(cycle)
                else:
                    # Pad/truncate raw signal
                    if len(cycle) < self.max_len:
                        features = np.pad(cycle, (0, self.max_len - len(cycle)))
                    else:
                        features = cycle[:self.max_len]
                
                X.append(features)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"\n✅ Hoàn thành!")
            print(f"   Total samples: {len(y)}")
            print(f"   - Absent (0): {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
            print(f"   - Present (1): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
            print(f"   Feature shape: {X.shape}")
        
        return X, y
    
    def balance_dataset(self, X, y, random_state=42):
        """
        Balance dataset bằng cách downsample class đa số
        
        Args:
            X: Features array
            y: Labels array
            random_state: Random seed
            
        Returns:
            X_balanced, y_balanced
        """
        # Filter indices by label
        idx_present = np.where(y == 1)[0]
        idx_absent = np.where(y == 0)[0]
        
        # Số samples trong Present class
        n_present = len(idx_present)
        
        # Downsample Absent class về bằng Present
        np.random.seed(random_state)
        idx_absent_reduced = np.random.choice(idx_absent, size=n_present, replace=False)
        
        # Kết hợp 2 classes
        idx_final = np.concatenate([idx_present, idx_absent_reduced])
        
        # Shuffle
        idx_final = shuffle(idx_final, random_state=random_state)
        
        # Extract balanced dataset
        X_balanced = X[idx_final]
        y_balanced = y[idx_final]
        
        print(f"Số samples sau balancing: {len(y_balanced)}")
        print(f"   - Absent (0): {np.sum(y_balanced == 0)}")
        print(f"   - Present (1): {np.sum(y_balanced == 1)}")
        
        return X_balanced, y_balanced
    
    def save_processed_data(self, X, y, output_dir, prefix='processed'):
        """
        Lưu dữ liệu đã xử lý
        
        Args:
            X: Features array
            y: Labels array
            output_dir: Thư mục output
            prefix: Prefix cho tên file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, f'{prefix}_X.npy'), X)
        np.save(os.path.join(output_dir, f'{prefix}_y.npy'), y)
        
        print(f"✅ Đã lưu dữ liệu vào {output_dir}")
    
    def load_processed_data(self, output_dir, prefix='processed'):
        """
        Load dữ liệu đã xử lý
        
        Args:
            output_dir: Thư mục chứa data
            prefix: Prefix của tên file
            
        Returns:
            X, y
        """
        X = np.load(os.path.join(output_dir, f'{prefix}_X.npy'))
        y = np.load(os.path.join(output_dir, f'{prefix}_y.npy'))
        
        print(f"✅ Đã load dữ liệu từ {output_dir}")
        print(f"   Shape: X={X.shape}, y={y.shape}")
        
        return X, y


# Example usage
if __name__ == "__main__":
    # Khởi tạo preprocessor
    preprocessor = HeartSoundPreprocessor(
        sr=4000,
        max_len=3000,
        cutoff_freq=500,
        scattering_j=6
    )
    
    # Xử lý dataset
    RAW_DATA_DIR = '../data/raw/training_data/'
    METADATA_FILE = '../data/raw/training_data.csv'
    OUTPUT_DIR = '../data/processed_cnn_method/'
    
    X, y = preprocessor.process_dataset(RAW_DATA_DIR, METADATA_FILE)
    
    # Balance dataset
    X_balanced, y_balanced = preprocessor.balance_dataset(X, y)
    
    # Save
    preprocessor.save_processed_data(X_balanced, y_balanced, OUTPUT_DIR, prefix='balanced')
