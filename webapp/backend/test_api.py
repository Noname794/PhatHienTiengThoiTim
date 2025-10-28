"""
Script để test API của Heart Sound Classifier
"""

import requests
import os
import sys

API_URL = "http://localhost:5000/api"

def test_health():
    """Test health check endpoint"""
    print("Testing /api/health...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting /api/model-info...")
    try:
        response = requests.get(f"{API_URL}/model-info")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model: {data.get('model_name')}")
        print(f"Accuracy: {data.get('accuracy')}")
        print(f"Classes: {data.get('classes')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict(audio_path, tsv_path=None):
    """Test prediction endpoint"""
    print(f"\nTesting /api/predict with {audio_path}...")
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return False
    
    try:
        files = {'audio': open(audio_path, 'rb')}
        
        if tsv_path and os.path.exists(tsv_path):
            files['segmentation'] = open(tsv_path, 'rb')
            print(f"Including segmentation file: {tsv_path}")
        
        response = requests.post(f"{API_URL}/predict", files=files)
        
        # Close files
        for f in files.values():
            f.close()
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*50)
            print("PREDICTION RESULTS")
            print("="*50)
            print(f"Prediction: {data.get('prediction')}")
            print(f"Confidence: {data.get('confidence'):.2%}")
            print(f"\nProbabilities:")
            for cls, prob in data.get('probabilities', {}).items():
                print(f"  {cls}: {prob:.2%}")
            print("="*50)
            return True
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("Heart Sound Classifier API Test")
    print("="*60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        print("Start the server with: python app.py")
        return
    
    print("✅ Health check passed")
    
    # Test 2: Model info
    if not test_model_info():
        print("\n❌ Model info failed")
        return
    
    print("✅ Model info passed")
    
    # Test 3: Prediction
    if len(sys.argv) > 1:
        # Use provided file
        audio_path = sys.argv[1]
        tsv_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if test_predict(audio_path, tsv_path):
            print("\n✅ Prediction test passed")
        else:
            print("\n❌ Prediction test failed")
    else:
        # Try to find a sample file
        sample_dir = "../../data/raw/training_data"
        if os.path.exists(sample_dir):
            wav_files = [f for f in os.listdir(sample_dir) if f.endswith('.wav')]
            if wav_files:
                sample_file = wav_files[0]
                audio_path = os.path.join(sample_dir, sample_file)
                
                # Look for corresponding TSV
                tsv_file = sample_file.replace('.wav', '.tsv')
                tsv_path = os.path.join(sample_dir, tsv_file)
                
                if test_predict(audio_path, tsv_path if os.path.exists(tsv_path) else None):
                    print("\n✅ Prediction test passed")
                else:
                    print("\n❌ Prediction test failed")
            else:
                print("\nNo sample WAV files found for testing")
                print("Usage: python test_api.py <audio_file.wav> [segmentation.tsv]")
        else:
            print("\nNo sample data found")
            print("Usage: python test_api.py <audio_file.wav> [segmentation.tsv]")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
