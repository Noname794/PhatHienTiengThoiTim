import React, { useState, useRef } from 'react';
import { Upload, Heart, Activity, CheckCircle, XCircle, Loader, Info } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [segmentationFile, setSegmentationFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  
  const audioInputRef = useRef(null);
  const segmentationInputRef = useRef(null);

  // Fetch model info on mount
  React.useEffect(() => {
    axios.get(`${API_URL}/model-info`)
      .then(response => setModelInfo(response.data))
      .catch(err => console.error('Failed to fetch model info:', err));
  }, []);

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleSegmentationChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSegmentationFile(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!audioFile) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('audio', audioFile);
    if (segmentationFile) {
      formData.append('segmentation', segmentationFile);
    }

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setAudioFile(null);
    setSegmentationFile(null);
    setResult(null);
    setError(null);
    if (audioInputRef.current) audioInputRef.current.value = '';
    if (segmentationInputRef.current) segmentationInputRef.current.value = '';
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <div className="bg-white p-4 rounded-full shadow-lg">
              <Heart className="w-12 h-12 text-red-500 animate-pulse-slow" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            Heart Sound Classifier
          </h1>
          <p className="text-white/80 text-lg">
            AI-Powered Heart Murmur Detection using Deep Learning
          </p>
        </div>

        {/* Model Info Card */}
        {modelInfo && (
          <div className="glass-effect rounded-2xl p-6 mb-8 text-white">
            <div className="flex items-center mb-4">
              <Info className="w-5 h-5 mr-2" />
              <h2 className="text-xl font-semibold">Model Information</h2>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-white/70">Architecture</p>
                <p className="font-medium">{modelInfo.architecture}</p>
              </div>
              <div>
                <p className="text-white/70">Accuracy</p>
                <p className="font-medium">{(modelInfo.accuracy * 100).toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-white/70">Features</p>
                <p className="font-medium">{modelInfo.features}</p>
              </div>
              <div>
                <p className="text-white/70">Sampling Rate</p>
                <p className="font-medium">{modelInfo.sampling_rate} Hz</p>
              </div>
            </div>
          </div>
        )}

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div className="p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Audio File Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Heart Sound Recording (WAV) *
                </label>
                <div className="relative">
                  <input
                    ref={audioInputRef}
                    type="file"
                    accept=".wav"
                    onChange={handleAudioChange}
                    className="hidden"
                    id="audio-upload"
                  />
                  <label
                    htmlFor="audio-upload"
                    className="flex items-center justify-center w-full px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-primary-500 transition-colors"
                  >
                    <Upload className="w-6 h-6 text-gray-400 mr-2" />
                    <span className="text-gray-600">
                      {audioFile ? audioFile.name : 'Click to upload WAV file'}
                    </span>
                  </label>
                </div>
              </div>

              {/* Segmentation File Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Segmentation File (TSV)
                </label>
                <div className="relative">
                  <input
                    ref={segmentationInputRef}
                    type="file"
                    accept=".tsv"
                    onChange={handleSegmentationChange}
                    className="hidden"
                    id="segmentation-upload"
                  />
                  <label
                    htmlFor="segmentation-upload"
                    className="flex items-center justify-center w-full px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-primary-500 transition-colors"
                  >
                    <Upload className="w-6 h-6 text-gray-400 mr-2" />
                    <span className="text-gray-600">
                      {segmentationFile ? segmentationFile.name : 'Click to upload TSV file (optional)'}
                    </span>
                  </label>
                </div>
                <p className="mt-2 text-sm text-gray-500">
                  If provided, the model will use segmentation data for more accurate predictions
                </p>
              </div>

              {/* Buttons */}
              <div className="flex gap-4">
                <button
                  type="submit"
                  disabled={!audioFile || loading}
                  className="flex-1 flex items-center justify-center px-6 py-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg font-medium hover:from-primary-600 hover:to-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity className="w-5 h-5 mr-2" />
                      Analyze Heart Sound
                    </>
                  )}
                </button>
                
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                >
                  Reset
                </button>
              </div>
            </form>

            {/* Error Message */}
            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
                <XCircle className="w-5 h-5 text-red-500 mr-3 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="text-red-800 font-medium">Error</h3>
                  <p className="text-red-700 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="mt-8 space-y-6">
                <div className="border-t pt-6">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                    <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
                    Analysis Results
                  </h3>
                  
                  {/* Prediction Card */}
                  <div className={`p-6 rounded-xl ${
                    result.prediction === 'Present' 
                      ? 'bg-red-50 border-2 border-red-200' 
                      : 'bg-green-50 border-2 border-green-200'
                  }`}>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Prediction</p>
                        <p className={`text-3xl font-bold ${
                          result.prediction === 'Present' ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {result.prediction === 'Present' ? 'Murmur Present' : 'No Murmur'}
                        </p>
                      </div>
                      <div className={`p-4 rounded-full ${
                        result.prediction === 'Present' ? 'bg-red-100' : 'bg-green-100'
                      }`}>
                        {result.prediction === 'Present' ? (
                          <XCircle className="w-12 h-12 text-red-600" />
                        ) : (
                          <CheckCircle className="w-12 h-12 text-green-600" />
                        )}
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <p className="text-sm text-gray-600 mb-2">Confidence</p>
                      <div className="flex items-center">
                        <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                          <div
                            className={`h-3 rounded-full ${
                              result.prediction === 'Present' ? 'bg-red-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${result.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-lg font-semibold text-gray-900">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Probabilities */}
                  <div className="mt-6 grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Absent Probability</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(result.probabilities.Absent * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Present Probability</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(result.probabilities.Present * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Disclaimer */}
                  <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-sm text-yellow-800">
                      <strong>Disclaimer:</strong> This is an AI-based prediction tool for research purposes. 
                      Always consult with a qualified healthcare professional for medical diagnosis and treatment.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-white/70 text-sm">
          <p>Powered by 1D CNN with Scattering Transform</p>
          <p className="mt-1">Model Accuracy: {modelInfo ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : 'Loading...'}</p>
        </div>
      </div>
    </div>
  );
}

export default App;
