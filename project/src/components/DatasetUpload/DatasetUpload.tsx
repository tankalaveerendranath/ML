import React, { useState, useCallback } from 'react';
import { ArrowLeft, Upload, FileText, BarChart3, Brain, Zap } from 'lucide-react';
import { Dataset, DatasetRecommendation } from '../../types';

interface DatasetUploadProps {
  onBack: () => void;
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({ onBack }) => {
  const [file, setFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [recommendations, setRecommendations] = useState<DatasetRecommendation[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const analyzeDataset = useCallback((file: File) => {
    setIsAnalyzing(true);
    
    // Simulate dataset analysis
    setTimeout(() => {
      const mockDataset: Dataset = {
        name: file.name,
        size: Math.round(file.size / 1024), // KB
        features: Math.floor(Math.random() * 20) + 5,
        samples: Math.floor(Math.random() * 10000) + 1000,
        type: ['numerical', 'categorical', 'mixed'][Math.floor(Math.random() * 3)] as any,
        target: ['classification', 'regression', 'clustering'][Math.floor(Math.random() * 3)] as any
      };

      const mockRecommendations = generateRecommendations(mockDataset);
      
      setDataset(mockDataset);
      setRecommendations(mockRecommendations);
      setIsAnalyzing(false);
    }, 2000);
  }, []);

  const generateRecommendations = (dataset: Dataset): DatasetRecommendation[] => {
    const recommendations: DatasetRecommendation[] = [];
    
    // Logic based on dataset characteristics
    if (dataset.target === 'classification') {
      if (dataset.samples < 1000) {
        recommendations.push({
          algorithm: 'K-Nearest Neighbors',
          confidence: 85,
          reasoning: 'KNN works well with small datasets and provides good classification results.'
        });
      }
      
      if (dataset.features > 10) {
        recommendations.push({
          algorithm: 'Random Forest',
          confidence: 92,
          reasoning: 'Random Forest handles high-dimensional data well and provides feature importance.'
        });
      } else {
        recommendations.push({
          algorithm: 'Decision Tree',
          confidence: 78,
          reasoning: 'Decision trees are interpretable and work well with moderate feature counts.'
        });
      }
      
      if (dataset.samples > 5000) {
        recommendations.push({
          algorithm: 'Support Vector Machine',
          confidence: 88,
          reasoning: 'SVM is effective for large datasets with complex decision boundaries.'
        });
      }
    }
    
    if (dataset.target === 'regression') {
      recommendations.push({
        algorithm: 'Linear Regression',
        confidence: 75,
        reasoning: 'Good baseline for regression problems, especially with numerical features.'
      });
      
      if (dataset.features > 5) {
        recommendations.push({
          algorithm: 'Random Forest',
          confidence: 89,
          reasoning: 'Handles non-linear relationships and multiple features effectively.'
        });
      }
    }
    
    if (dataset.target === 'clustering') {
      recommendations.push({
        algorithm: 'K-Means Clustering',
        confidence: 82,
        reasoning: 'Effective for discovering natural groupings in your data.'
      });
    }
    
    if (dataset.samples > 10000 && dataset.features > 20) {
      recommendations.push({
        algorithm: 'Neural Networks',
        confidence: 91,
        reasoning: 'Deep learning excels with large datasets and high-dimensional feature spaces.'
      });
    }
    
    return recommendations.sort((a, b) => b.confidence - a.confidence).slice(0, 3);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      analyzeDataset(droppedFile);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      analyzeDataset(selectedFile);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-100';
    if (confidence >= 80) return 'text-blue-600 bg-blue-100';
    if (confidence >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Dashboard</span>
        </button>
        
        <h1 className="text-2xl font-bold text-gray-900">Dataset Analysis & Recommendations</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Upload Your Dataset</h2>
            
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                dragActive
                  ? 'border-blue-400 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Drop your dataset here
              </h3>
              <p className="text-gray-600 mb-4">
                or click to browse your files
              </p>
              <input
                type="file"
                onChange={handleFileChange}
                accept=".csv,.json,.xlsx,.xls"
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors"
              >
                Choose File
              </label>
              <p className="text-xs text-gray-500 mt-3">
                Supported formats: CSV, JSON, Excel (.xlsx, .xls)
              </p>
            </div>

            {file && (
              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <div>
                    <p className="font-medium text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Dataset Info */}
          {dataset && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Dataset Overview
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{dataset.samples.toLocaleString()}</div>
                  <div className="text-sm text-blue-700">Samples</div>
                </div>
                <div className="text-center p-4 bg-emerald-50 rounded-lg">
                  <div className="text-2xl font-bold text-emerald-600">{dataset.features}</div>
                  <div className="text-sm text-emerald-700">Features</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-sm font-medium text-purple-600 capitalize">{dataset.type}</div>
                  <div className="text-xs text-purple-700">Data Type</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-sm font-medium text-orange-600 capitalize">{dataset.target}</div>
                  <div className="text-xs text-orange-700">Problem Type</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Analysis & Recommendations */}
        <div className="space-y-6">
          {isAnalyzing && (
            <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
              <Brain className="w-12 h-12 text-blue-600 mx-auto mb-4 animate-pulse" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Analyzing Dataset...</h3>
              <p className="text-gray-600">Our AI is examining your data to provide the best algorithm recommendations.</p>
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                </div>
              </div>
            </div>
          )}

          {recommendations.length > 0 && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
                <Zap className="w-5 h-5 mr-2 text-yellow-500" />
                Algorithm Recommendations
              </h3>
              
              <div className="space-y-4">
                {recommendations.map((rec, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-medium text-gray-900">{rec.algorithm}</h4>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(rec.confidence)}`}>
                        {rec.confidence}% match
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">{rec.reasoning}</p>
                    
                    <div className="mt-3 flex items-center justify-between">
                      <div className="w-full bg-gray-200 rounded-full h-2 mr-3">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${rec.confidence}%` }}
                        ></div>
                      </div>
                      <button className="text-blue-600 hover:text-blue-700 text-sm font-medium whitespace-nowrap">
                        Learn More
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-700">
                  <strong>Pro Tip:</strong> These recommendations are based on your dataset characteristics. 
                  Consider trying multiple algorithms and comparing their performance on your specific problem.
                </p>
              </div>
            </div>
          )}

          {!isAnalyzing && !file && (
            <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready to Analyze</h3>
              <p className="text-gray-600">
                Upload your dataset to get personalized algorithm recommendations based on your data characteristics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};