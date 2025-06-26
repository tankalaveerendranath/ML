import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Sparkles } from 'lucide-react';
import { AlgorithmCard } from './AlgorithmCard';
import { algorithms } from '../../data/algorithms';
import { Algorithm } from '../../types';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const handleAlgorithmClick = (algorithm: Algorithm) => {
    navigate(`/algorithm/${algorithm.id}`);
  };

  return (
    <main className="flex-1 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Explore Machine Learning Algorithms
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Discover how different ML algorithms work through interactive explanations, 
            step-by-step animations, and intelligent dataset analysis.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/dataset')}
              className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-medium rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 shadow-md hover:shadow-lg"
            >
              <Upload className="w-5 h-5 mr-2" />
              Upload Dataset & Get Recommendations
            </button>
            <button 
              onClick={() => navigate('/about')}
              className="inline-flex items-center px-6 py-3 bg-white text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors border border-gray-300 shadow-sm"
            >
              <Sparkles className="w-5 h-5 mr-2" />
              Learn About Machine Learning
            </button>
          </div>
        </div>

        {/* Algorithm Categories */}
        <div className="mb-8">
          <h3 className="text-2xl font-semibold text-gray-900 mb-6">Algorithm Categories</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {['Supervised Learning', 'Unsupervised Learning', 'Ensemble Learning', 'Deep Learning'].map((category) => {
              const count = algorithms.filter(algo => algo.category === category).length;
              const colors = {
                'Supervised Learning': 'from-blue-500 to-blue-600',
                'Unsupervised Learning': 'from-purple-500 to-purple-600',
                'Ensemble Learning': 'from-emerald-500 to-emerald-600',
                'Deep Learning': 'from-orange-500 to-orange-600'
              };
              
              return (
                <div key={category} className={`bg-gradient-to-r ${colors[category as keyof typeof colors]} text-white p-4 rounded-lg`}>
                  <h4 className="font-semibold mb-1">{category}</h4>
                  <p className="text-sm opacity-90">{count} algorithm{count !== 1 ? 's' : ''}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Algorithms Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {algorithms.map((algorithm) => (
            <AlgorithmCard
              key={algorithm.id}
              algorithm={algorithm}
              onClick={() => handleAlgorithmClick(algorithm)}
            />
          ))}
        </div>
      </div>
    </main>
  );
};