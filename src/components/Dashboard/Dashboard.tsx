import React, { useState } from 'react';
import { Upload, Sparkles } from 'lucide-react';
import { Header } from '../Layout/Header';
import { Footer } from '../Layout/Footer';
import { AlgorithmCard } from './AlgorithmCard';
import { AlgorithmExplainer } from '../Algorithms/AlgorithmExplainer';
import { DatasetUpload } from '../DatasetUpload/DatasetUpload';
import { AboutML } from '../Pages/AboutML';
import { Contact } from '../Pages/Contact';
import { algorithms } from '../../data/algorithms';
import { Algorithm } from '../../types';

type ViewType = 'dashboard' | 'algorithm' | 'dataset' | 'about' | 'contact';

export const Dashboard: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewType>('dashboard');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);

  const handleAlgorithmClick = (algorithm: Algorithm) => {
    setSelectedAlgorithm(algorithm);
    setCurrentView('algorithm');
  };

  const handleNavigate = (view: string) => {
    setCurrentView(view as ViewType);
    setSelectedAlgorithm(null);
  };

  const handleBackToDashboard = () => {
    setCurrentView('dashboard');
    setSelectedAlgorithm(null);
  };

  if (currentView === 'algorithm' && selectedAlgorithm) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header currentView={currentView} onNavigate={handleNavigate} />
        <div className="flex-1">
          <AlgorithmExplainer 
            algorithm={selectedAlgorithm} 
            onBack={handleBackToDashboard}
          />
        </div>
        <Footer />
      </div>
    );
  }

  if (currentView === 'dataset') {
    return (
      <div className="min-h-screen flex flex-col">
        <Header currentView={currentView} onNavigate={handleNavigate} />
        <div className="flex-1 bg-gray-50">
          <DatasetUpload onBack={handleBackToDashboard} />
        </div>
        <Footer />
      </div>
    );
  }

  if (currentView === 'about') {
    return (
      <div className="min-h-screen flex flex-col">
        <Header currentView={currentView} onNavigate={handleNavigate} />
        <div className="flex-1">
          <AboutML />
        </div>
        <Footer />
      </div>
    );
  }

  if (currentView === 'contact') {
    return (
      <div className="min-h-screen flex flex-col">
        <Header currentView={currentView} onNavigate={handleNavigate} />
        <div className="flex-1">
          <Contact />
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header currentView={currentView} onNavigate={handleNavigate} />
      
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
                onClick={() => setCurrentView('dataset')}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-medium rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 shadow-md hover:shadow-lg"
              >
                <Upload className="w-5 h-5 mr-2" />
                Upload Dataset & Get Recommendations
              </button>
              <button 
                onClick={() => setCurrentView('about')}
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
      
      <Footer />
    </div>
  );
};