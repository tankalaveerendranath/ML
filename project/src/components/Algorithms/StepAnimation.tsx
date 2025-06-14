import React, { useEffect, useState } from 'react';
import { AlgorithmStep } from '../../types';
import { LinearRegressionViz } from './Visualizations/LinearRegressionViz';
import { DecisionTreeViz } from './Visualizations/DecisionTreeViz';
import { RandomForestViz } from './Visualizations/RandomForestViz';
import { KMeansViz } from './Visualizations/KMeansViz';
import { NeuralNetworkViz } from './Visualizations/NeuralNetworkViz';
import { SVMViz } from './Visualizations/SVMViz';
import { PseudocodeDisplay } from './PseudocodeDisplay';

interface StepAnimationProps {
  step: AlgorithmStep;
  isActive: boolean;
  algorithmId: string;
}

export const StepAnimation: React.FC<StepAnimationProps> = ({ step, isActive, algorithmId }) => {
  const [shouldAnimate, setShouldAnimate] = useState(false);

  useEffect(() => {
    if (isActive) {
      setShouldAnimate(true);
    }
  }, [isActive]);

  const getAnimationClass = () => {
    if (!shouldAnimate) return 'opacity-0 transform translate-y-4';
    
    switch (step.animation) {
      case 'slideInLeft':
        return 'animate-slide-in-left';
      case 'slideInRight':
        return 'animate-slide-in-right';
      case 'slideInUp':
        return 'animate-slide-in-up';
      case 'slideInDown':
        return 'animate-slide-in-down';
      case 'fadeInUp':
        return 'animate-fade-in-up';
      case 'fadeInDown':
        return 'animate-fade-in-down';
      case 'bounceIn':
        return 'animate-bounce-in';
      case 'zoomIn':
        return 'animate-zoom-in';
      case 'rotateIn':
        return 'animate-rotate-in';
      default:
        return 'animate-fade-in';
    }
  };

  const renderVisualization = () => {
    if (!isActive) return null;

    switch (algorithmId) {
      case 'linear-regression':
        return <LinearRegressionViz isActive={isActive} step={step.id} />;
      case 'decision-tree':
        return <DecisionTreeViz isActive={isActive} step={step.id} />;
      case 'random-forest':
        return <RandomForestViz isActive={isActive} step={step.id} />;
      case 'kmeans':
        return <KMeansViz isActive={isActive} step={step.id} />;
      case 'neural-networks':
        return <NeuralNetworkViz isActive={isActive} step={step.id} />;
      case 'svm':
        return <SVMViz isActive={isActive} step={step.id} />;
      default:
        return null;
    }
  };

  return (
    <div className={`transition-all duration-700 ${getAnimationClass()}`}>
      <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
        isActive 
          ? 'border-blue-300 bg-blue-50 shadow-lg' 
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}>
        <div className="flex items-start space-x-4">
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
            isActive 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-600'
          }`}>
            {step.id}
          </div>
          <div className="flex-1">
            <h3 className={`font-semibold mb-2 transition-colors ${
              isActive ? 'text-blue-900' : 'text-gray-900'
            }`}>
              {step.title}
            </h3>
            <p className={`text-sm transition-colors ${
              isActive ? 'text-blue-700' : 'text-gray-600'
            }`}>
              {step.description}
            </p>
            {step.code && isActive && (
              <div className="mt-3 p-3 bg-gray-900 rounded-lg">
                <code className="text-green-400 text-xs font-mono">
                  {step.code}
                </code>
              </div>
            )}
          </div>
        </div>
        
        {/* Visualization Component */}
        {isActive && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              {renderVisualization()}
            </div>
            <div>
              <PseudocodeDisplay 
                algorithmId={algorithmId} 
                isActive={isActive} 
                step={step.id} 
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};