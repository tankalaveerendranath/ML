import React, { useState } from 'react';
import { ArrowLeft, Play, Pause, RotateCcw, CheckCircle, XCircle } from 'lucide-react';
import { Algorithm } from '../../types';
import { StepAnimation } from './StepAnimation';

interface AlgorithmExplainerProps {
  algorithm: Algorithm;
  onBack: () => void;
}

export const AlgorithmExplainer: React.FC<AlgorithmExplainerProps> = ({ algorithm, onBack }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && currentStep < algorithm.steps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1;
          setCompletedSteps(completed => new Set([...completed, prev]));
          return next;
        });
      }, 4000); // Increased time to allow for animations
    } else if (isPlaying && currentStep === algorithm.steps.length - 1) {
      setCompletedSteps(completed => new Set([...completed, currentStep]));
      setIsPlaying(false);
    }

    return () => clearInterval(interval);
  }, [isPlaying, currentStep, algorithm.steps.length]);

  const handlePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setCompletedSteps(new Set());
  };

  const handleStepClick = (stepIndex: number) => {
    setCurrentStep(stepIndex);
    setIsPlaying(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={onBack}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Dashboard</span>
            </button>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={handlePlay}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isPlaying 
                    ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                    : 'bg-green-100 text-green-700 hover:bg-green-200'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Play Animation</span>
                  </>
                )}
              </button>
              
              <button
                onClick={handleReset}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reset</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Algorithm Header */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <div className="flex flex-col lg:flex-row lg:items-start lg:space-x-8">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-gray-900 mb-4">{algorithm.name}</h1>
              <p className="text-lg text-gray-600 mb-6">{algorithm.description}</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                    Advantages
                  </h3>
                  <ul className="space-y-2">
                    {algorithm.pros.map((pro, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                    <XCircle className="w-5 h-5 text-red-600 mr-2" />
                    Limitations
                  </h3>
                  <ul className="space-y-2">
                    {algorithm.cons.map((con, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="w-1.5 h-1.5 bg-red-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="lg:w-80 mt-6 lg:mt-0">
              <div className="bg-gray-50 rounded-xl p-6">
                <h3 className="font-semibold text-gray-900 mb-4">Algorithm Details</h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm text-gray-500">Category:</span>
                    <span className="ml-2 text-sm font-medium text-gray-900">{algorithm.category}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">Complexity:</span>
                    <span className="ml-2 text-sm font-medium text-gray-900">{algorithm.complexity}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">Common Use Cases:</span>
                    <p className="text-sm text-gray-700 mt-1">{algorithm.useCase}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Learning Progress</h2>
            <span className="text-sm text-gray-500">
              Step {currentStep + 1} of {algorithm.steps.length}
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${((currentStep + 1) / algorithm.steps.length) * 100}%` }}
            ></div>
          </div>
          
          <div className="flex space-x-2 overflow-x-auto">
            {algorithm.steps.map((step, index) => (
              <button
                key={step.id}
                onClick={() => handleStepClick(index)}
                className={`flex-shrink-0 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  index === currentStep
                    ? 'bg-blue-600 text-white'
                    : completedSteps.has(index)
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {step.title}
              </button>
            ))}
          </div>
        </div>

        {/* Algorithm Steps */}
        <div className="space-y-6">
          {algorithm.steps.map((step, index) => (
            <StepAnimation
              key={step.id}
              step={step}
              isActive={index === currentStep}
              algorithmId={algorithm.id}
            />
          ))}
        </div>
      </div>
    </div>
  );
};