import React, { useState } from 'react';
import { ArrowLeft, Play, Pause, RotateCcw, CheckCircle, XCircle, Calculator } from 'lucide-react';
import { Algorithm } from '../../types';
import { StepAnimation } from './StepAnimation';
import { MathematicalCalculations } from './MathematicalCalculations';
import { motion } from 'framer-motion';

interface AlgorithmExplainerProps {
  algorithm: Algorithm;
  onBack: () => void;
}

export const AlgorithmExplainer: React.FC<AlgorithmExplainerProps> = ({ algorithm, onBack }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [showCalculations, setShowCalculations] = useState(false);

  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && currentStep < algorithm.steps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1;
          setCompletedSteps(completed => new Set([...completed, prev]));
          return next;
        });
      }, 4000);
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
                onClick={() => setShowCalculations(!showCalculations)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  showCalculations 
                    ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Calculator className="w-4 h-4" />
                <span>{showCalculations ? 'Hide' : 'Show'} Calculations</span>
              </button>
              
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
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg p-8 mb-8"
        >
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
                      <motion.li 
                        key={index}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="text-sm text-gray-600 flex items-start"
                      >
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                        {pro}
                      </motion.li>
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
                      <motion.li 
                        key={index}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 + 0.3 }}
                        className="text-sm text-gray-600 flex items-start"
                      >
                        <span className="w-1.5 h-1.5 bg-red-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                        {con}
                      </motion.li>
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
        </motion.div>

        {/* Mathematical Calculations */}
        {showCalculations && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mb-8"
          >
            <MathematicalCalculations algorithm={algorithm} currentStep={currentStep + 1} />
          </motion.div>
        )}

        {/* Progress Bar */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl shadow-sm p-6 mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Learning Progress</h2>
            <span className="text-sm text-gray-500">
              Step {currentStep + 1} of {algorithm.steps.length}
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
            <motion.div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
              initial={{ width: "0%" }}
              animate={{ width: `${((currentStep + 1) / algorithm.steps.length) * 100}%` }}
            />
          </div>
          
          <div className="flex space-x-2 overflow-x-auto">
            {algorithm.steps.map((step, index) => (
              <button
                key={step.id}
                onClick={() => handleStepClick(index)}
                className={`flex-shrink-0 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  index === currentStep
                    ? 'bg-blue-600 text-white shadow-md'
                    : completedSteps.has(index)
                    ? 'bg-green-100 text-green-700 hover:bg-green-200'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {step.title}
              </button>
            ))}
          </div>
        </motion.div>

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