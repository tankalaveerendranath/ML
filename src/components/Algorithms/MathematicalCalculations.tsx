import React, { useState } from 'react';
import { Calculator, ChevronDown, ChevronRight, BookOpen } from 'lucide-react';
import { Algorithm, CalculationStep } from '../../types';
import { motion, AnimatePresence } from 'framer-motion';

interface MathematicalCalculationsProps {
  algorithm: Algorithm;
  currentStep: number;
}

export const MathematicalCalculations: React.FC<MathematicalCalculationsProps> = ({ 
  algorithm, 
  currentStep 
}) => {
  const [expandedFormula, setExpandedFormula] = useState<number | null>(null);
  const [showStepCalculations, setShowStepCalculations] = useState(true);

  const getCalculationSteps = (): CalculationStep[] => {
    const steps: CalculationStep[] = [];
    
    algorithm.steps.forEach((step, index) => {
      if (step.mathematicalStep && step.exampleCalculation) {
        steps.push({
          step: index + 1,
          description: step.title,
          formula: step.mathematicalStep,
          calculation: step.exampleCalculation,
          result: extractResult(step.exampleCalculation)
        });
      }
    });
    
    return steps;
  };

  const extractResult = (calculation: string): string | number => {
    // Extract numerical results from calculation strings
    const matches = calculation.match(/=\s*([\d.,]+)/);
    if (matches) {
      return matches[1];
    }
    // Extract final predictions or outcomes
    const finalMatches = calculation.match(/→\s*(.+)$/);
    if (finalMatches) {
      return finalMatches[1];
    }
    return 'See calculation';
  };

  const calculationSteps = getCalculationSteps();

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-900 flex items-center">
          <Calculator className="w-6 h-6 mr-2 text-blue-600" />
          Mathematical Calculations
        </h3>
        <button
          onClick={() => setShowStepCalculations(!showStepCalculations)}
          className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 transition-colors"
        >
          {showStepCalculations ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="text-sm font-medium">
            {showStepCalculations ? 'Hide' : 'Show'} Step Calculations
          </span>
        </button>
      </div>

      {/* Mathematical Formulas */}
      <div className="mb-8">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BookOpen className="w-5 h-5 mr-2" />
          Key Formulas
        </h4>
        <div className="space-y-3">
          {algorithm.mathematicalFormulas.map((formula, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border border-gray-200 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => setExpandedFormula(expandedFormula === index ? null : index)}
                className="w-full p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h5 className="font-medium text-gray-900">{formula.name}</h5>
                    <code className="text-sm text-blue-600 font-mono bg-blue-50 px-2 py-1 rounded mt-1 inline-block">
                      {formula.formula}
                    </code>
                  </div>
                  {expandedFormula === index ? 
                    <ChevronDown className="w-4 h-4 text-gray-400" /> : 
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  }
                </div>
              </button>
              
              <AnimatePresence>
                {expandedFormula === index && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="border-t border-gray-200 p-4 bg-gray-50"
                  >
                    <p className="text-gray-700 mb-3">{formula.description}</p>
                    <div className="space-y-2">
                      <h6 className="font-medium text-gray-900">Variables:</h6>
                      {Object.entries(formula.variables).map(([variable, description]) => (
                        <div key={variable} className="flex items-start space-x-2 text-sm">
                          <code className="bg-gray-200 px-2 py-1 rounded font-mono text-gray-800 min-w-fit">
                            {variable}
                          </code>
                          <span className="text-gray-600">{description}</span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Step-by-Step Calculations */}
      <AnimatePresence>
        {showStepCalculations && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Example Calculations with {algorithm.exampleDataset.name}
            </h4>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-6">
              <h5 className="font-medium text-blue-900 mb-2">Dataset: {algorithm.exampleDataset.name}</h5>
              <p className="text-blue-700 text-sm mb-3">{algorithm.exampleDataset.description}</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                {algorithm.exampleDataset.features.map((feature, index) => (
                  <div key={index} className="bg-white rounded px-2 py-1 text-blue-800 font-medium">
                    {feature}
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              {calculationSteps.map((step, index) => (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ 
                    opacity: index <= currentStep - 1 ? 1 : 0.5,
                    x: 0,
                    scale: index === currentStep - 1 ? 1.02 : 1
                  }}
                  transition={{ delay: index * 0.1 }}
                  className={`border rounded-lg p-4 transition-all duration-300 ${
                    index === currentStep - 1 
                      ? 'border-blue-300 bg-blue-50 shadow-md' 
                      : index < currentStep - 1
                      ? 'border-green-200 bg-green-50'
                      : 'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-start space-x-4">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                      index === currentStep - 1
                        ? 'bg-blue-600 text-white'
                        : index < currentStep - 1
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-300 text-gray-600'
                    }`}>
                      {step.step}
                    </div>
                    <div className="flex-1">
                      <h5 className="font-medium text-gray-900 mb-2">{step.description}</h5>
                      
                      <div className="space-y-3">
                        <div>
                          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Mathematical Step</span>
                          <div className="bg-white rounded p-3 border border-gray-200 mt-1">
                            <code className="text-sm font-mono text-gray-800">{step.formula}</code>
                          </div>
                        </div>
                        
                        <div>
                          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Example Calculation</span>
                          <div className="bg-white rounded p-3 border border-gray-200 mt-1">
                            <p className="text-sm text-gray-700">{step.calculation}</p>
                          </div>
                        </div>
                        
                        {typeof step.result === 'number' && (
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Result:</span>
                            <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm font-medium">
                              {step.result}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};