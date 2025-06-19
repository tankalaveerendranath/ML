import React, { useState } from 'react';
import { Calculator, ChevronRight, ChevronDown, BarChart3, TrendingUp } from 'lucide-react';
import { MathematicalExample as MathExample } from '../../types';

interface MathematicalExampleProps {
  example: MathExample;
}

export const MathematicalExample: React.FC<MathematicalExampleProps> = ({ example }) => {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const toggleStep = (stepIndex: number) => {
    setExpandedStep(expandedStep === stepIndex ? null : stepIndex);
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
      <div className="flex items-center mb-6">
        <Calculator className="w-6 h-6 text-purple-600 mr-3" />
        <h2 className="text-2xl font-bold text-gray-900">{example.title}</h2>
      </div>

      {/* Dataset Overview */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 text-blue-600 mr-2" />
          Dataset
        </h3>
        <p className="text-gray-600 mb-4">{example.dataset.description}</p>
        
        <div className="bg-gray-50 rounded-lg p-4 mb-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600">{example.dataset.data.length}</div>
              <div className="text-sm text-gray-600">Data Points</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-green-600">{example.dataset.features.length}</div>
              <div className="text-sm text-gray-600">Features</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-purple-600">1</div>
              <div className="text-sm text-gray-600">Target Variable</div>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  {example.dataset.features.map((feature, index) => (
                    <th key={index} className="text-left py-2 px-3 text-sm font-medium text-gray-700">
                      {feature}
                    </th>
                  ))}
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-700">
                    {example.dataset.target}
                  </th>
                </tr>
              </thead>
              <tbody>
                {example.dataset.data.slice(0, 5).map((row, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    {Object.values(row).map((value, cellIndex) => (
                      <td key={cellIndex} className="py-2 px-3 text-sm text-gray-600">
                        {typeof value === 'number' ? value.toLocaleString() : value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Step-by-step Calculations */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
          Step-by-Step Calculations
        </h3>
        
        <div className="space-y-3">
          {example.calculations.map((calc, index) => (
            <div key={index} className="border border-gray-200 rounded-lg">
              <button
                onClick={() => toggleStep(index)}
                className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                    {calc.step}
                  </div>
                  <span className="font-medium text-gray-900">{calc.title}</span>
                </div>
                {expandedStep === index ? (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronRight className="w-5 h-5 text-gray-400" />
                )}
              </button>
              
              {expandedStep === index && (
                <div className="px-4 pb-4 border-t border-gray-100">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Formula:</h4>
                      <div className="bg-blue-50 p-3 rounded-lg font-mono text-sm text-blue-800">
                        {calc.formula}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Calculation:</h4>
                      <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700">
                        {calc.calculation}
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">Result:</h4>
                    <div className="bg-green-50 p-3 rounded-lg">
                      <div className="font-mono text-sm text-green-800 mb-2">{calc.result}</div>
                      <div className="text-sm text-green-700">{calc.explanation}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Final Result */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">{example.result.description}</h3>
        <div className="bg-white p-4 rounded-lg mb-3">
          <div className="font-mono text-lg text-purple-800 font-bold">
            {example.result.value}
          </div>
        </div>
        <p className="text-gray-700 text-sm leading-relaxed">
          <strong>Interpretation:</strong> {example.result.interpretation}
        </p>
      </div>
    </div>
  );
};