import React, { useEffect, useState } from 'react';

interface Point {
  x: number;
  y: number;
}

interface LinearRegressionVizProps {
  isActive: boolean;
  step: number;
}

export const LinearRegressionViz: React.FC<LinearRegressionVizProps> = ({ isActive, step }) => {
  const [points] = useState<Point[]>([
    { x: 20, y: 180 }, { x: 40, y: 160 }, { x: 60, y: 140 }, { x: 80, y: 130 },
    { x: 100, y: 110 }, { x: 120, y: 100 }, { x: 140, y: 80 }, { x: 160, y: 70 },
    { x: 180, y: 50 }, { x: 200, y: 40 }
  ]);
  
  const [currentLine, setCurrentLine] = useState({ slope: -0.5, intercept: 150 });
  const [iteration, setIteration] = useState(0);

  useEffect(() => {
    if (isActive && step >= 3) {
      const interval = setInterval(() => {
        setIteration(prev => {
          const newIteration = prev + 1;
          // Gradually improve the line fit
          const targetSlope = -0.7;
          const targetIntercept = 180;
          const progress = Math.min(newIteration / 10, 1);
          
          setCurrentLine({
            slope: -0.5 + (targetSlope + 0.5) * progress,
            intercept: 150 + (targetIntercept - 150) * progress
          });
          
          return newIteration > 15 ? 0 : newIteration;
        });
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isActive, step]);

  const getLineY = (x: number) => currentLine.slope * x + currentLine.intercept;

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <h4 className="text-lg font-semibold mb-4 text-center">Linear Regression Visualization</h4>
      
      <svg width="300" height="250" className="mx-auto border border-gray-300 rounded">
        {/* Grid lines */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Axes */}
        <line x1="20" y1="230" x2="280" y2="230" stroke="#666" strokeWidth="2" />
        <line x1="20" y1="230" x2="20" y2="20" stroke="#666" strokeWidth="2" />
        
        {/* Data points */}
        {points.map((point, index) => (
          <circle
            key={index}
            cx={point.x + 20}
            cy={point.y + 20}
            r="4"
            fill="#3b82f6"
            className={`transition-all duration-500 ${
              isActive && step >= 1 ? 'opacity-100' : 'opacity-0'
            }`}
            style={{ transitionDelay: `${index * 100}ms` }}
          />
        ))}
        
        {/* Regression line */}
        {isActive && step >= 3 && (
          <line
            x1="20"
            y1={getLineY(0) + 20}
            x2="280"
            y2={getLineY(260) + 20}
            stroke="#ef4444"
            strokeWidth="3"
            className="transition-all duration-500"
          />
        )}
        
        {/* Prediction lines (residuals) */}
        {isActive && step >= 4 && points.map((point, index) => (
          <line
            key={`residual-${index}`}
            x1={point.x + 20}
            y1={point.y + 20}
            x2={point.x + 20}
            y2={getLineY(point.x) + 20}
            stroke="#fbbf24"
            strokeWidth="2"
            strokeDasharray="3,3"
            className="transition-all duration-300"
            style={{ transitionDelay: `${index * 50}ms` }}
          />
        ))}
        
        {/* Labels */}
        <text x="150" y="245" textAnchor="middle" className="text-xs fill-gray-600">Features (X)</text>
        <text x="10" y="125" textAnchor="middle" className="text-xs fill-gray-600" transform="rotate(-90 10 125)">Target (Y)</text>
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>Data Points</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-1 bg-red-500 mr-2"></div>
            <span>Best Fit Line</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-1 bg-yellow-500 mr-2" style={{ borderStyle: 'dashed' }}></div>
            <span>Residuals</span>
          </div>
        </div>
        {isActive && step >= 5 && (
          <div className="mt-2 text-xs text-gray-600">
            Iteration: {iteration} | MSE: {(Math.random() * 0.5 + 0.1).toFixed(3)}
          </div>
        )}
      </div>
    </div>
  );
};