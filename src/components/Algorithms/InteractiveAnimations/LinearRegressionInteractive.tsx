import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';

interface Point {
  x: number;
  y: number;
  id: number;
}

interface LinearRegressionInteractiveProps {
  isActive: boolean;
  step: number;
}

export const LinearRegressionInteractive: React.FC<LinearRegressionInteractiveProps> = ({ isActive, step }) => {
  const [points, setPoints] = useState<Point[]>([
    { x: 20, y: 180, id: 1 }, { x: 40, y: 160, id: 2 }, { x: 60, y: 140, id: 3 }, 
    { x: 80, y: 130, id: 4 }, { x: 100, y: 110, id: 5 }, { x: 120, y: 100, id: 6 },
    { x: 140, y: 80, id: 7 }, { x: 160, y: 70, id: 8 }, { x: 180, y: 50, id: 9 }, { x: 200, y: 40, id: 10 }
  ]);
  
  const [currentLine, setCurrentLine] = useState({ slope: -0.3, intercept: 200 });
  const [iteration, setIteration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cost, setCost] = useState(0);
  const [gradients, setGradients] = useState({ slope: 0, intercept: 0 });
  const [learningRate] = useState(0.0001);

  const calculateCost = (slope: number, intercept: number) => {
    const predictions = points.map(p => slope * p.x + intercept);
    const errors = points.map((p, i) => predictions[i] - p.y);
    return errors.reduce((sum, error) => sum + error * error, 0) / (2 * points.length);
  };

  const calculateGradients = (slope: number, intercept: number) => {
    const predictions = points.map(p => slope * p.x + intercept);
    const errors = points.map((p, i) => predictions[i] - p.y);
    
    const slopeGradient = errors.reduce((sum, error, i) => sum + error * points[i].x, 0) / points.length;
    const interceptGradient = errors.reduce((sum, error) => sum + error, 0) / points.length;
    
    return { slope: slopeGradient, intercept: interceptGradient };
  };

  useEffect(() => {
    if (isActive && step >= 3 && isPlaying) {
      const interval = setInterval(() => {
        setIteration(prev => {
          const newIteration = prev + 1;
          
          // Calculate current cost and gradients
          const currentCost = calculateCost(currentLine.slope, currentLine.intercept);
          const currentGradients = calculateGradients(currentLine.slope, currentLine.intercept);
          
          setCost(currentCost);
          setGradients(currentGradients);
          
          // Update parameters using gradient descent
          setCurrentLine(prevLine => ({
            slope: prevLine.slope - learningRate * currentGradients.slope,
            intercept: prevLine.intercept - learningRate * currentGradients.intercept
          }));
          
          if (newIteration > 100) {
            setIsPlaying(false);
            return 0;
          }
          
          return newIteration;
        });
      }, 200);
      
      return () => clearInterval(interval);
    }
  }, [isActive, step, isPlaying, currentLine, learningRate, points]);

  const handleAddPoint = (event: React.MouseEvent<SVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left - 20;
    const y = event.clientY - rect.top - 20;
    
    if (x >= 0 && x <= 260 && y >= 0 && y <= 200) {
      const newPoint = { x, y, id: Date.now() };
      setPoints(prev => [...prev, newPoint]);
    }
  };

  const getLineY = (x: number) => currentLine.slope * x + currentLine.intercept;

  const reset = () => {
    setCurrentLine({ slope: -0.3, intercept: 200 });
    setIteration(0);
    setIsPlaying(false);
    setCost(0);
    setGradients({ slope: 0, intercept: 0 });
  };

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-semibold">Interactive Linear Regression</h4>
        <div className="flex space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg ${isPlaying ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}
            disabled={step < 3}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={reset}
            className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      <svg 
        width="300" 
        height="250" 
        className="mx-auto border border-gray-300 rounded cursor-crosshair"
        onClick={handleAddPoint}
      >
        {/* Grid */}
        <defs>
          <pattern id="interactive-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#interactive-grid)" />
        
        {/* Axes */}
        <line x1="20" y1="230" x2="280" y2="230" stroke="#666" strokeWidth="2" />
        <line x1="20" y1="230" x2="20" y2="20" stroke="#666" strokeWidth="2" />
        
        {/* Data points with animation */}
        {points.map((point, index) => (
          <g key={point.id}>
            <circle
              cx={point.x + 20}
              cy={point.y + 20}
              r="5"
              fill="#3b82f6"
              className={`transition-all duration-500 ${
                isActive && step >= 1 ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
              }`}
              style={{ 
                transitionDelay: `${index * 100}ms`,
                filter: isPlaying ? 'drop-shadow(0 0 8px #3b82f6)' : 'none'
              }}
            />
            {/* Point labels for first few points */}
            {index < 3 && step >= 1 && (
              <text
                x={point.x + 25}
                y={point.y + 15}
                className="text-xs fill-blue-600 font-medium"
              >
                ({Math.round(point.x)}, {Math.round(point.y)})
              </text>
            )}
          </g>
        ))}
        
        {/* Regression line with gradient animation */}
        {isActive && step >= 3 && (
          <line
            x1="20"
            y1={Math.max(20, Math.min(230, getLineY(0) + 20))}
            x2="280"
            y2={Math.max(20, Math.min(230, getLineY(260) + 20))}
            stroke="#ef4444"
            strokeWidth="3"
            className="transition-all duration-300"
            style={{
              filter: isPlaying ? 'drop-shadow(0 0 8px #ef4444)' : 'none',
              strokeDasharray: isPlaying ? '5,5' : 'none',
              animation: isPlaying ? 'dash 1s linear infinite' : 'none'
            }}
          />
        )}
        
        {/* Prediction lines (residuals) with staggered animation */}
        {isActive && step >= 4 && points.map((point, index) => {
          const predictedY = getLineY(point.x);
          return (
            <line
              key={`residual-${point.id}`}
              x1={point.x + 20}
              y1={point.y + 20}
              x2={point.x + 20}
              y2={Math.max(20, Math.min(230, predictedY + 20))}
              stroke="#fbbf24"
              strokeWidth="2"
              strokeDasharray="3,3"
              className="transition-all duration-300"
              style={{ 
                transitionDelay: `${index * 50}ms`,
                opacity: isPlaying ? 0.8 : 0.6
              }}
            />
          );
        })}
        
        {/* Cost visualization */}
        {step >= 3 && (
          <text x="150" y="15" textAnchor="middle" className="text-sm font-bold fill-red-600">
            Cost: {cost.toFixed(3)}
          </text>
        )}
      </svg>
      
      {/* Interactive Controls */}
      <div className="mt-4 space-y-3">
        {step >= 2 && (
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm font-medium text-blue-800 mb-2">Current Parameters:</div>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-blue-600">Slope (θ₁):</span>
                <span className="ml-2 font-mono">{currentLine.slope.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-blue-600">Intercept (θ₀):</span>
                <span className="ml-2 font-mono">{currentLine.intercept.toFixed(2)}</span>
              </div>
            </div>
          </div>
        )}
        
        {step >= 4 && (
          <div className="bg-yellow-50 p-3 rounded-lg">
            <div className="text-sm font-medium text-yellow-800 mb-2">Gradients:</div>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-yellow-600">∂J/∂θ₁:</span>
                <span className="ml-2 font-mono">{gradients.slope.toFixed(6)}</span>
              </div>
              <div>
                <span className="text-yellow-600">∂J/∂θ₀:</span>
                <span className="ml-2 font-mono">{gradients.intercept.toFixed(6)}</span>
              </div>
            </div>
          </div>
        )}
        
        {step >= 5 && (
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="text-sm font-medium text-green-800 mb-2">Training Progress:</div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-green-600">Iteration: {iteration}</span>
              <span className="text-green-600">Learning Rate: {learningRate}</span>
            </div>
            <div className="w-full bg-green-200 rounded-full h-2 mt-2">
              <div 
                className="bg-green-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, (iteration / 100) * 100)}%` }}
              ></div>
            </div>
          </div>
        )}
        
        <div className="text-xs text-gray-600 text-center">
          💡 Click on the chart to add more data points and see how the line adapts!
        </div>
      </div>
      
      <style jsx>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -10;
          }
        }
      `}</style>
    </div>
  );
};