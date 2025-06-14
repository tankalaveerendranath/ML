import React, { useState, useEffect } from 'react';

interface DataPoint {
  x: number;
  y: number;
  class: 'A' | 'B';
  id: number;
  isSupport?: boolean;
}

interface SVMVizProps {
  isActive: boolean;
  step: number;
}

export const SVMViz: React.FC<SVMVizProps> = ({ isActive, step }) => {
  const [points] = useState<DataPoint[]>([
    // Class A points (blue)
    { x: 80, y: 60, class: 'A', id: 1 },
    { x: 90, y: 80, class: 'A', id: 2 },
    { x: 70, y: 90, class: 'A', id: 3 },
    { x: 100, y: 70, class: 'A', id: 4 },
    { x: 85, y: 100, class: 'A', id: 5 },
    { x: 110, y: 90, class: 'A', id: 6, isSupport: true },
    
    // Class B points (red)
    { x: 180, y: 140, class: 'B', id: 7 },
    { x: 200, y: 160, class: 'B', id: 8 },
    { x: 190, y: 180, class: 'B', id: 9 },
    { x: 170, y: 170, class: 'B', id: 10 },
    { x: 210, y: 150, class: 'B', id: 11 },
    { x: 160, y: 150, class: 'B', id: 12, isSupport: true }
  ]);

  const [hyperplane, setHyperplane] = useState({ 
    x1: 50, y1: 120, x2: 250, y2: 140 
  });
  
  const [margins, setMargins] = useState({
    upper: { x1: 50, y1: 100, x2: 250, y2: 120 },
    lower: { x1: 50, y1: 140, x2: 250, y2: 160 }
  });

  const [showMargins, setShowMargins] = useState(false);
  const [showSupportVectors, setShowSupportVectors] = useState(false);

  useEffect(() => {
    if (isActive) {
      if (step >= 3) {
        setShowSupportVectors(true);
      }
      if (step >= 4) {
        setShowMargins(true);
      }
    }
  }, [isActive, step]);

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <h4 className="text-lg font-semibold mb-4 text-center">Support Vector Machine Visualization</h4>
      
      <svg width="300" height="250" className="mx-auto border border-gray-300 rounded">
        {/* Grid background */}
        <defs>
          <pattern id="svm-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f5f5f5" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#svm-grid)" />
        
        {/* Margin lines */}
        {showMargins && (
          <g>
            <line
              x1={margins.upper.x1}
              y1={margins.upper.y1}
              x2={margins.upper.x2}
              y2={margins.upper.y2}
              stroke="#94a3b8"
              strokeWidth="2"
              strokeDasharray="5,5"
              className="animate-fade-in"
            />
            <line
              x1={margins.lower.x1}
              y1={margins.lower.y1}
              x2={margins.lower.x2}
              y2={margins.lower.y2}
              stroke="#94a3b8"
              strokeWidth="2"
              strokeDasharray="5,5"
              className="animate-fade-in"
            />
            
            {/* Margin area */}
            <rect
              x={margins.upper.x1}
              y={margins.upper.y1}
              width={margins.upper.x2 - margins.upper.x1}
              height={margins.lower.y1 - margins.upper.y1}
              fill="#f1f5f9"
              opacity="0.3"
              className="animate-fade-in"
            />
          </g>
        )}
        
        {/* Decision boundary (hyperplane) */}
        {isActive && step >= 4 && (
          <line
            x1={hyperplane.x1}
            y1={hyperplane.y1}
            x2={hyperplane.x2}
            y2={hyperplane.y2}
            stroke="#000"
            strokeWidth="3"
            className="animate-fade-in"
          />
        )}
        
        {/* Data points */}
        {points.map((point, index) => (
          <g key={point.id}>
            <circle
              cx={point.x}
              cy={point.y}
              r="6"
              fill={point.class === 'A' ? '#3b82f6' : '#ef4444'}
              stroke="#fff"
              strokeWidth="2"
              className={`transition-all duration-500 ${
                isActive && step >= 1 ? 'opacity-100' : 'opacity-0'
              }`}
              style={{ transitionDelay: `${index * 50}ms` }}
            />
            
            {/* Support vector highlighting */}
            {showSupportVectors && point.isSupport && (
              <circle
                cx={point.x}
                cy={point.y}
                r="12"
                fill="none"
                stroke="#fbbf24"
                strokeWidth="3"
                className="animate-pulse"
              />
            )}
          </g>
        ))}
        
        {/* Support vector lines to hyperplane */}
        {showSupportVectors && step >= 5 && points
          .filter(p => p.isSupport)
          .map(point => {
            // Calculate perpendicular distance to hyperplane
            const perpX = point.class === 'A' ? point.x + 20 : point.x - 20;
            const perpY = hyperplane.y1 + ((perpX - hyperplane.x1) * (hyperplane.y2 - hyperplane.y1)) / (hyperplane.x2 - hyperplane.x1);
            
            return (
              <line
                key={`support-${point.id}`}
                x1={point.x}
                y1={point.y}
                x2={perpX}
                y2={perpY}
                stroke="#fbbf24"
                strokeWidth="2"
                strokeDasharray="3,3"
                className="animate-fade-in"
              />
            );
          })
        }
        
        {/* Class regions */}
        {step >= 6 && (
          <g>
            <text x="80" y="40" textAnchor="middle" className="text-sm font-bold fill-blue-600">
              Class A
            </text>
            <text x="200" y="220" textAnchor="middle" className="text-sm font-bold fill-red-600">
              Class B
            </text>
          </g>
        )}
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm mb-2">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>Class A</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
            <span>Class B</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-1 bg-black mr-2"></div>
            <span>Decision Boundary</span>
          </div>
        </div>
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 border-2 border-yellow-500 rounded-full mr-2"></div>
            <span>Support Vectors</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-1 bg-gray-400 mr-2" style={{ borderStyle: 'dashed' }}></div>
            <span>Margins</span>
          </div>
        </div>
        {isActive && step >= 4 && (
          <div className="mt-2 text-xs text-gray-600">
            Maximizing margin between classes using support vectors
          </div>
        )}
      </div>
    </div>
  );
};