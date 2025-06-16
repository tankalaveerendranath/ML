import React, { useState, useEffect } from 'react';

interface Tree {
  id: number;
  x: number;
  y: number;
  isVisible: boolean;
  prediction: 'A' | 'B';
}

interface RandomForestVizProps {
  isActive: boolean;
  step: number;
}

export const RandomForestViz: React.FC<RandomForestVizProps> = ({ isActive, step }) => {
  const [trees, setTrees] = useState<Tree[]>([
    { id: 1, x: 80, y: 100, isVisible: false, prediction: 'A' },
    { id: 2, x: 160, y: 100, isVisible: false, prediction: 'B' },
    { id: 3, x: 240, y: 100, isVisible: false, prediction: 'A' },
    { id: 4, x: 320, y: 100, isVisible: false, prediction: 'A' },
    { id: 5, x: 400, y: 100, isVisible: false, prediction: 'A' }
  ]);

  const [votes, setVotes] = useState<{ A: number; B: number }>({ A: 0, B: 0 });
  const [finalPrediction, setFinalPrediction] = useState<string>('');

  useEffect(() => {
    if (isActive) {
      if (step >= 3) {
        // Show trees one by one
        trees.forEach((tree, index) => {
          setTimeout(() => {
            setTrees(prev => prev.map(t => 
              t.id === tree.id ? { ...t, isVisible: true } : t
            ));
          }, index * 300);
        });
      }

      if (step >= 4) {
        // Show predictions
        setTimeout(() => {
          const aVotes = trees.filter(t => t.prediction === 'A').length;
          const bVotes = trees.filter(t => t.prediction === 'B').length;
          setVotes({ A: aVotes, B: bVotes });
        }, 2000);
      }

      if (step >= 5) {
        // Show final prediction
        setTimeout(() => {
          setFinalPrediction(votes.A > votes.B ? 'A' : 'B');
        }, 2500);
      }
    }
  }, [isActive, step, trees, votes.A, votes.B]);

  const renderTree = (tree: Tree) => (
    <g key={tree.id}>
      {/* Tree trunk */}
      <rect
        x={tree.x - 3}
        y={tree.y + 20}
        width="6"
        height="30"
        fill="#8b4513"
        className={`transition-all duration-500 ${
          tree.isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
        }`}
      />
      
      {/* Tree crown */}
      <circle
        cx={tree.x}
        cy={tree.y}
        r="25"
        fill="#22c55e"
        stroke="#16a34a"
        strokeWidth="2"
        className={`transition-all duration-500 ${
          tree.isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
        }`}
      />
      
      {/* Tree ID */}
      <text
        x={tree.x}
        y={tree.y + 4}
        textAnchor="middle"
        className="text-xs font-bold fill-white"
      >
        T{tree.id}
      </text>
      
      {/* Prediction bubble */}
      {step >= 4 && (
        <g>
          <circle
            cx={tree.x}
            cy={tree.y - 40}
            r="12"
            fill={tree.prediction === 'A' ? '#3b82f6' : '#ef4444'}
            className="animate-bounce"
          />
          <text
            x={tree.x}
            y={tree.y - 36}
            textAnchor="middle"
            className="text-xs font-bold fill-white"
          >
            {tree.prediction}
          </text>
        </g>
      )}
    </g>
  );

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <h4 className="text-lg font-semibold mb-4 text-center">Random Forest Visualization</h4>
      
      <svg width="480" height="200" className="mx-auto">
        {/* Bootstrap samples indicator */}
        {step >= 1 && (
          <g>
            <rect x="10" y="10" width="460" height="30" fill="#f3f4f6" stroke="#d1d5db" rx="5" />
            <text x="240" y="30" textAnchor="middle" className="text-sm fill-gray-700">
              Bootstrap Sampling: Creating {trees.length} different training sets
            </text>
          </g>
        )}
        
        {/* Trees */}
        {trees.map(renderTree)}
        
        {/* Voting section */}
        {step >= 5 && (
          <g>
            <rect x="150" y="160" width="180" height="35" fill="#f8fafc" stroke="#e2e8f0" rx="5" />
            <text x="240" y="175" textAnchor="middle" className="text-sm font-semibold fill-gray-800">
              Voting Results
            </text>
            <text x="240" y="190" textAnchor="middle" className="text-xs fill-gray-600">
              A: {votes.A} votes | B: {votes.B} votes
            </text>
          </g>
        )}
        
        {/* Final prediction */}
        {finalPrediction && (
          <g>
            <circle cx="240" cy="140" r="20" fill="#10b981" className="animate-pulse" />
            <text x="240" y="146" textAnchor="middle" className="text-lg font-bold fill-white">
              {finalPrediction}
            </text>
          </g>
        )}
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
            <span>Decision Tree</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
            <span>Prediction A</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
            <span>Prediction B</span>
          </div>
        </div>
        {step >= 2 && (
          <div className="mt-2 text-xs text-gray-600">
            Each tree trains on a different bootstrap sample with random feature selection
          </div>
        )}
      </div>
    </div>
  );
};