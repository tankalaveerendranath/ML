import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Info } from 'lucide-react';

interface DataPoint {
  x: number;
  y: number;
  class: 'A' | 'B';
  id: number;
}

interface TreeNode {
  id: string;
  question?: string;
  value?: string;
  left?: TreeNode;
  right?: TreeNode;
  x: number;
  y: number;
  isVisible: boolean;
  isHighlighted: boolean;
  splitLine?: { x1: number; y1: number; x2: number; y2: number };
}

interface DecisionTreeInteractiveProps {
  isActive: boolean;
  step: number;
}

export const DecisionTreeInteractive: React.FC<DecisionTreeInteractiveProps> = ({ isActive, step }) => {
  const [dataPoints] = useState<DataPoint[]>([
    { x: 60, y: 80, class: 'A', id: 1 }, { x: 80, y: 60, class: 'A', id: 2 },
    { x: 70, y: 90, class: 'A', id: 3 }, { x: 90, y: 70, class: 'A', id: 4 },
    { x: 180, y: 160, class: 'B', id: 5 }, { x: 200, y: 140, class: 'B', id: 6 },
    { x: 190, y: 170, class: 'B', id: 7 }, { x: 210, y: 150, class: 'B', id: 8 },
    { x: 120, y: 200, class: 'A', id: 9 }, { x: 140, y: 180, class: 'A', id: 10 },
    { x: 250, y: 100, class: 'B', id: 11 }, { x: 230, y: 120, class: 'B', id: 12 }
  ]);

  const [tree, setTree] = useState<TreeNode>({
    id: 'root',
    question: 'X > 150?',
    x: 200,
    y: 50,
    isVisible: false,
    isHighlighted: false,
    splitLine: { x1: 150, y1: 0, x2: 150, y2: 250 },
    left: {
      id: 'left1',
      question: 'Y > 120?',
      x: 100,
      y: 120,
      isVisible: false,
      isHighlighted: false,
      splitLine: { x1: 0, y1: 120, x2: 150, y2: 120 },
      left: {
        id: 'left2',
        value: 'Class A',
        x: 50,
        y: 190,
        isVisible: false,
        isHighlighted: false
      },
      right: {
        id: 'right2',
        value: 'Class A',
        x: 150,
        y: 190,
        isVisible: false,
        isHighlighted: false
      }
    },
    right: {
      id: 'right1',
      question: 'Y > 130?',
      x: 300,
      y: 120,
      isVisible: false,
      isHighlighted: false,
      splitLine: { x1: 150, y1: 130, x2: 300, y2: 130 },
      left: {
        id: 'left3',
        value: 'Class B',
        x: 250,
        y: 190,
        isVisible: false,
        isHighlighted: false
      },
      right: {
        id: 'right3',
        value: 'Class B',
        x: 350,
        y: 190,
        isVisible: false,
        isHighlighted: false
      }
    }
  });

  const [currentSplit, setCurrentSplit] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [informationGain, setInformationGain] = useState<number>(0);

  useEffect(() => {
    if (isActive && isPlaying) {
      const sequence = [
        () => updateNodeVisibility('root', true, true),
        () => {
          setCurrentSplit('root');
          setInformationGain(0.45);
        },
        () => {
          updateNodeVisibility('left1', true, true);
          updateNodeVisibility('right1', true, true);
          updateNodeVisibility('root', true, false);
        },
        () => {
          setCurrentSplit('left1');
          setInformationGain(0.32);
        },
        () => {
          updateNodeVisibility('left2', true, true);
          updateNodeVisibility('right2', true, true);
          updateNodeVisibility('left1', true, false);
        },
        () => {
          setCurrentSplit('right1');
          setInformationGain(0.28);
        },
        () => {
          updateNodeVisibility('left3', true, true);
          updateNodeVisibility('right3', true, true);
          updateNodeVisibility('right1', true, false);
          setCurrentSplit(null);
          setIsPlaying(false);
        }
      ];

      let currentStep = 0;
      const interval = setInterval(() => {
        if (currentStep < sequence.length) {
          sequence[currentStep]();
          currentStep++;
        } else {
          clearInterval(interval);
        }
      }, 1500);

      return () => clearInterval(interval);
    }
  }, [isActive, isPlaying]);

  const updateNodeVisibility = (nodeId: string, visible: boolean, highlighted: boolean) => {
    const updateNode = (node: TreeNode): TreeNode => {
      if (node.id === nodeId) {
        return { ...node, isVisible: visible, isHighlighted: highlighted };
      }
      return {
        ...node,
        left: node.left ? updateNode(node.left) : undefined,
        right: node.right ? updateNode(node.right) : undefined
      };
    };
    setTree(prev => updateNode(prev));
  };

  const reset = () => {
    setIsPlaying(false);
    setCurrentSplit(null);
    setInformationGain(0);
    const resetNode = (node: TreeNode): TreeNode => ({
      ...node,
      isVisible: false,
      isHighlighted: false,
      left: node.left ? resetNode(node.left) : undefined,
      right: node.right ? resetNode(node.right) : undefined
    });
    setTree(prev => resetNode(prev));
  };

  const renderNode = (node: TreeNode): JSX.Element => (
    <g key={node.id}>
      {/* Split line visualization */}
      {node.splitLine && node.isVisible && currentSplit === node.id && (
        <line
          x1={node.splitLine.x1}
          y1={node.splitLine.y1}
          x2={node.splitLine.x2}
          y2={node.splitLine.y2}
          stroke="#ff6b6b"
          strokeWidth="3"
          strokeDasharray="5,5"
          className="animate-pulse"
        />
      )}
      
      {/* Node shape */}
      {node.question ? (
        <rect
          x={node.x - 45}
          y={node.y - 20}
          width="90"
          height="40"
          rx="8"
          fill={node.isHighlighted ? "#3b82f6" : "#e5e7eb"}
          stroke={node.isHighlighted ? "#1e40af" : "#9ca3af"}
          strokeWidth="2"
          className={`transition-all duration-500 ${
            node.isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
          }`}
          style={{
            filter: node.isHighlighted ? 'drop-shadow(0 0 10px #3b82f6)' : 'none'
          }}
        />
      ) : (
        <ellipse
          cx={node.x}
          cy={node.y}
          rx="40"
          ry="25"
          fill={node.value?.includes('A') ? '#10b981' : '#ef4444'}
          stroke={node.value?.includes('A') ? '#059669' : '#dc2626'}
          strokeWidth="2"
          className={`transition-all duration-500 ${
            node.isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
          }`}
          style={{
            filter: node.isHighlighted ? 'drop-shadow(0 0 10px currentColor)' : 'none'
          }}
        />
      )}
      
      {/* Node text */}
      <text
        x={node.x}
        y={node.y + 5}
        textAnchor="middle"
        className={`text-sm font-medium transition-all duration-500 ${
          node.isVisible ? 'opacity-100' : 'opacity-0'
        } ${node.question ? (node.isHighlighted ? 'fill-white' : 'fill-gray-700') : 'fill-white'}`}
      >
        {node.question || node.value}
      </text>
      
      {/* Information gain display */}
      {node.isHighlighted && currentSplit === node.id && (
        <text
          x={node.x}
          y={node.y - 35}
          textAnchor="middle"
          className="text-xs font-bold fill-red-600 animate-bounce"
        >
          IG: {informationGain.toFixed(2)}
        </text>
      )}
      
      {/* Connections */}
      {node.left && (
        <>
          <line
            x1={node.x - 25}
            y1={node.y + 20}
            x2={node.left.x}
            y2={node.left.y - 20}
            stroke="#6b7280"
            strokeWidth="2"
            className={`transition-all duration-500 ${
              node.left.isVisible ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <text
            x={(node.x - 25 + node.left.x) / 2 - 15}
            y={(node.y + 20 + node.left.y - 20) / 2}
            className="text-xs fill-green-600 font-medium"
          >
            Yes
          </text>
          {renderNode(node.left)}
        </>
      )}
      
      {node.right && (
        <>
          <line
            x1={node.x + 25}
            y1={node.y + 20}
            x2={node.right.x}
            y2={node.right.y - 20}
            stroke="#6b7280"
            strokeWidth="2"
            className={`transition-all duration-500 ${
              node.right.isVisible ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <text
            x={(node.x + 25 + node.right.x) / 2 + 15}
            y={(node.y + 20 + node.right.y - 20) / 2}
            className="text-xs fill-red-600 font-medium"
          >
            No
          </text>
          {renderNode(node.right)}
        </>
      )}
    </g>
  );

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-semibold">Interactive Decision Tree Building</h4>
        <div className="flex space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg ${isPlaying ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}
            disabled={step < 2}
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
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data visualization */}
        <div>
          <h5 className="text-sm font-medium mb-2">Training Data</h5>
          <svg width="300" height="250" className="border border-gray-300 rounded">
            {/* Background regions */}
            {currentSplit === 'root' && (
              <>
                <rect x="0" y="0" width="150" height="250" fill="#3b82f6" opacity="0.1" />
                <rect x="150" y="0" width="150" height="250" fill="#ef4444" opacity="0.1" />
              </>
            )}
            
            {/* Data points */}
            {dataPoints.map((point, index) => (
              <circle
                key={point.id}
                cx={point.x}
                cy={point.y}
                r="6"
                fill={point.class === 'A' ? '#10b981' : '#ef4444'}
                stroke="#fff"
                strokeWidth="2"
                className={`transition-all duration-500 ${
                  isActive && step >= 1 ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
                }`}
                style={{ 
                  transitionDelay: `${index * 50}ms`,
                  filter: isPlaying ? 'drop-shadow(0 0 6px currentColor)' : 'none'
                }}
              />
            ))}
            
            {/* Split lines */}
            {tree.splitLine && tree.isVisible && currentSplit === 'root' && (
              <line
                x1={tree.splitLine.x1}
                y1={tree.splitLine.y1}
                x2={tree.splitLine.x2}
                y2={tree.splitLine.y2}
                stroke="#ff6b6b"
                strokeWidth="3"
                strokeDasharray="5,5"
                className="animate-pulse"
              />
            )}
          </svg>
        </div>
        
        {/* Tree visualization */}
        <div>
          <h5 className="text-sm font-medium mb-2">Decision Tree</h5>
          <svg width="400" height="250">
            {renderNode(tree)}
          </svg>
        </div>
      </div>
      
      {/* Information panel */}
      <div className="mt-4 space-y-3">
        {currentSplit && (
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Info className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-800">Current Split Analysis</span>
            </div>
            <div className="text-xs text-blue-700">
              <div>Evaluating split: {tree.question}</div>
              <div>Information Gain: {informationGain.toFixed(3)}</div>
              <div>This split separates the classes effectively!</div>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-3 gap-3 text-center text-xs">
          <div className="bg-green-50 p-2 rounded">
            <div className="font-medium text-green-800">Class A</div>
            <div className="text-green-600">{dataPoints.filter(p => p.class === 'A').length} points</div>
          </div>
          <div className="bg-red-50 p-2 rounded">
            <div className="font-medium text-red-800">Class B</div>
            <div className="text-red-600">{dataPoints.filter(p => p.class === 'B').length} points</div>
          </div>
          <div className="bg-gray-50 p-2 rounded">
            <div className="font-medium text-gray-800">Total</div>
            <div className="text-gray-600">{dataPoints.length} points</div>
          </div>
        </div>
        
        <div className="text-xs text-gray-600 text-center">
          🌳 Watch how the algorithm finds the best splits to separate the classes!
        </div>
      </div>
    </div>
  );
};