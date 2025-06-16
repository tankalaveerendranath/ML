import React, { useState, useEffect } from 'react';

interface TreeNode {
  id: string;
  question?: string;
  value?: string;
  left?: TreeNode;
  right?: TreeNode;
  x: number;
  y: number;
  isVisible: boolean;
}

interface DecisionTreeVizProps {
  isActive: boolean;
  step: number;
}

export const DecisionTreeViz: React.FC<DecisionTreeVizProps> = ({ isActive, step }) => {
  const [tree, setTree] = useState<TreeNode>({
    id: 'root',
    question: 'Age > 30?',
    x: 200,
    y: 50,
    isVisible: false,
    left: {
      id: 'left1',
      question: 'Income > 50k?',
      x: 100,
      y: 120,
      isVisible: false,
      left: {
        id: 'left2',
        value: 'No Loan',
        x: 50,
        y: 190,
        isVisible: false
      },
      right: {
        id: 'right2',
        value: 'Loan',
        x: 150,
        y: 190,
        isVisible: false
      }
    },
    right: {
      id: 'right1',
      question: 'Credit > 700?',
      x: 300,
      y: 120,
      isVisible: false,
      left: {
        id: 'left3',
        value: 'No Loan',
        x: 250,
        y: 190,
        isVisible: false
      },
      right: {
        id: 'right3',
        value: 'Loan',
        x: 350,
        y: 190,
        isVisible: false
      }
    }
  });

  const [currentPath, setCurrentPath] = useState<string[]>([]);

  useEffect(() => {
    if (isActive) {
      const showNodes = async () => {
        const updateNodeVisibility = (node: TreeNode, nodeId: string, visible: boolean): TreeNode => {
          if (node.id === nodeId) {
            return { ...node, isVisible: visible };
          }
          return {
            ...node,
            left: node.left ? updateNodeVisibility(node.left, nodeId, visible) : undefined,
            right: node.right ? updateNodeVisibility(node.right, nodeId, visible) : undefined
          };
        };

        if (step >= 1) {
          setTree(prev => updateNodeVisibility(prev, 'root', true));
        }
        if (step >= 3) {
          setTimeout(() => {
            setTree(prev => updateNodeVisibility(prev, 'left1', true));
            setTree(prev => updateNodeVisibility(prev, 'right1', true));
          }, 500);
        }
        if (step >= 5) {
          setTimeout(() => {
            setTree(prev => updateNodeVisibility(prev, 'left2', true));
            setTree(prev => updateNodeVisibility(prev, 'right2', true));
            setTree(prev => updateNodeVisibility(prev, 'left3', true));
            setTree(prev => updateNodeVisibility(prev, 'right3', true));
          }, 1000);
        }
      };

      showNodes();
    }
  }, [isActive, step]);

  const renderNode = (node: TreeNode): JSX.Element => (
    <g key={node.id}>
      {/* Node circle/rectangle */}
      {node.question ? (
        <rect
          x={node.x - 40}
          y={node.y - 15}
          width="80"
          height="30"
          rx="5"
          fill="#3b82f6"
          stroke="#1e40af"
          strokeWidth="2"
          className={`transition-all duration-500 ${
            node.isVisible ? 'opacity-100' : 'opacity-0'
          }`}
        />
      ) : (
        <ellipse
          cx={node.x}
          cy={node.y}
          rx="35"
          ry="20"
          fill={node.value === 'Loan' ? '#10b981' : '#ef4444'}
          stroke={node.value === 'Loan' ? '#059669' : '#dc2626'}
          strokeWidth="2"
          className={`transition-all duration-500 ${
            node.isVisible ? 'opacity-100' : 'opacity-0'
          }`}
        />
      )}
      
      {/* Node text */}
      <text
        x={node.x}
        y={node.y + 4}
        textAnchor="middle"
        className={`text-xs font-medium transition-all duration-500 ${
          node.isVisible ? 'opacity-100' : 'opacity-0'
        } ${node.question ? 'fill-white' : 'fill-white'}`}
      >
        {node.question || node.value}
      </text>
      
      {/* Connections to children */}
      {node.left && (
        <>
          <line
            x1={node.x - 20}
            y1={node.y + 15}
            x2={node.left.x}
            y2={node.left.y - 15}
            stroke="#6b7280"
            strokeWidth="2"
            className={`transition-all duration-500 ${
              node.left.isVisible ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <text
            x={(node.x - 20 + node.left.x) / 2 - 10}
            y={(node.y + 15 + node.left.y - 15) / 2}
            className="text-xs fill-gray-600"
          >
            Yes
          </text>
          {renderNode(node.left)}
        </>
      )}
      
      {node.right && (
        <>
          <line
            x1={node.x + 20}
            y1={node.y + 15}
            x2={node.right.x}
            y2={node.right.y - 15}
            stroke="#6b7280"
            strokeWidth="2"
            className={`transition-all duration-500 ${
              node.right.isVisible ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <text
            x={(node.x + 20 + node.right.x) / 2 + 10}
            y={(node.y + 15 + node.right.y - 15) / 2}
            className="text-xs fill-gray-600"
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
      <h4 className="text-lg font-semibold mb-4 text-center">Decision Tree Visualization</h4>
      
      <svg width="400" height="250" className="mx-auto">
        {renderNode(tree)}
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
            <span>Decision Node</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
            <span>Positive Outcome</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
            <span>Negative Outcome</span>
          </div>
        </div>
        {isActive && step >= 2 && (
          <div className="mt-2 text-xs text-gray-600">
            Building tree by finding best splits based on information gain
          </div>
        )}
      </div>
    </div>
  );
};