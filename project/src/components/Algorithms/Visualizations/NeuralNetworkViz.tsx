import React, { useState, useEffect } from 'react';

interface Neuron {
  id: string;
  x: number;
  y: number;
  layer: number;
  activation: number;
  isActive: boolean;
}

interface Connection {
  from: string;
  to: string;
  weight: number;
  isActive: boolean;
}

interface NeuralNetworkVizProps {
  isActive: boolean;
  step: number;
}

export const NeuralNetworkViz: React.FC<NeuralNetworkVizProps> = ({ isActive, step }) => {
  const [neurons] = useState<Neuron[]>([
    // Input layer
    { id: 'i1', x: 50, y: 60, layer: 0, activation: 0.8, isActive: false },
    { id: 'i2', x: 50, y: 120, layer: 0, activation: 0.6, isActive: false },
    { id: 'i3', x: 50, y: 180, layer: 0, activation: 0.9, isActive: false },
    
    // Hidden layer 1
    { id: 'h1', x: 150, y: 40, layer: 1, activation: 0, isActive: false },
    { id: 'h2', x: 150, y: 90, layer: 1, activation: 0, isActive: false },
    { id: 'h3', x: 150, y: 140, layer: 1, activation: 0, isActive: false },
    { id: 'h4', x: 150, y: 190, layer: 1, activation: 0, isActive: false },
    
    // Hidden layer 2
    { id: 'h5', x: 250, y: 70, layer: 2, activation: 0, isActive: false },
    { id: 'h6', x: 250, y: 130, layer: 2, activation: 0, isActive: false },
    { id: 'h7', x: 250, y: 190, layer: 2, activation: 0, isActive: false },
    
    // Output layer
    { id: 'o1', x: 350, y: 100, layer: 3, activation: 0, isActive: false },
    { id: 'o2', x: 350, y: 160, layer: 3, activation: 0, isActive: false }
  ]);

  const [connections] = useState<Connection[]>([
    // Input to hidden layer 1
    { from: 'i1', to: 'h1', weight: 0.5, isActive: false },
    { from: 'i1', to: 'h2', weight: -0.3, isActive: false },
    { from: 'i1', to: 'h3', weight: 0.8, isActive: false },
    { from: 'i1', to: 'h4', weight: 0.2, isActive: false },
    { from: 'i2', to: 'h1', weight: 0.7, isActive: false },
    { from: 'i2', to: 'h2', weight: 0.4, isActive: false },
    { from: 'i2', to: 'h3', weight: -0.6, isActive: false },
    { from: 'i2', to: 'h4', weight: 0.9, isActive: false },
    { from: 'i3', to: 'h1', weight: -0.2, isActive: false },
    { from: 'i3', to: 'h2', weight: 0.6, isActive: false },
    { from: 'i3', to: 'h3', weight: 0.3, isActive: false },
    { from: 'i3', to: 'h4', weight: -0.4, isActive: false },
    
    // Hidden layer 1 to hidden layer 2
    { from: 'h1', to: 'h5', weight: 0.3, isActive: false },
    { from: 'h1', to: 'h6', weight: 0.7, isActive: false },
    { from: 'h1', to: 'h7', weight: -0.2, isActive: false },
    { from: 'h2', to: 'h5', weight: -0.5, isActive: false },
    { from: 'h2', to: 'h6', weight: 0.4, isActive: false },
    { from: 'h2', to: 'h7', weight: 0.8, isActive: false },
    { from: 'h3', to: 'h5', weight: 0.6, isActive: false },
    { from: 'h3', to: 'h6', weight: -0.3, isActive: false },
    { from: 'h3', to: 'h7', weight: 0.5, isActive: false },
    { from: 'h4', to: 'h5', weight: 0.2, isActive: false },
    { from: 'h4', to: 'h6', weight: 0.9, isActive: false },
    { from: 'h4', to: 'h7', weight: -0.7, isActive: false },
    
    // Hidden layer 2 to output
    { from: 'h5', to: 'o1', weight: 0.8, isActive: false },
    { from: 'h5', to: 'o2', weight: -0.4, isActive: false },
    { from: 'h6', to: 'o1', weight: 0.3, isActive: false },
    { from: 'h6', to: 'o2', weight: 0.7, isActive: false },
    { from: 'h7', to: 'o1', weight: -0.6, isActive: false },
    { from: 'h7', to: 'o2', weight: 0.5, isActive: false }
  ]);

  const [activeNeurons, setActiveNeurons] = useState<Set<string>>(new Set());
  const [activeConnections, setActiveConnections] = useState<Set<string>>(new Set());
  const [currentLayer, setCurrentLayer] = useState(0);

  useEffect(() => {
    if (isActive && step >= 3) {
      const interval = setInterval(() => {
        setCurrentLayer(prev => {
          const nextLayer = (prev + 1) % 4;
          
          // Activate neurons in current layer
          const layerNeurons = neurons.filter(n => n.layer === nextLayer);
          setActiveNeurons(new Set(layerNeurons.map(n => n.id)));
          
          // Activate connections to next layer
          if (nextLayer < 3) {
            const layerConnections = connections.filter(c => {
              const fromNeuron = neurons.find(n => n.id === c.from);
              return fromNeuron?.layer === nextLayer;
            });
            setActiveConnections(new Set(layerConnections.map(c => `${c.from}-${c.to}`)));
          }
          
          return nextLayer;
        });
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isActive, step, neurons, connections]);

  const getNeuron = (id: string) => neurons.find(n => n.id === id);

  const getConnectionColor = (weight: number) => {
    return weight > 0 ? '#10b981' : '#ef4444';
  };

  const getConnectionWidth = (weight: number) => {
    return Math.abs(weight) * 3 + 1;
  };

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <h4 className="text-lg font-semibold mb-4 text-center">Neural Network Visualization</h4>
      
      <svg width="420" height="250" className="mx-auto">
        {/* Connections */}
        {connections.map(conn => {
          const fromNeuron = getNeuron(conn.from);
          const toNeuron = getNeuron(conn.to);
          if (!fromNeuron || !toNeuron) return null;
          
          const isActive = activeConnections.has(`${conn.from}-${conn.to}`);
          
          return (
            <line
              key={`${conn.from}-${conn.to}`}
              x1={fromNeuron.x}
              y1={fromNeuron.y}
              x2={toNeuron.x}
              y2={toNeuron.y}
              stroke={getConnectionColor(conn.weight)}
              strokeWidth={getConnectionWidth(conn.weight)}
              opacity={isActive ? 0.8 : 0.3}
              className={`transition-all duration-300 ${
                isActive ? 'animate-pulse' : ''
              }`}
            />
          );
        })}
        
        {/* Neurons */}
        {neurons.map(neuron => {
          const isActive = activeNeurons.has(neuron.id);
          const layerColors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981'];
          
          return (
            <g key={neuron.id}>
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r="12"
                fill={layerColors[neuron.layer]}
                stroke="#fff"
                strokeWidth="2"
                opacity={isActive ? 1 : 0.6}
                className={`transition-all duration-300 ${
                  isActive ? 'animate-pulse' : ''
                }`}
              />
              <text
                x={neuron.x}
                y={neuron.y + 3}
                textAnchor="middle"
                className="text-xs font-bold fill-white"
              >
                {neuron.activation.toFixed(1)}
              </text>
            </g>
          );
        })}
        
        {/* Layer labels */}
        <text x="50" y="20" textAnchor="middle" className="text-sm font-semibold fill-gray-700">Input</text>
        <text x="150" y="20" textAnchor="middle" className="text-sm font-semibold fill-gray-700">Hidden 1</text>
        <text x="250" y="20" textAnchor="middle" className="text-sm font-semibold fill-gray-700">Hidden 2</text>
        <text x="350" y="20" textAnchor="middle" className="text-sm font-semibold fill-gray-700">Output</text>
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm mb-2">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>Input Layer</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
            <span>Hidden Layers</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
            <span>Output Layer</span>
          </div>
        </div>
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-1 bg-green-500 mr-2"></div>
            <span>Positive Weight</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-1 bg-red-500 mr-2"></div>
            <span>Negative Weight</span>
          </div>
        </div>
        {isActive && step >= 3 && (
          <div className="mt-2 text-xs text-gray-600">
            Forward propagation: Layer {currentLayer + 1} active
          </div>
        )}
      </div>
    </div>
  );
};