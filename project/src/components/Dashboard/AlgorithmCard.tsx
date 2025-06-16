import React from 'react';
import { TrendingUp, GitBranch, Trees, Circle, Brain, SeparatorVertical as Separator, ArrowRight } from 'lucide-react';
import { Algorithm } from '../../types';

const iconMap = {
  TrendingUp,
  GitBranch,
  Trees,
  Circle,
  Brain,
  Separator
};

interface AlgorithmCardProps {
  algorithm: Algorithm;
  onClick: () => void;
}

export const AlgorithmCard: React.FC<AlgorithmCardProps> = ({ algorithm, onClick }) => {
  const IconComponent = iconMap[algorithm.icon as keyof typeof iconMap];
  
  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Beginner': return 'bg-green-100 text-green-800';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'Advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Supervised Learning': return 'bg-blue-100 text-blue-800';
      case 'Unsupervised Learning': return 'bg-purple-100 text-purple-800';
      case 'Ensemble Learning': return 'bg-emerald-100 text-emerald-800';
      case 'Deep Learning': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div 
      className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer transform hover:-translate-y-1 border border-gray-100"
      onClick={onClick}
    >
      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-blue-50 rounded-lg">
              <IconComponent className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{algorithm.name}</h3>
              <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(algorithm.category)}`}>
                {algorithm.category}
              </span>
            </div>
          </div>
          <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(algorithm.complexity)}`}>
            {algorithm.complexity}
          </span>
        </div>
        
        <p className="text-gray-600 text-sm mb-4 line-clamp-3">
          {algorithm.description}
        </p>
        
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Common Use Cases:</h4>
          <p className="text-xs text-gray-600">{algorithm.useCase}</p>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-500">
            {algorithm.steps.length} steps to learn
          </div>
          <ArrowRight className="w-4 h-4 text-blue-600 group-hover:translate-x-1 transition-transform" />
        </div>
      </div>
    </div>
  );
};