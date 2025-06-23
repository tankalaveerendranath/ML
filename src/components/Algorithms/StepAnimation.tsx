import React, { useEffect, useState } from 'react';
import { AlgorithmStep } from '../../types';
import { LinearRegressionViz } from './Visualizations/LinearRegressionViz';
import { DecisionTreeViz } from './Visualizations/DecisionTreeViz';
import { RandomForestViz } from './Visualizations/RandomForestViz';
import { KMeansViz } from './Visualizations/KMeansViz';
import { NeuralNetworkViz } from './Visualizations/NeuralNetworkViz';
import { SVMViz } from './Visualizations/SVMViz';
import { LinearRegressionInteractive } from './InteractiveAnimations/LinearRegressionInteractive';
import { DecisionTreeInteractive } from './InteractiveAnimations/DecisionTreeInteractive';
import { PseudocodeDisplay } from './PseudocodeDisplay';

interface StepAnimationProps {
  step: AlgorithmStep;
  isActive: boolean;
  algorithmId: string;
}

export const StepAnimation: React.FC<StepAnimationProps> = ({ step, isActive, algorithmId }) => {
  const [shouldAnimate, setShouldAnimate] = useState(false);
  const [showInteractive, setShowInteractive] = useState(false);

  useEffect(() => {
    if (isActive) {
      setShouldAnimate(true);
    }
  }, [isActive]);

  const getAnimationClass = () => {
    if (!shouldAnimate) return 'opacity-0 transform translate-y-4';
    
    switch (step.animation) {
      case 'slideInLeft':
        return 'animate-slide-in-left';
      case 'slideInRight':
        return 'animate-slide-in-right';
      case 'slideInUp':
        return 'animate-slide-in-up';
      case 'slideInDown':
        return 'animate-slide-in-down';
      case 'fadeInUp':
        return 'animate-fade-in-up';
      case 'fadeInDown':
        return 'animate-fade-in-down';
      case 'bounceIn':
        return 'animate-bounce-in';
      case 'zoomIn':
        return 'animate-zoom-in';
      case 'rotateIn':
        return 'animate-rotate-in';
      default:
        return 'animate-fade-in';
    }
  };

  const renderVisualization = () => {
    if (!isActive) return null;

    // Show interactive version for certain algorithms and steps
    if (showInteractive) {
      switch (algorithmId) {
        case 'linear-regression':
          return <LinearRegressionInteractive isActive={isActive} step={step.id} />;
        case 'decision-tree':
          return <DecisionTreeInteractive isActive={isActive} step={step.id} />;
        default:
          break;
      }
    }

    // Default visualizations
    switch (algorithmId) {
      case 'linear-regression':
        return <LinearRegressionViz isActive={isActive} step={step.id} />;
      case 'decision-tree':
        return <DecisionTreeViz isActive={isActive} step={step.id} />;
      case 'random-forest':
        return <RandomForestViz isActive={isActive} step={step.id} />;
      case 'kmeans':
        return <KMeansViz isActive={isActive} step={step.id} />;
      case 'neural-networks':
        return <NeuralNetworkViz isActive={isActive} step={step.id} />;
      case 'svm':
        return <SVMViz isActive={isActive} step={step.id} />;
      default:
        return null;
    }
  };

  const hasInteractiveVersion = ['linear-regression', 'decision-tree'].includes(algorithmId);

  return (
    <div className={`transition-all duration-700 ${getAnimationClass()}`}>
      <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
        isActive 
          ? 'border-blue-300 bg-blue-50 shadow-lg' 
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}>
        <div className="flex items-start space-x-4">
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
            isActive 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-600'
          }`}>
            {step.id}
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <h3 className={`font-semibold transition-colors ${
                isActive ? 'text-blue-900' : 'text-gray-900'
              }`}>
                {step.title}
              </h3>
              {hasInteractiveVersion && isActive && (
                <button
                  onClick={() => setShowInteractive(!showInteractive)}
                  className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                    showInteractive 
                      ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' 
                      : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                  }`}
                >
                  {showInteractive ? '📊 Static View' : '🎮 Interactive'}
                </button>
              )}
            </div>
            <p className={`text-sm transition-colors ${
              isActive ? 'text-blue-700' : 'text-gray-600'
            }`}>
              {step.description}
            </p>
            {step.code && isActive && (
              <div className="mt-3 p-3 bg-gray-900 rounded-lg">
                <code className="text-green-400 text-xs font-mono">
                  {step.code}
                </code>
              </div>
            )}
          </div>
        </div>
        
        {/* Enhanced Visualization Component */}
        {isActive && (
          <div className="mt-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <div className="space-y-4">
                {renderVisualization()}
                
                {/* Step-specific tips */}
                {isActive && (
                  <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-start space-x-2">
                      <div className="text-yellow-600 mt-0.5">💡</div>
                      <div>
                        <h4 className="text-sm font-medium text-yellow-800 mb-1">
                          Step {step.id} Insight
                        </h4>
                        <p className="text-xs text-yellow-700">
                          {getStepInsight(algorithmId, step.id)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              <div>
                <PseudocodeDisplay 
                  algorithmId={algorithmId} 
                  isActive={isActive} 
                  step={step.id} 
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const getStepInsight = (algorithmId: string, stepId: number): string => {
  const insights: Record<string, Record<number, string>> = {
    'linear-regression': {
      1: 'Good initialization is crucial! Random values help avoid getting stuck in poor local minima.',
      2: 'The prediction formula ŷ = θ₀ + θ₁x is the heart of linear regression - it defines our hypothesis.',
      3: 'MSE penalizes larger errors more heavily due to the squared term, making the model sensitive to outliers.',
      4: 'Gradients tell us which direction to move our parameters to reduce the cost function.',
      5: 'Learning rate α controls how big steps we take. Too large and we overshoot, too small and we converge slowly.',
      6: 'Convergence occurs when the cost function stops decreasing significantly between iterations.'
    },
    'decision-tree': {
      1: 'Starting with all data at the root gives us the complete picture before making any decisions.',
      2: 'Information gain measures how much uncertainty we remove with each split - higher is better!',
      3: 'Each split creates two subsets that should be more "pure" (homogeneous) than the parent node.',
      4: 'Child nodes represent the outcomes of our decision - left for "yes", right for "no".',
      5: 'Recursion allows us to build complex decision boundaries by repeating the splitting process.',
      6: 'Leaf nodes contain our final predictions - the most common class in that region.'
    },
    'random-forest': {
      1: 'Bootstrap sampling creates diversity by training each tree on slightly different data.',
      2: 'Random feature selection prevents trees from being too similar and reduces overfitting.',
      3: 'Multiple trees capture different patterns in the data, making the ensemble more robust.',
      4: 'Each tree votes independently, bringing different "perspectives" to the prediction.',
      5: 'Majority voting (classification) or averaging (regression) combines the wisdom of all trees.',
      6: 'The final prediction is more reliable than any single tree due to the ensemble effect.'
    },
    'kmeans': {
      1: 'Choosing k is crucial - too few clusters miss patterns, too many create noise.',
      2: 'Initial centroid placement affects convergence. K-means++ initialization often works better than random.',
      3: 'Distance-based assignment ensures each point belongs to its nearest cluster center.',
      4: 'Moving centroids to cluster centers minimizes within-cluster variance.',
      5: 'Convergence occurs when centroids stop moving significantly between iterations.',
      6: 'Final clusters represent natural groupings discovered in your data.'
    },
    'neural-networks': {
      1: 'Network architecture determines the model\'s capacity to learn complex patterns.',
      2: 'Proper weight initialization prevents vanishing/exploding gradients during training.',
      3: 'Forward propagation transforms input through layers using learned weights and biases.',
      4: 'Loss functions measure how far our predictions are from the true values.',
      5: 'Backpropagation efficiently computes gradients by working backwards through the network.',
      6: 'Weight updates using gradients gradually improve the network\'s performance.'
    },
    'svm': {
      1: 'Feature scaling ensures all dimensions contribute equally to distance calculations.',
      2: 'Kernel choice determines how the algorithm handles non-linear relationships.',
      3: 'Support vectors are the critical points that define the decision boundary.',
      4: 'Maximum margin principle creates the most robust decision boundary possible.',
      5: 'Kernel trick allows linear algorithms to solve non-linear problems efficiently.',
      6: 'The decision function uses support vectors to classify new data points.'
    }
  };

  return insights[algorithmId]?.[stepId] || 'This step is important for understanding how the algorithm works!';
};