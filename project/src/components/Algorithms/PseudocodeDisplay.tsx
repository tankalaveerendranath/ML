import React from 'react';
import { Code } from 'lucide-react';

interface PseudocodeDisplayProps {
  algorithmId: string;
  isActive: boolean;
  step: number;
}

export const PseudocodeDisplay: React.FC<PseudocodeDisplayProps> = ({ algorithmId, isActive, step }) => {
  const getPseudocode = () => {
    switch (algorithmId) {
      case 'linear-regression':
        return {
          title: 'Linear Regression Pseudocode',
          code: `ALGORITHM: Linear Regression with Gradient Descent

INPUT: Training data X (features), y (targets), learning_rate α, max_iterations
OUTPUT: Optimized parameters θ₀ (intercept), θ₁ (slope)

1. INITIALIZE:
   θ₀ ← random_value()
   θ₁ ← random_value()
   m ← number_of_samples

2. FOR iteration = 1 to max_iterations:
   
   3. FORWARD PASS:
      FOR i = 1 to m:
         prediction[i] ← θ₀ + θ₁ × X[i]
      END FOR
   
   4. COMPUTE COST:
      cost ← (1/2m) × Σ(prediction[i] - y[i])²
   
   5. COMPUTE GRADIENTS:
      gradient_θ₀ ← (1/m) × Σ(prediction[i] - y[i])
      gradient_θ₁ ← (1/m) × Σ((prediction[i] - y[i]) × X[i])
   
   6. UPDATE PARAMETERS:
      θ₀ ← θ₀ - α × gradient_θ₀
      θ₁ ← θ₁ - α × gradient_θ₁
   
   7. CHECK CONVERGENCE:
      IF |cost_change| < tolerance:
         BREAK
      END IF

END FOR

RETURN θ₀, θ₁`
        };

      case 'decision-tree':
        return {
          title: 'Decision Tree Pseudocode',
          code: `ALGORITHM: Decision Tree Construction (ID3/C4.5)

INPUT: Training dataset D, feature set F, target attribute T
OUTPUT: Decision tree root node

1. FUNCTION BuildTree(D, F, T):
   
   2. IF all samples in D have same class:
      RETURN leaf node with that class
   
   3. IF F is empty:
      RETURN leaf node with majority class in D
   
   4. FIND BEST SPLIT:
      best_feature ← NULL
      best_gain ← -∞
      
      FOR each feature f in F:
         FOR each possible threshold t for f:
            gain ← InformationGain(D, f, t)
            IF gain > best_gain:
               best_gain ← gain
               best_feature ← f
               best_threshold ← t
            END IF
         END FOR
      END FOR
   
   5. CREATE DECISION NODE:
      node ← new DecisionNode(best_feature, best_threshold)
   
   6. SPLIT DATA:
      D_left ← samples where feature ≤ threshold
      D_right ← samples where feature > threshold
   
   7. RECURSIVE CALLS:
      node.left ← BuildTree(D_left, F, T)
      node.right ← BuildTree(D_right, F, T)
   
   RETURN node

8. FUNCTION InformationGain(D, feature, threshold):
   entropy_before ← Entropy(D)
   
   D_left, D_right ← Split(D, feature, threshold)
   weighted_entropy ← (|D_left|/|D|) × Entropy(D_left) + 
                      (|D_right|/|D|) × Entropy(D_right)
   
   RETURN entropy_before - weighted_entropy`
        };

      case 'random-forest':
        return {
          title: 'Random Forest Pseudocode',
          code: `ALGORITHM: Random Forest

INPUT: Training data D, number of trees n_trees, 
       max_features, bootstrap_size
OUTPUT: Ensemble of decision trees

1. INITIALIZE:
   forest ← empty list
   
2. FOR i = 1 to n_trees:
   
   3. BOOTSTRAP SAMPLING:
      D_bootstrap ← RandomSample(D, bootstrap_size, with_replacement=True)
   
   4. FEATURE RANDOMIZATION:
      F_random ← RandomSubset(all_features, max_features)
   
   5. BUILD TREE:
      tree_i ← BuildDecisionTree(D_bootstrap, F_random)
      forest.add(tree_i)

END FOR

6. FUNCTION Predict(x):
   predictions ← empty list
   
   FOR each tree in forest:
      prediction ← tree.predict(x)
      predictions.add(prediction)
   END FOR
   
   IF classification:
      RETURN MajorityVote(predictions)
   ELSE: // regression
      RETURN Average(predictions)
   END IF

7. FUNCTION MajorityVote(predictions):
   vote_counts ← CountVotes(predictions)
   RETURN class_with_max_votes

8. FUNCTION BuildDecisionTree(D, F):
   // Same as standard decision tree but with
   // random feature selection at each split
   RETURN DecisionTree(D, F)`
        };

      case 'kmeans':
        return {
          title: 'K-Means Clustering Pseudocode',
          code: `ALGORITHM: K-Means Clustering

INPUT: Dataset X, number of clusters k, max_iterations, tolerance
OUTPUT: Cluster centroids C, cluster assignments

1. INITIALIZE:
   C ← RandomInitializeCentroids(k, X)
   assignments ← empty array
   
2. FOR iteration = 1 to max_iterations:
   
   3. ASSIGNMENT STEP:
      FOR each point x_i in X:
         min_distance ← ∞
         closest_centroid ← -1
         
         FOR j = 1 to k:
            distance ← EuclideanDistance(x_i, C[j])
            IF distance < min_distance:
               min_distance ← distance
               closest_centroid ← j
            END IF
         END FOR
         
         assignments[i] ← closest_centroid
      END FOR
   
   4. UPDATE STEP:
      new_centroids ← empty array
      
      FOR j = 1 to k:
         cluster_points ← {x_i | assignments[i] = j}
         
         IF cluster_points is not empty:
            new_centroids[j] ← Mean(cluster_points)
         ELSE:
            new_centroids[j] ← C[j]  // Keep old centroid
         END IF
      END FOR
   
   5. CHECK CONVERGENCE:
      centroid_shift ← 0
      FOR j = 1 to k:
         centroid_shift += Distance(C[j], new_centroids[j])
      END FOR
      
      IF centroid_shift < tolerance:
         BREAK
      END IF
      
      C ← new_centroids

END FOR

RETURN C, assignments`
        };

      case 'neural-networks':
        return {
          title: 'Neural Network (Backpropagation) Pseudocode',
          code: `ALGORITHM: Neural Network with Backpropagation

INPUT: Training data X, labels y, network architecture, 
       learning_rate α, epochs
OUTPUT: Trained network weights W, biases b

1. INITIALIZE:
   FOR each layer l:
      W[l] ← RandomWeights(layer_size[l-1], layer_size[l])
      b[l] ← RandomBiases(layer_size[l])
   END FOR

2. FOR epoch = 1 to epochs:
   
   FOR each training sample (x, target):
   
      3. FORWARD PROPAGATION:
         a[0] ← x  // Input layer activation
         
         FOR layer l = 1 to L:
            z[l] ← W[l] × a[l-1] + b[l]
            a[l] ← ActivationFunction(z[l])
         END FOR
         
         output ← a[L]
      
      4. COMPUTE LOSS:
         loss ← LossFunction(output, target)
      
      5. BACKWARD PROPAGATION:
         // Output layer error
         δ[L] ← ∇loss × ActivationDerivative(z[L])
         
         // Hidden layer errors (backpropagate)
         FOR layer l = L-1 down to 1:
            δ[l] ← (W[l+1]ᵀ × δ[l+1]) ⊙ ActivationDerivative(z[l])
         END FOR
      
      6. UPDATE WEIGHTS AND BIASES:
         FOR layer l = 1 to L:
            ∇W[l] ← δ[l] × a[l-1]ᵀ
            ∇b[l] ← δ[l]
            
            W[l] ← W[l] - α × ∇W[l]
            b[l] ← b[l] - α × ∇b[l]
         END FOR
   
   END FOR
END FOR

7. FUNCTION ActivationFunction(z):
   // Common choices: ReLU, Sigmoid, Tanh
   RETURN max(0, z)  // ReLU example

8. FUNCTION LossFunction(predicted, actual):
   // For classification: CrossEntropy
   // For regression: MeanSquaredError
   RETURN 0.5 × (predicted - actual)²`
        };

      case 'svm':
        return {
          title: 'Support Vector Machine Pseudocode',
          code: `ALGORITHM: Support Vector Machine (SMO Algorithm)

INPUT: Training data X, labels y, regularization C, 
       kernel function, tolerance
OUTPUT: Support vectors, weights α, bias b

1. INITIALIZE:
   α ← zeros(n_samples)  // Lagrange multipliers
   b ← 0                 // Bias term
   
2. WHILE not converged:
   
   3. SELECT WORKING SET:
      // Choose two variables αᵢ, αⱼ that violate KKT conditions
      i, j ← SelectWorkingSet(α, X, y)
      
      IF no violating pair found:
         BREAK  // Converged
      END IF
   
   4. COMPUTE BOUNDS:
      IF y[i] ≠ y[j]:
         L ← max(0, α[j] - α[i])
         H ← min(C, C + α[j] - α[i])
      ELSE:
         L ← max(0, α[i] + α[j] - C)
         H ← min(C, α[i] + α[j])
      END IF
   
   5. COMPUTE KERNEL VALUES:
      K_ii ← Kernel(X[i], X[i])
      K_jj ← Kernel(X[j], X[j])
      K_ij ← Kernel(X[i], X[j])
      η ← K_ii + K_jj - 2×K_ij
   
   6. UPDATE αⱼ:
      E_i ← PredictionError(i)
      E_j ← PredictionError(j)
      
      α_j_new ← α[j] + y[j]×(E_i - E_j)/η
      α_j_new ← Clip(α_j_new, L, H)
   
   7. UPDATE αᵢ:
      α_i_new ← α[i] + y[i]×y[j]×(α[j] - α_j_new)
   
   8. UPDATE BIAS:
      IF 0 < α_i_new < C:
         b_new ← E_i + y[i]×(α_i_new - α[i])×K_ii + 
                      y[j]×(α_j_new - α[j])×K_ij + b
      ELSE IF 0 < α_j_new < C:
         b_new ← E_j + y[i]×(α_i_new - α[i])×K_ij + 
                      y[j]×(α_j_new - α[j])×K_jj + b
      ELSE:
         b_new ← (b1 + b2)/2
      END IF
   
   9. UPDATE VALUES:
      α[i] ← α_i_new
      α[j] ← α_j_new
      b ← b_new

END WHILE

10. EXTRACT SUPPORT VECTORS:
    support_vectors ← {i | α[i] > 0}
    
RETURN support_vectors, α, b

11. FUNCTION Predict(x):
    decision_value ← Σ(α[i] × y[i] × Kernel(X[i], x)) + b
    RETURN sign(decision_value)`
        };

      default:
        return { title: 'Pseudocode', code: 'No pseudocode available for this algorithm.' };
    }
  };

  const pseudocode = getPseudocode();
  const lines = pseudocode.code.split('\n');

  const getHighlightedStep = () => {
    switch (algorithmId) {
      case 'linear-regression':
        if (step === 1) return [0, 4]; // Initialize
        if (step === 2) return [6, 8]; // Forward pass
        if (step === 3) return [10, 11]; // Compute cost
        if (step === 4) return [13, 15]; // Compute gradients
        if (step === 5) return [17, 19]; // Update parameters
        if (step === 6) return [21, 25]; // Check convergence
        break;
      case 'decision-tree':
        if (step === 1) return [0, 2]; // Start with root
        if (step === 2) return [8, 20]; // Choose best split
        if (step === 3) return [22, 25]; // Split data
        if (step === 4) return [27, 29]; // Create child nodes
        if (step === 5) return [27, 29]; // Recursive splitting
        if (step === 6) return [2, 5]; // Assign class labels
        break;
      case 'random-forest':
        if (step === 1) return [0, 4]; // Bootstrap sampling
        if (step === 2) return [6, 7]; // Feature randomization
        if (step === 3) return [9, 11]; // Build trees
        if (step === 4) return [15, 22]; // Make predictions
        if (step === 5) return [24, 30]; // Aggregate results
        if (step === 6) return [24, 30]; // Final prediction
        break;
      case 'kmeans':
        if (step === 1) return [0, 2]; // Choose K
        if (step === 2) return [4, 6]; // Initialize centroids
        if (step === 3) return [8, 22]; // Assign points
        if (step === 4) return [24, 34]; // Update centroids
        if (step === 5) return [36, 45]; // Check convergence
        if (step === 6) return [49, 49]; // Final clusters
        break;
      case 'neural-networks':
        if (step === 1) return [0, 5]; // Network architecture
        if (step === 2) return [7, 12]; // Initialize weights
        if (step === 3) return [16, 23]; // Forward propagation
        if (step === 4) return [25, 26]; // Calculate loss
        if (step === 5) return [28, 34]; // Backpropagation
        if (step === 6) return [36, 43]; // Update weights
        break;
      case 'svm':
        if (step === 1) return [0, 3]; // Data preparation
        if (step === 2) return [5, 7]; // Choose kernel
        if (step === 3) return [60, 61]; // Find support vectors
        if (step === 4) return [9, 18]; // Optimize hyperplane
        if (step === 5) return [20, 30]; // Handle non-linearity
        if (step === 6) return [65, 67]; // Make predictions
        break;
    }
    return [0, 0];
  };

  const [highlightStart, highlightEnd] = getHighlightedStep();

  if (!isActive) return null;

  return (
    <div className="bg-gray-900 rounded-lg p-6 border-2 border-gray-700 mt-4">
      <div className="flex items-center mb-4">
        <Code className="w-5 h-5 text-green-400 mr-2" />
        <h4 className="text-lg font-semibold text-white">{pseudocode.title}</h4>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
        <pre className="text-sm font-mono">
          {lines.map((line, index) => (
            <div
              key={index}
              className={`transition-all duration-300 ${
                index >= highlightStart && index <= highlightEnd
                  ? 'bg-blue-900 bg-opacity-50 text-blue-200 border-l-4 border-blue-400 pl-2'
                  : 'text-gray-300'
              }`}
            >
              <span className="text-gray-500 mr-3 select-none">
                {String(index + 1).padStart(2, '0')}
              </span>
              {line}
            </div>
          ))}
        </pre>
      </div>
      
      <div className="mt-4 flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-400 rounded mr-2"></div>
            <span className="text-gray-300">Current Step</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-400 rounded mr-2"></div>
            <span className="text-gray-300">Keywords</span>
          </div>
        </div>
        <div className="text-gray-400">
          Step {step} of 6 highlighted
        </div>
      </div>
    </div>
  );
};