import { Algorithm } from '../types';

export const algorithms: Algorithm[] = [
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    category: 'Supervised Learning',
    description: 'A fundamental algorithm for predicting continuous values by finding the best-fitting line through data points using gradient descent optimization.',
    icon: 'TrendingUp',
    complexity: 'Beginner',
    useCase: 'Predicting house prices, sales forecasting, risk assessment, trend analysis',
    pros: ['Simple to understand and implement', 'Fast training and prediction', 'Good baseline model', 'Interpretable results', 'No hyperparameter tuning needed'],
    cons: ['Assumes linear relationship', 'Sensitive to outliers', 'May not capture complex patterns', 'Requires feature scaling'],
    steps: [
      {
        id: 1,
        title: 'Data Preparation & Initialization',
        description: 'Collect and prepare your dataset with features (X) and target values (y). Initialize parameters (slope and intercept) with random values.',
        animation: 'slideInLeft',
        code: 'θ₀ = random(), θ₁ = random(), α = 0.01'
      },
      {
        id: 2,
        title: 'Forward Pass - Make Predictions',
        description: 'For each data point, calculate the predicted value using current parameters: ŷ = θ₀ + θ₁ × x',
        animation: 'fadeInUp',
        code: 'for i in range(m): predictions[i] = θ₀ + θ₁ * X[i]'
      },
      {
        id: 3,
        title: 'Calculate Cost Function',
        description: 'Compute the Mean Squared Error (MSE) between predicted and actual values to measure model performance.',
        animation: 'slideInRight',
        code: 'cost = (1/2m) * Σ(predictions[i] - y[i])²'
      },
      {
        id: 4,
        title: 'Compute Gradients',
        description: 'Calculate the partial derivatives of the cost function with respect to each parameter to determine update direction.',
        animation: 'bounceIn',
        code: '∂J/∂θ₀ = (1/m) * Σ(ŷ - y), ∂J/∂θ₁ = (1/m) * Σ((ŷ - y) * x)'
      },
      {
        id: 5,
        title: 'Update Parameters',
        description: 'Use gradient descent to adjust parameters in the direction that minimizes the cost function.',
        animation: 'rotateIn',
        code: 'θ₀ = θ₀ - α * ∂J/∂θ₀, θ₁ = θ₁ - α * ∂J/∂θ₁'
      },
      {
        id: 6,
        title: 'Iterate Until Convergence',
        description: 'Repeat steps 2-5 until the cost function converges or reaches maximum iterations.',
        animation: 'slideInUp',
        code: 'while |cost_change| > tolerance and iter < max_iter'
      }
    ]
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree',
    category: 'Supervised Learning',
    description: 'A tree-like model that makes decisions by recursively splitting data based on feature values that maximize information gain.',
    icon: 'GitBranch',
    complexity: 'Intermediate',
    useCase: 'Medical diagnosis, customer segmentation, feature selection, rule extraction',
    pros: ['Easy to understand and visualize', 'Requires minimal data preparation', 'Handles both numerical and categorical data', 'Provides feature importance', 'No assumptions about data distribution'],
    cons: ['Prone to overfitting', 'Unstable (small data changes can result in different trees)', 'Biased toward features with many levels', 'Can create overly complex trees'],
    steps: [
      {
        id: 1,
        title: 'Start with Root Node',
        description: 'Begin with all training data at the root node of the tree. This represents the entire dataset before any splits.',
        animation: 'fadeInDown',
        code: 'root = Node(data=training_set, depth=0)'
      },
      {
        id: 2,
        title: 'Choose Best Split',
        description: 'Find the feature and threshold that best separates the data using information gain, Gini impurity, or entropy measures.',
        animation: 'slideInLeft',
        code: 'best_feature, best_threshold = argmax(information_gain(feature, threshold))'
      },
      {
        id: 3,
        title: 'Split the Data',
        description: 'Divide the data into subsets based on the chosen feature and threshold value.',
        animation: 'zoomIn',
        code: 'left_data = data[feature <= threshold], right_data = data[feature > threshold]'
      },
      {
        id: 4,
        title: 'Create Child Nodes',
        description: 'Create left and right child nodes for the split data subsets and assign them to the current node.',
        animation: 'slideInRight',
        code: 'node.left = Node(left_data), node.right = Node(right_data)'
      },
      {
        id: 5,
        title: 'Recursive Splitting',
        description: 'Repeat the splitting process for each child node until stopping criteria are met (max depth, min samples, etc.).',
        animation: 'bounceIn',
        code: 'if not stopping_criteria: recursively_split(child_nodes)'
      },
      {
        id: 6,
        title: 'Assign Class Labels',
        description: 'Assign the majority class (for classification) or average value (for regression) to leaf nodes.',
        animation: 'fadeInUp',
        code: 'leaf.prediction = majority_class(leaf_data) or mean(leaf_data)'
      }
    ]
  },
  {
    id: 'random-forest',
    name: 'Random Forest',
    category: 'Ensemble Learning',
    description: 'An ensemble method that combines multiple decision trees trained on different bootstrap samples with random feature selection to improve accuracy and reduce overfitting.',
    icon: 'Trees',
    complexity: 'Intermediate',
    useCase: 'Image classification, bioinformatics, stock market analysis, feature importance ranking',
    pros: ['Reduces overfitting compared to single trees', 'Handles missing values well', 'Provides feature importance', 'Works well with default parameters', 'Robust to outliers'],
    cons: ['Less interpretable than single trees', 'Can overfit with very noisy data', 'Memory intensive', 'Slower prediction than single trees'],
    steps: [
      {
        id: 1,
        title: 'Bootstrap Sampling',
        description: 'Create multiple bootstrap samples from the original training dataset by sampling with replacement.',
        animation: 'slideInDown',
        code: 'for i in n_trees: bootstrap_sample[i] = random_sample(data, size=len(data), replace=True)'
      },
      {
        id: 2,
        title: 'Feature Randomization',
        description: 'For each tree, randomly select a subset of features at each split to increase diversity among trees.',
        animation: 'rotateIn',
        code: 'max_features = sqrt(total_features) for classification, total_features/3 for regression'
      },
      {
        id: 3,
        title: 'Build Decision Trees',
        description: 'Train individual decision trees using different bootstrap samples and random feature subsets.',
        animation: 'zoomIn',
        code: 'for i in n_trees: tree[i] = DecisionTree(bootstrap_sample[i], random_features)'
      },
      {
        id: 4,
        title: 'Make Individual Predictions',
        description: 'Each tree in the forest makes predictions independently on new data points.',
        animation: 'slideInLeft',
        code: 'for tree in forest: predictions.append(tree.predict(new_data))'
      },
      {
        id: 5,
        title: 'Aggregate Results',
        description: 'Combine predictions from all trees using majority voting for classification or averaging for regression.',
        animation: 'bounceIn',
        code: 'final_prediction = majority_vote(predictions) or mean(predictions)'
      },
      {
        id: 6,
        title: 'Output Final Prediction',
        description: 'Return the final prediction based on the aggregated results from all trees in the forest.',
        animation: 'fadeInUp',
        code: 'return final_prediction, feature_importance, out_of_bag_score'
      }
    ]
  },
  {
    id: 'kmeans',
    name: 'K-Means Clustering',
    category: 'Unsupervised Learning',
    description: 'Groups data into k clusters by iteratively minimizing the distance between points and their assigned cluster centers (centroids).',
    icon: 'Circle',
    complexity: 'Beginner',
    useCase: 'Customer segmentation, image segmentation, market research, data compression',
    pros: ['Simple and fast algorithm', 'Works well with spherical clusters', 'Scalable to large datasets', 'Guaranteed convergence', 'Easy to implement'],
    cons: ['Requires predefined number of clusters (k)', 'Sensitive to initialization', 'Assumes spherical clusters', 'Sensitive to outliers', 'Struggles with varying cluster sizes'],
    steps: [
      {
        id: 1,
        title: 'Choose Number of Clusters (k)',
        description: 'Decide on the number of clusters (k) you want to create. This can be determined using methods like the elbow method or silhouette analysis.',
        animation: 'fadeInDown',
        code: 'k = 3  # or use elbow_method(data) to find optimal k'
      },
      {
        id: 2,
        title: 'Initialize Centroids',
        description: 'Randomly place k centroids in the feature space or use smart initialization like K-means++.',
        animation: 'zoomIn',
        code: 'centroids = random_initialize(k, data_bounds) or kmeans_plus_plus(data, k)'
      },
      {
        id: 3,
        title: 'Assign Points to Clusters',
        description: 'Assign each data point to the nearest centroid based on Euclidean distance.',
        animation: 'slideInLeft',
        code: 'for point in data: cluster[point] = argmin(distance(point, centroid))'
      },
      {
        id: 4,
        title: 'Update Centroids',
        description: 'Move each centroid to the center (mean) of all points assigned to its cluster.',
        animation: 'bounceIn',
        code: 'for i in k: centroids[i] = mean(points_in_cluster[i])'
      },
      {
        id: 5,
        title: 'Check for Convergence',
        description: 'Check if centroids have stopped moving significantly or maximum iterations have been reached.',
        animation: 'rotateIn',
        code: 'if sum(distance(old_centroids, new_centroids)) < tolerance: break'
      },
      {
        id: 6,
        title: 'Output Final Clusters',
        description: 'Return the final cluster assignments and centroid positions.',
        animation: 'fadeInUp',
        code: 'return cluster_assignments, final_centroids, inertia'
      }
    ]
  },
  {
    id: 'neural-networks',
    name: 'Neural Networks',
    category: 'Deep Learning',
    description: 'Networks of interconnected nodes (neurons) organized in layers that learn complex patterns through forward propagation and backpropagation.',
    icon: 'Brain',
    complexity: 'Advanced',
    useCase: 'Image recognition, natural language processing, speech recognition, game playing',
    pros: ['Can learn complex non-linear patterns', 'Versatile for many problem types', 'State-of-the-art performance in many domains', 'Automatic feature learning', 'Scalable with more data'],
    cons: ['Requires large amounts of data', 'Computationally expensive', 'Black box (difficult to interpret)', 'Many hyperparameters to tune', 'Prone to overfitting'],
    steps: [
      {
        id: 1,
        title: 'Define Network Architecture',
        description: 'Design the network structure: input layer size, number of hidden layers, neurons per layer, and output layer size.',
        animation: 'slideInDown',
        code: 'network = [input_size, hidden1_size, hidden2_size, output_size]'
      },
      {
        id: 2,
        title: 'Initialize Weights and Biases',
        description: 'Randomly initialize weights and biases for all connections between neurons using techniques like Xavier or He initialization.',
        animation: 'zoomIn',
        code: 'W = random_normal(0, sqrt(2/n_inputs)), b = zeros(n_neurons)'
      },
      {
        id: 3,
        title: 'Forward Propagation',
        description: 'Pass input data through the network, calculating weighted sums and applying activation functions at each layer.',
        animation: 'slideInRight',
        code: 'for layer in network: z = W*a + b, a = activation_function(z)'
      },
      {
        id: 4,
        title: 'Calculate Loss',
        description: 'Compare predicted outputs with actual targets using a loss function (MSE for regression, cross-entropy for classification).',
        animation: 'bounceIn',
        code: 'loss = cross_entropy(predictions, targets) or mse(predictions, targets)'
      },
      {
        id: 5,
        title: 'Backpropagation',
        description: 'Calculate gradients by propagating errors backward through the network using the chain rule.',
        animation: 'slideInLeft',
        code: 'for layer in reverse(network): δ = δ_next * W.T * activation_derivative(z)'
      },
      {
        id: 6,
        title: 'Update Weights and Biases',
        description: 'Adjust weights and biases using gradient descent or advanced optimizers like Adam to minimize the loss.',
        animation: 'fadeInUp',
        code: 'W = W - learning_rate * ∇W, b = b - learning_rate * ∇b'
      }
    ]
  },
  {
    id: 'svm',
    name: 'Support Vector Machine',
    category: 'Supervised Learning',
    description: 'Finds the optimal hyperplane that separates classes with maximum margin by identifying support vectors and using kernel tricks for non-linear data.',
    icon: 'Separator',
    complexity: 'Advanced',
    useCase: 'Text classification, image classification, bioinformatics, high-dimensional data',
    pros: ['Effective in high-dimensional spaces', 'Memory efficient (uses support vectors only)', 'Versatile with different kernel functions', 'Works well with small datasets', 'Strong theoretical foundation'],
    cons: ['Poor performance on large datasets', 'Sensitive to feature scaling', 'No probabilistic output', 'Choice of kernel and parameters is crucial', 'Training time scales poorly'],
    steps: [
      {
        id: 1,
        title: 'Data Preparation and Scaling',
        description: 'Prepare and scale your feature data for optimal SVM performance. Feature scaling is crucial for SVM.',
        animation: 'fadeInDown',
        code: 'X_scaled = StandardScaler().fit_transform(X)'
      },
      {
        id: 2,
        title: 'Choose Kernel Function',
        description: 'Select appropriate kernel function (linear, polynomial, RBF, sigmoid) based on data complexity and dimensionality.',
        animation: 'slideInLeft',
        code: 'kernel = "rbf" or "linear" or "poly", C = regularization_parameter'
      },
      {
        id: 3,
        title: 'Identify Support Vectors',
        description: 'Find data points closest to the decision boundary that will determine the optimal hyperplane.',
        animation: 'zoomIn',
        code: 'support_vectors = points where 0 < α_i < C (on margin boundary)'
      },
      {
        id: 4,
        title: 'Optimize Hyperplane',
        description: 'Find the hyperplane that maximizes the margin between classes using quadratic programming or SMO algorithm.',
        animation: 'rotateIn',
        code: 'maximize: Σα_i - 0.5*ΣΣα_i*α_j*y_i*y_j*K(x_i,x_j) subject to constraints'
      },
      {
        id: 5,
        title: 'Apply Kernel Trick',
        description: 'Use kernel functions to handle non-linearly separable data by mapping to higher-dimensional space.',
        animation: 'bounceIn',
        code: 'K(x_i, x_j) = φ(x_i)·φ(x_j) where φ maps to higher dimension'
      },
      {
        id: 6,
        title: 'Make Predictions',
        description: 'Classify new data points based on their position relative to the decision hyperplane.',
        animation: 'fadeInUp',
        code: 'prediction = sign(Σα_i*y_i*K(x_i, x_new) + b)'
      }
    ]
  }
];