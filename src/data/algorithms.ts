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
    exampleDataset: {
      name: 'House Price Prediction',
      description: 'Predicting house prices based on size and location',
      features: ['House Size (sq ft)', 'Distance to City Center (miles)'],
      data: [
        [1200, 5], [1500, 3], [1800, 2], [2000, 1], [2200, 4],
        [1000, 8], [1300, 6], [1600, 3], [1900, 2], [2100, 1]
      ],
      target: [200000, 280000, 350000, 420000, 380000, 150000, 220000, 300000, 380000, 450000],
      featureNames: ['size', 'distance']
    },
    mathematicalFormulas: [
      {
        name: 'Linear Equation',
        formula: 'y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ',
        description: 'The fundamental linear relationship between features and target',
        variables: {
          'y': 'Predicted value',
          'θ₀': 'Intercept (bias term)',
          'θᵢ': 'Coefficient for feature i',
          'xᵢ': 'Feature value i'
        }
      },
      {
        name: 'Cost Function (MSE)',
        formula: 'J(θ) = (1/2m) × Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²',
        description: 'Mean Squared Error - measures prediction accuracy',
        variables: {
          'J(θ)': 'Cost function',
          'm': 'Number of training examples',
          'hθ(x⁽ⁱ⁾)': 'Predicted value for example i',
          'y⁽ⁱ⁾': 'Actual value for example i'
        }
      },
      {
        name: 'Gradient Descent Update',
        formula: 'θⱼ := θⱼ - α × (∂J(θ)/∂θⱼ)',
        description: 'Parameter update rule for optimization',
        variables: {
          'α': 'Learning rate',
          '∂J(θ)/∂θⱼ': 'Partial derivative of cost with respect to parameter j'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Data Preparation & Initialization',
        description: 'Collect and prepare your dataset with features (X) and target values (y). Initialize parameters (slope and intercept) with random values.',
        animation: 'slideInLeft',
        code: 'θ₀ = random(), θ₁ = random(), α = 0.01',
        mathematicalStep: 'θ₀ = 0.1, θ₁ = 0.05, θ₂ = -0.02',
        exampleCalculation: 'Initial prediction: ŷ = 0.1 + 0.05×1200 + (-0.02)×5 = 59.9'
      },
      {
        id: 2,
        title: 'Forward Pass - Make Predictions',
        description: 'For each data point, calculate the predicted value using current parameters: ŷ = θ₀ + θ₁ × x',
        animation: 'fadeInUp',
        code: 'for i in range(m): predictions[i] = θ₀ + θ₁ * X[i]',
        mathematicalStep: 'ŷ⁽¹⁾ = θ₀ + θ₁×x₁⁽¹⁾ + θ₂×x₂⁽¹⁾',
        exampleCalculation: 'For house 1: ŷ = 0.1 + 0.05×1200 + (-0.02)×5 = 59.9k (actual: 200k)'
      },
      {
        id: 3,
        title: 'Calculate Cost Function',
        description: 'Compute the Mean Squared Error (MSE) between predicted and actual values to measure model performance.',
        animation: 'slideInRight',
        code: 'cost = (1/2m) * Σ(predictions[i] - y[i])²',
        mathematicalStep: 'J(θ) = (1/2×10) × Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²',
        exampleCalculation: 'Error for house 1: (59.9 - 200)² = 19,628.01, Total MSE = 15,234.5'
      },
      {
        id: 4,
        title: 'Compute Gradients',
        description: 'Calculate the partial derivatives of the cost function with respect to each parameter to determine update direction.',
        animation: 'bounceIn',
        code: '∂J/∂θ₀ = (1/m) * Σ(ŷ - y), ∂J/∂θ₁ = (1/m) * Σ((ŷ - y) * x)',
        mathematicalStep: '∂J/∂θ₀ = -140.1, ∂J/∂θ₁ = -210,150, ∂J/∂θ₂ = 560.4',
        exampleCalculation: 'Gradients indicate direction to minimize cost'
      },
      {
        id: 5,
        title: 'Update Parameters',
        description: 'Use gradient descent to adjust parameters in the direction that minimizes the cost function.',
        animation: 'rotateIn',
        code: 'θ₀ = θ₀ - α * ∂J/∂θ₀, θ₁ = θ₁ - α * ∂J/∂θ₁',
        mathematicalStep: 'θ₀ = 0.1 - 0.01×(-140.1) = 1.501',
        exampleCalculation: 'New θ₁ = 0.05 - 0.01×(-210,150) = 2101.55'
      },
      {
        id: 6,
        title: 'Iterate Until Convergence',
        description: 'Repeat steps 2-5 until the cost function converges or reaches maximum iterations.',
        animation: 'slideInUp',
        code: 'while |cost_change| > tolerance and iter < max_iter',
        mathematicalStep: 'Final: θ₀ = 50,000, θ₁ = 150, θ₂ = -8,000',
        exampleCalculation: 'Final model: Price = 50,000 + 150×size - 8,000×distance'
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
    exampleDataset: {
      name: 'Loan Approval Prediction',
      description: 'Predicting loan approval based on applicant characteristics',
      features: ['Age', 'Income ($)', 'Credit Score', 'Employment Years'],
      data: [
        [25, 35000, 650, 2], [35, 55000, 720, 8], [45, 75000, 780, 15],
        [28, 40000, 680, 3], [52, 85000, 800, 20], [30, 45000, 700, 5],
        [38, 60000, 740, 10], [42, 70000, 760, 12], [26, 32000, 620, 1],
        [48, 80000, 790, 18]
      ],
      target: ['Denied', 'Approved', 'Approved', 'Denied', 'Approved', 'Approved', 'Approved', 'Approved', 'Denied', 'Approved'],
      featureNames: ['age', 'income', 'credit_score', 'employment_years']
    },
    mathematicalFormulas: [
      {
        name: 'Information Gain',
        formula: 'IG(S,A) = Entropy(S) - Σ(|Sᵥ|/|S|) × Entropy(Sᵥ)',
        description: 'Measures the reduction in entropy after splitting on attribute A',
        variables: {
          'S': 'Current dataset',
          'A': 'Attribute to split on',
          'Sᵥ': 'Subset of S where attribute A has value v',
          '|S|': 'Size of dataset S'
        }
      },
      {
        name: 'Entropy',
        formula: 'Entropy(S) = -Σ pᵢ × log₂(pᵢ)',
        description: 'Measures the impurity or randomness in the dataset',
        variables: {
          'pᵢ': 'Proportion of examples belonging to class i',
          'log₂': 'Logarithm base 2'
        }
      },
      {
        name: 'Gini Impurity',
        formula: 'Gini(S) = 1 - Σ pᵢ²',
        description: 'Alternative impurity measure (used in CART algorithm)',
        variables: {
          'pᵢ': 'Proportion of examples belonging to class i'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Start with Root Node',
        description: 'Begin with all training data at the root node of the tree. This represents the entire dataset before any splits.',
        animation: 'fadeInDown',
        code: 'root = Node(data=training_set, depth=0)',
        mathematicalStep: 'Initial entropy = -6/10×log₂(6/10) - 4/10×log₂(4/10) = 0.971',
        exampleCalculation: '10 samples: 6 Approved, 4 Denied → High entropy (mixed classes)'
      },
      {
        id: 2,
        title: 'Choose Best Split',
        description: 'Find the feature and threshold that best separates the data using information gain, Gini impurity, or entropy measures.',
        animation: 'slideInLeft',
        code: 'best_feature, best_threshold = argmax(information_gain(feature, threshold))',
        mathematicalStep: 'Test Credit Score ≥ 700: IG = 0.971 - (3/10×0 + 7/10×0.592) = 0.557',
        exampleCalculation: 'Credit Score ≥ 700 gives highest information gain of 0.557'
      },
      {
        id: 3,
        title: 'Split the Data',
        description: 'Divide the data into subsets based on the chosen feature and threshold value.',
        animation: 'zoomIn',
        code: 'left_data = data[feature <= threshold], right_data = data[feature > threshold]',
        mathematicalStep: 'Left: Credit < 700 (3 samples, all Denied)',
        exampleCalculation: 'Right: Credit ≥ 700 (7 samples, 6 Approved, 1 Denied)'
      },
      {
        id: 4,
        title: 'Create Child Nodes',
        description: 'Create left and right child nodes for the split data subsets and assign them to the current node.',
        animation: 'slideInRight',
        code: 'node.left = Node(left_data), node.right = Node(right_data)',
        mathematicalStep: 'Left node: Pure (entropy = 0), Right node: entropy = 0.592',
        exampleCalculation: 'Left becomes leaf (all Denied), Right needs further splitting'
      },
      {
        id: 5,
        title: 'Recursive Splitting',
        description: 'Repeat the splitting process for each child node until stopping criteria are met (max depth, min samples, etc.).',
        animation: 'bounceIn',
        code: 'if not stopping_criteria: recursively_split(child_nodes)',
        mathematicalStep: 'For right node, test Income ≥ 50000: IG = 0.592 - 0.286 = 0.306',
        exampleCalculation: 'Continue splitting until pure nodes or stopping criteria met'
      },
      {
        id: 6,
        title: 'Assign Class Labels',
        description: 'Assign the majority class (for classification) or average value (for regression) to leaf nodes.',
        animation: 'fadeInUp',
        code: 'leaf.prediction = majority_class(leaf_data) or mean(leaf_data)',
        mathematicalStep: 'Final tree: 4 leaf nodes with pure or majority class predictions',
        exampleCalculation: 'Decision rules: If Credit<700→Denied, If Credit≥700 & Income≥50000→Approved'
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
    exampleDataset: {
      name: 'Customer Churn Prediction',
      description: 'Predicting whether customers will cancel their subscription',
      features: ['Monthly Charges', 'Total Charges', 'Contract Length (months)', 'Support Calls'],
      data: [
        [50, 1200, 12, 2], [80, 3200, 24, 1], [120, 7200, 36, 0],
        [45, 900, 6, 5], [95, 4750, 24, 3], [110, 6600, 36, 1],
        [60, 1800, 12, 4], [75, 3000, 24, 2], [40, 800, 6, 6],
        [100, 5000, 36, 1]
      ],
      target: ['Churn', 'Stay', 'Stay', 'Churn', 'Churn', 'Stay', 'Churn', 'Stay', 'Churn', 'Stay'],
      featureNames: ['monthly_charges', 'total_charges', 'contract_length', 'support_calls']
    },
    mathematicalFormulas: [
      {
        name: 'Bootstrap Sampling',
        formula: 'Bᵢ = Sample(D, |D|, with_replacement=True)',
        description: 'Create bootstrap samples by sampling with replacement',
        variables: {
          'Bᵢ': 'Bootstrap sample i',
          'D': 'Original dataset',
          '|D|': 'Size of original dataset'
        }
      },
      {
        name: 'Random Feature Selection',
        formula: 'Fᵢ = RandomSubset(F, √|F|) for classification',
        description: 'Randomly select subset of features at each split',
        variables: {
          'Fᵢ': 'Feature subset for split i',
          'F': 'All available features',
          '√|F|': 'Square root of total features (typical choice)'
        }
      },
      {
        name: 'Ensemble Prediction',
        formula: 'ŷ = MajorityVote(T₁(x), T₂(x), ..., Tₙ(x))',
        description: 'Combine predictions from all trees',
        variables: {
          'Tᵢ(x)': 'Prediction from tree i',
          'n': 'Number of trees in forest'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Bootstrap Sampling',
        description: 'Create multiple bootstrap samples from the original training dataset by sampling with replacement.',
        animation: 'slideInDown',
        code: 'for i in n_trees: bootstrap_sample[i] = random_sample(data, size=len(data), replace=True)',
        mathematicalStep: 'Create 5 bootstrap samples, each with 10 examples (with replacement)',
        exampleCalculation: 'Sample 1: [1,3,3,7,9,2,5,8,1,6], Sample 2: [2,4,6,8,9,1,3,5,7,10]...'
      },
      {
        id: 2,
        title: 'Feature Randomization',
        description: 'For each tree, randomly select a subset of features at each split to increase diversity among trees.',
        animation: 'rotateIn',
        code: 'max_features = sqrt(total_features) for classification, total_features/3 for regression',
        mathematicalStep: 'With 4 features, select √4 = 2 features randomly at each split',
        exampleCalculation: 'Split 1: [monthly_charges, support_calls], Split 2: [contract_length, total_charges]'
      },
      {
        id: 3,
        title: 'Build Decision Trees',
        description: 'Train individual decision trees using different bootstrap samples and random feature subsets.',
        animation: 'zoomIn',
        code: 'for i in n_trees: tree[i] = DecisionTree(bootstrap_sample[i], random_features)',
        mathematicalStep: 'Tree 1: Root split on support_calls ≥ 3, Tree 2: Root split on monthly_charges ≥ 70',
        exampleCalculation: 'Each tree learns different patterns due to different data and features'
      },
      {
        id: 4,
        title: 'Make Individual Predictions',
        description: 'Each tree in the forest makes predictions independently on new data points.',
        animation: 'slideInLeft',
        code: 'for tree in forest: predictions.append(tree.predict(new_data))',
        mathematicalStep: 'New customer: [65, 2600, 18, 3] → Tree predictions: [Churn, Stay, Churn, Churn, Stay]',
        exampleCalculation: 'Tree 1: Churn, Tree 2: Stay, Tree 3: Churn, Tree 4: Churn, Tree 5: Stay'
      },
      {
        id: 5,
        title: 'Aggregate Results',
        description: 'Combine predictions from all trees using majority voting for classification or averaging for regression.',
        animation: 'bounceIn',
        code: 'final_prediction = majority_vote(predictions) or mean(predictions)',
        mathematicalStep: 'Votes: Churn=3, Stay=2 → Majority vote = Churn',
        exampleCalculation: 'Final prediction: Churn (60% confidence based on 3/5 trees)'
      },
      {
        id: 6,
        title: 'Output Final Prediction',
        description: 'Return the final prediction based on the aggregated results from all trees in the forest.',
        animation: 'fadeInUp',
        code: 'return final_prediction, feature_importance, out_of_bag_score',
        mathematicalStep: 'Feature importance: support_calls=0.35, monthly_charges=0.28, contract_length=0.22, total_charges=0.15',
        exampleCalculation: 'Customer predicted to Churn with 60% confidence, support_calls most important feature'
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
    exampleDataset: {
      name: 'Customer Segmentation',
      description: 'Segmenting customers based on spending behavior',
      features: ['Annual Income ($)', 'Spending Score (1-100)'],
      data: [
        [15000, 39], [16000, 81], [17000, 6], [18000, 77], [19000, 40],
        [20000, 76], [21000, 6], [22000, 94], [23000, 3], [24000, 72],
        [25000, 14], [26000, 99], [27000, 15], [28000, 77], [29000, 13]
      ],
      target: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], // Cluster assignments
      featureNames: ['income', 'spending_score']
    },
    mathematicalFormulas: [
      {
        name: 'Euclidean Distance',
        formula: 'd(x, c) = √(Σ(xᵢ - cᵢ)²)',
        description: 'Distance between data point and centroid',
        variables: {
          'x': 'Data point',
          'c': 'Centroid',
          'xᵢ, cᵢ': 'i-th coordinate of point and centroid'
        }
      },
      {
        name: 'Centroid Update',
        formula: 'cⱼ = (1/|Cⱼ|) × Σ(x ∈ Cⱼ) x',
        description: 'New centroid as mean of assigned points',
        variables: {
          'cⱼ': 'Centroid of cluster j',
          'Cⱼ': 'Set of points assigned to cluster j',
          '|Cⱼ|': 'Number of points in cluster j'
        }
      },
      {
        name: 'Within-Cluster Sum of Squares',
        formula: 'WCSS = Σⱼ Σ(x ∈ Cⱼ) ||x - cⱼ||²',
        description: 'Objective function to minimize',
        variables: {
          'WCSS': 'Within-cluster sum of squares',
          '||x - cⱼ||²': 'Squared distance from point to centroid'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Choose Number of Clusters (k)',
        description: 'Decide on the number of clusters (k) you want to create. This can be determined using methods like the elbow method or silhouette analysis.',
        animation: 'fadeInDown',
        code: 'k = 3  # or use elbow_method(data) to find optimal k',
        mathematicalStep: 'Using elbow method, optimal k = 3 for customer data',
        exampleCalculation: 'k=3 chosen based on WCSS reduction: k=1→15000, k=2→8000, k=3→4500, k=4→4200'
      },
      {
        id: 2,
        title: 'Initialize Centroids',
        description: 'Randomly place k centroids in the feature space or use smart initialization like K-means++.',
        animation: 'zoomIn',
        code: 'centroids = random_initialize(k, data_bounds) or kmeans_plus_plus(data, k)',
        mathematicalStep: 'Initial centroids: C₁=(20000, 50), C₂=(25000, 25), C₃=(22000, 75)',
        exampleCalculation: 'Random initialization within data bounds: Income[15k-29k], Score[3-99]'
      },
      {
        id: 3,
        title: 'Assign Points to Clusters',
        description: 'Assign each data point to the nearest centroid based on Euclidean distance.',
        animation: 'slideInLeft',
        code: 'for point in data: cluster[point] = argmin(distance(point, centroid))',
        mathematicalStep: 'Point (16000, 81): d₁=√((16000-20000)² + (81-50)²) = √(16M + 961) = 4012',
        exampleCalculation: 'd₁=4012, d₂=√((16000-25000)² + (81-25)²)=9062, d₃=√((16000-22000)² + (81-75)²)=6001 → Assign to C₁'
      },
      {
        id: 4,
        title: 'Update Centroids',
        description: 'Move each centroid to the center (mean) of all points assigned to its cluster.',
        animation: 'bounceIn',
        code: 'for i in k: centroids[i] = mean(points_in_cluster[i])',
        mathematicalStep: 'C₁ new = mean of assigned points = (18000, 45)',
        exampleCalculation: 'Cluster 1 points: [(15000,39), (17000,6), (19000,40)] → New C₁ = (17000, 28.3)'
      },
      {
        id: 5,
        title: 'Check for Convergence',
        description: 'Check if centroids have stopped moving significantly or maximum iterations have been reached.',
        animation: 'rotateIn',
        code: 'if sum(distance(old_centroids, new_centroids)) < tolerance: break',
        mathematicalStep: 'Centroid movement: ||C₁ₙₑw - C₁ₒₗd|| = ||(17000,28.3) - (20000,50)|| = 3606',
        exampleCalculation: 'Total movement = 3606 + 2100 + 1800 = 7506 > tolerance(100) → Continue'
      },
      {
        id: 6,
        title: 'Output Final Clusters',
        description: 'Return the final cluster assignments and centroid positions.',
        animation: 'fadeInUp',
        code: 'return cluster_assignments, final_centroids, inertia',
        mathematicalStep: 'Final clusters: Low spenders (5 points), High spenders (6 points), Medium spenders (4 points)',
        exampleCalculation: 'WCSS = 2100 + 1800 + 1200 = 5100, Silhouette score = 0.72 (good clustering)'
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
    exampleDataset: {
      name: 'Handwritten Digit Recognition',
      description: 'Classifying handwritten digits (0-9) from pixel data',
      features: ['Pixel 1', 'Pixel 2', 'Pixel 3', 'Pixel 4'], // Simplified 2x2 image
      data: [
        [0.1, 0.9, 0.1, 0.9], [0.9, 0.1, 0.9, 0.1], [0.8, 0.8, 0.2, 0.2],
        [0.2, 0.8, 0.2, 0.8], [0.7, 0.3, 0.7, 0.3], [0.3, 0.7, 0.3, 0.7],
        [0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9], [0.6, 0.4, 0.6, 0.4],
        [0.4, 0.6, 0.4, 0.6]
      ],
      target: [1, 7, 0, 1, 7, 1, 0, 0, 7, 1], // Simplified: just 0, 1, 7
      featureNames: ['pixel_1', 'pixel_2', 'pixel_3', 'pixel_4']
    },
    mathematicalFormulas: [
      {
        name: 'Forward Propagation',
        formula: 'aˡ = σ(Wˡaˡ⁻¹ + bˡ)',
        description: 'Compute activations for layer l',
        variables: {
          'aˡ': 'Activations in layer l',
          'Wˡ': 'Weight matrix for layer l',
          'bˡ': 'Bias vector for layer l',
          'σ': 'Activation function (e.g., ReLU, sigmoid)'
        }
      },
      {
        name: 'Cross-Entropy Loss',
        formula: 'L = -Σᵢ yᵢ log(ŷᵢ)',
        description: 'Loss function for classification',
        variables: {
          'yᵢ': 'True label (one-hot encoded)',
          'ŷᵢ': 'Predicted probability for class i'
        }
      },
      {
        name: 'Backpropagation',
        formula: 'δˡ = ((Wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ\'(zˡ)',
        description: 'Error propagation backwards',
        variables: {
          'δˡ': 'Error term for layer l',
          '⊙': 'Element-wise multiplication',
          'σ\'': 'Derivative of activation function'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Define Network Architecture',
        description: 'Design the network structure: input layer size, number of hidden layers, neurons per layer, and output layer size.',
        animation: 'slideInDown',
        code: 'network = [input_size, hidden1_size, hidden2_size, output_size]',
        mathematicalStep: 'Architecture: 4 inputs → 6 hidden → 4 hidden → 3 outputs',
        exampleCalculation: 'Input: 4 pixels, Hidden: 6 neurons (ReLU), Hidden: 4 neurons (ReLU), Output: 3 classes (softmax)'
      },
      {
        id: 2,
        title: 'Initialize Weights and Biases',
        description: 'Randomly initialize weights and biases for all connections between neurons using techniques like Xavier or He initialization.',
        animation: 'zoomIn',
        code: 'W = random_normal(0, sqrt(2/n_inputs)), b = zeros(n_neurons)',
        mathematicalStep: 'W¹ ~ N(0, √(2/4)), W² ~ N(0, √(2/6)), W³ ~ N(0, √(2/4))',
        exampleCalculation: 'W¹[0,0] = 0.71, W¹[0,1] = -0.23, b¹ = [0,0,0,0,0,0]'
      },
      {
        id: 3,
        title: 'Forward Propagation',
        description: 'Pass input data through the network, calculating weighted sums and applying activation functions at each layer.',
        animation: 'slideInRight',
        code: 'for layer in network: z = W*a + b, a = activation_function(z)',
        mathematicalStep: 'z¹ = W¹×[0.1,0.9,0.1,0.9] + b¹, a¹ = ReLU(z¹)',
        exampleCalculation: 'z¹[0] = 0.71×0.1 + (-0.23)×0.9 + ... = 0.45, a¹[0] = max(0, 0.45) = 0.45'
      },
      {
        id: 4,
        title: 'Calculate Loss',
        description: 'Compare predicted outputs with actual targets using a loss function (MSE for regression, cross-entropy for classification).',
        animation: 'bounceIn',
        code: 'loss = cross_entropy(predictions, targets) or mse(predictions, targets)',
        mathematicalStep: 'ŷ = [0.2, 0.7, 0.1], y = [0, 1, 0] (target class 1)',
        exampleCalculation: 'Loss = -log(0.7) = 0.357 (want to maximize probability of correct class)'
      },
      {
        id: 5,
        title: 'Backpropagation',
        description: 'Calculate gradients by propagating errors backward through the network using the chain rule.',
        animation: 'slideInLeft',
        code: 'for layer in reverse(network): δ = δ_next * W.T * activation_derivative(z)',
        mathematicalStep: 'δ³ = ŷ - y = [0.2, -0.3, 0.1], δ² = (W³)ᵀδ³ ⊙ ReLU\'(z²)',
        exampleCalculation: '∂L/∂W³ = δ³ × (a²)ᵀ, ∂L/∂W² = δ² × (a¹)ᵀ'
      },
      {
        id: 6,
        title: 'Update Weights and Biases',
        description: 'Adjust weights and biases using gradient descent or advanced optimizers like Adam to minimize the loss.',
        animation: 'fadeInUp',
        code: 'W = W - learning_rate * ∇W, b = b - learning_rate * ∇b',
        mathematicalStep: 'W³ = W³ - 0.01 × ∂L/∂W³, b³ = b³ - 0.01 × ∂L/∂b³',
        exampleCalculation: 'After 1000 epochs: Loss = 0.05, Accuracy = 95% on training data'
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
    exampleDataset: {
      name: 'Email Spam Classification',
      description: 'Classifying emails as spam or not spam based on features',
      features: ['Word Count', 'Capital Letters %', 'Exclamation Marks', 'Links Count'],
      data: [
        [50, 5, 1, 0], [200, 25, 8, 5], [100, 10, 2, 1], [300, 40, 12, 8],
        [75, 8, 1, 0], [250, 35, 10, 6], [120, 12, 3, 1], [180, 30, 7, 4],
        [60, 6, 1, 0], [220, 38, 9, 7]
      ],
      target: ['Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam'],
      featureNames: ['word_count', 'capital_percent', 'exclamation_marks', 'links_count']
    },
    mathematicalFormulas: [
      {
        name: 'Linear Decision Function',
        formula: 'f(x) = wᵀx + b = Σᵢ αᵢyᵢK(xᵢ, x) + b',
        description: 'Decision function using support vectors',
        variables: {
          'w': 'Weight vector',
          'b': 'Bias term',
          'αᵢ': 'Lagrange multiplier for support vector i',
          'K(xᵢ, x)': 'Kernel function'
        }
      },
      {
        name: 'Margin Maximization',
        formula: 'max(2/||w||) ⟺ min(½||w||²)',
        description: 'Maximize margin between classes',
        variables: {
          '||w||': 'Norm of weight vector',
          '2/||w||': 'Margin width'
        }
      },
      {
        name: 'RBF Kernel',
        formula: 'K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)',
        description: 'Radial Basis Function kernel for non-linear separation',
        variables: {
          'γ': 'Kernel parameter (controls influence radius)',
          '||xᵢ - xⱼ||²': 'Squared Euclidean distance'
        }
      }
    ],
    steps: [
      {
        id: 1,
        title: 'Data Preparation and Scaling',
        description: 'Prepare and scale your feature data for optimal SVM performance. Feature scaling is crucial for SVM.',
        animation: 'fadeInDown',
        code: 'X_scaled = StandardScaler().fit_transform(X)',
        mathematicalStep: 'Scale features: word_count: (50-300)→(-1.2,1.8), capital_percent: (5-40)→(-1.1,1.6)',
        exampleCalculation: 'Email 1: [50,5,1,0] → [-1.2,-1.1,-0.8,-1.0] after standardization'
      },
      {
        id: 2,
        title: 'Choose Kernel Function',
        description: 'Select appropriate kernel function (linear, polynomial, RBF, sigmoid) based on data complexity and dimensionality.',
        animation: 'slideInLeft',
        code: 'kernel = "rbf" or "linear" or "poly", C = regularization_parameter',
        mathematicalStep: 'Choose RBF kernel with γ=0.1, C=1.0 for non-linear separation',
        exampleCalculation: 'RBF kernel: K(x₁,x₂) = exp(-0.1×||x₁-x₂||²) maps to infinite dimensions'
      },
      {
        id: 3,
        title: 'Identify Support Vectors',
        description: 'Find data points closest to the decision boundary that will determine the optimal hyperplane.',
        animation: 'zoomIn',
        code: 'support_vectors = points where 0 < α_i < C (on margin boundary)',
        mathematicalStep: 'Support vectors: Email 3 (borderline not spam), Email 8 (borderline spam)',
        exampleCalculation: '3 support vectors found with α₃=0.8, α₈=0.6, α₁₀=0.4'
      },
      {
        id: 4,
        title: 'Optimize Hyperplane',
        description: 'Find the hyperplane that maximizes the margin between classes using quadratic programming or SMO algorithm.',
        animation: 'rotateIn',
        code: 'maximize: Σα_i - 0.5*ΣΣα_i*α_j*y_i*y_j*K(x_i,x_j) subject to constraints',
        mathematicalStep: 'Solve: max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ) s.t. Σαᵢyᵢ=0, 0≤αᵢ≤C',
        exampleCalculation: 'Optimal α = [0, 0, 0.8, 0, 0, 0, 0, 0.6, 0, 0.4], bias b = -0.3'
      },
      {
        id: 5,
        title: 'Apply Kernel Trick',
        description: 'Use kernel functions to handle non-linearly separable data by mapping to higher-dimensional space.',
        animation: 'bounceIn',
        code: 'K(x_i, x_j) = φ(x_i)·φ(x_j) where φ maps to higher dimension',
        mathematicalStep: 'RBF kernel implicitly maps to infinite-dimensional space',
        exampleCalculation: 'Non-linear boundary in original space becomes linear in kernel space'
      },
      {
        id: 6,
        title: 'Make Predictions',
        description: 'Classify new data points based on their position relative to the decision hyperplane.',
        animation: 'fadeInUp',
        code: 'prediction = sign(Σα_i*y_i*K(x_i, x_new) + b)',
        mathematicalStep: 'New email [150, 20, 5, 3]: f(x) = 0.8×1×K(x₃,x) + 0.6×(-1)×K(x₈,x) + 0.4×(-1)×K(x₁₀,x) - 0.3',
        exampleCalculation: 'f(x) = 0.45 > 0 → Classified as Spam with margin distance 0.45'
      }
    ]
  }
];