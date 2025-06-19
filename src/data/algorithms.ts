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
    mathematicalExample: {
      title: 'House Price Prediction Example',
      dataset: {
        description: 'Predicting house prices based on size (square feet)',
        data: [
          { size: 1000, price: 200000 },
          { size: 1200, price: 240000 },
          { size: 1500, price: 300000 },
          { size: 1800, price: 360000 },
          { size: 2000, price: 400000 }
        ],
        features: ['House Size (sq ft)'],
        target: 'Price ($)'
      },
      calculations: [
        {
          step: 1,
          title: 'Initialize Parameters',
          formula: 'θ₀ = 0, θ₁ = 0, α = 0.0001',
          calculation: 'Initial slope = 0, Initial intercept = 0, Learning rate = 0.0001',
          result: 'θ₀ = 0, θ₁ = 0',
          explanation: 'Start with zero weights and set a small learning rate for stable convergence.'
        },
        {
          step: 2,
          title: 'Calculate Predictions',
          formula: 'ŷᵢ = θ₀ + θ₁ × xᵢ',
          calculation: 'ŷ₁ = 0 + 0 × 1000 = 0, ŷ₂ = 0 + 0 × 1200 = 0, ...',
          result: 'All predictions = 0',
          explanation: 'With zero parameters, all initial predictions are zero.'
        },
        {
          step: 3,
          title: 'Compute Cost (MSE)',
          formula: 'J = (1/2m) × Σ(ŷᵢ - yᵢ)²',
          calculation: 'J = (1/10) × [(0-200000)² + (0-240000)² + ... + (0-400000)²]',
          result: 'J = 9.24 × 10¹⁰',
          explanation: 'High initial cost due to large prediction errors.'
        },
        {
          step: 4,
          title: 'Calculate Gradients',
          formula: '∂J/∂θ₀ = (1/m) × Σ(ŷᵢ - yᵢ), ∂J/∂θ₁ = (1/m) × Σ((ŷᵢ - yᵢ) × xᵢ)',
          calculation: '∂J/∂θ₀ = (1/5) × (-1,500,000) = -300,000, ∂J/∂θ₁ = (1/5) × (-2,220,000,000) = -444,000,000',
          result: '∂J/∂θ₀ = -300,000, ∂J/∂θ₁ = -444,000,000',
          explanation: 'Negative gradients indicate we need to increase both parameters.'
        },
        {
          step: 5,
          title: 'Update Parameters',
          formula: 'θ₀ = θ₀ - α × ∂J/∂θ₀, θ₁ = θ₁ - α × ∂J/∂θ₁',
          calculation: 'θ₀ = 0 - 0.0001 × (-300,000) = 30, θ₁ = 0 - 0.0001 × (-444,000,000) = 44,400',
          result: 'θ₀ = 30, θ₁ = 44,400',
          explanation: 'Parameters move in direction that reduces cost.'
        },
        {
          step: 6,
          title: 'Final Model (after convergence)',
          formula: 'ŷ = θ₀ + θ₁ × x',
          calculation: 'After 1000 iterations: θ₀ ≈ 20,000, θ₁ ≈ 190',
          result: 'ŷ = 20,000 + 190 × size',
          explanation: 'Final model: Base price $20,000 + $190 per square foot.'
        }
      ],
      result: {
        description: 'Trained Linear Regression Model',
        value: 'Price = $20,000 + $190 × Size',
        interpretation: 'For every additional square foot, the house price increases by $190. The base price (y-intercept) is $20,000.'
      }
    },
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
    mathematicalExample: {
      title: 'Loan Approval Decision Tree',
      dataset: {
        description: 'Predicting loan approval based on age, income, and credit score',
        data: [
          { age: 25, income: 30000, credit: 650, approved: 'No' },
          { age: 35, income: 60000, credit: 720, approved: 'Yes' },
          { age: 45, income: 80000, credit: 680, approved: 'Yes' },
          { age: 28, income: 40000, credit: 600, approved: 'No' },
          { age: 55, income: 90000, credit: 750, approved: 'Yes' },
          { age: 32, income: 50000, credit: 700, approved: 'Yes' }
        ],
        features: ['Age', 'Income', 'Credit Score'],
        target: 'Loan Approved'
      },
      calculations: [
        {
          step: 1,
          title: 'Calculate Initial Entropy',
          formula: 'Entropy(S) = -Σ(pᵢ × log₂(pᵢ))',
          calculation: 'p(Yes) = 4/6 = 0.67, p(No) = 2/6 = 0.33, Entropy = -(0.67×log₂(0.67) + 0.33×log₂(0.33))',
          result: 'Entropy(S) = 0.918',
          explanation: 'High entropy indicates mixed classes in the dataset.'
        },
        {
          step: 2,
          title: 'Calculate Information Gain for Income > 50000',
          formula: 'Gain(S,A) = Entropy(S) - Σ((|Sᵥ|/|S|) × Entropy(Sᵥ))',
          calculation: 'Left: Income ≤ 50000 (3 samples: 1 Yes, 2 No), Right: Income > 50000 (3 samples: 3 Yes, 0 No)',
          result: 'Gain(Income) = 0.918 - (3/6×0.918 + 3/6×0) = 0.459',
          explanation: 'Income split provides good separation between classes.'
        },
        {
          step: 3,
          title: 'Calculate Information Gain for Credit > 700',
          formula: 'Gain(S,A) = Entropy(S) - Σ((|Sᵥ|/|S|) × Entropy(Sᵥ))',
          calculation: 'Left: Credit ≤ 700 (4 samples: 2 Yes, 2 No), Right: Credit > 700 (2 samples: 2 Yes, 0 No)',
          result: 'Gain(Credit) = 0.918 - (4/6×1.0 + 2/6×0) = 0.251',
          explanation: 'Credit score split provides less information gain than income.'
        },
        {
          step: 4,
          title: 'Select Best Split',
          formula: 'Best_Feature = argmax(Gain(S, Feature))',
          calculation: 'Compare: Gain(Income) = 0.459, Gain(Credit) = 0.251, Gain(Age) = 0.170',
          result: 'Best split: Income > 50000',
          explanation: 'Income provides the highest information gain, so it becomes the root split.'
        },
        {
          step: 5,
          title: 'Recursive Splitting',
          formula: 'Repeat process for each child node',
          calculation: 'Left subtree (Income ≤ 50000): Further split on Credit > 650, Right subtree: Pure (all Yes)',
          result: 'Tree depth = 2',
          explanation: 'Continue splitting until nodes are pure or stopping criteria are met.'
        },
        {
          step: 6,
          title: 'Final Decision Rules',
          formula: 'IF-THEN rules extracted from tree paths',
          calculation: 'Rule 1: IF Income > 50000 THEN Approved = Yes, Rule 2: IF Income ≤ 50000 AND Credit > 650 THEN Approved = Yes',
          result: 'Accuracy = 100% on training data',
          explanation: 'Decision tree creates interpretable rules for loan approval.'
        }
      ],
      result: {
        description: 'Decision Tree Classification Rules',
        value: 'IF Income > $50,000 THEN Approve, ELSE IF Credit > 650 THEN Approve, ELSE Reject',
        interpretation: 'The model prioritizes income over credit score, with a clear decision boundary at $50,000 annual income.'
      }
    },
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
    mathematicalExample: {
      title: 'Email Spam Classification',
      dataset: {
        description: 'Classifying emails as spam or not spam using word frequencies',
        data: [
          { word_free: 5, word_money: 3, word_urgent: 2, spam: 'Yes' },
          { word_free: 0, word_money: 1, word_urgent: 0, spam: 'No' },
          { word_free: 8, word_money: 6, word_urgent: 4, spam: 'Yes' },
          { word_free: 1, word_money: 0, word_urgent: 0, spam: 'No' },
          { word_free: 3, word_money: 4, word_urgent: 3, spam: 'Yes' },
          { word_free: 0, word_money: 2, word_urgent: 1, spam: 'No' }
        ],
        features: ['Word "free" count', 'Word "money" count', 'Word "urgent" count'],
        target: 'Spam Classification'
      },
      calculations: [
        {
          step: 1,
          title: 'Bootstrap Sample 1',
          formula: 'Sample with replacement from original dataset',
          calculation: 'Bootstrap 1: [sample1, sample3, sample5, sample1, sample6, sample2]',
          result: 'Tree 1 training data: 6 samples (with duplicates)',
          explanation: 'Each tree trains on a different bootstrap sample of the original data.'
        },
        {
          step: 2,
          title: 'Random Feature Selection',
          formula: 'Select √p features at each split (p = total features)',
          calculation: 'For 3 features, select √3 ≈ 2 random features at each split',
          result: 'Tree 1: Uses features [word_free, word_money] at root',
          explanation: 'Random feature selection increases diversity among trees.'
        },
        {
          step: 3,
          title: 'Build Tree 1',
          formula: 'Standard decision tree algorithm with feature subset',
          calculation: 'Best split: word_free > 2.5, Left: 2 No, Right: 4 Yes',
          result: 'Tree 1 prediction rule: IF word_free > 2.5 THEN Spam',
          explanation: 'Each tree creates its own decision rules based on its training data.'
        },
        {
          step: 4,
          title: 'Build Additional Trees (Tree 2, Tree 3)',
          formula: 'Repeat bootstrap + random features + tree building',
          calculation: 'Tree 2: word_money > 1.5, Tree 3: word_urgent > 0.5',
          result: 'Tree 2: Money-based rule, Tree 3: Urgency-based rule',
          explanation: 'Multiple trees capture different patterns in the data.'
        },
        {
          step: 5,
          title: 'Make Predictions (Voting)',
          formula: 'Majority vote for classification',
          calculation: 'New email [word_free=4, word_money=2, word_urgent=1]: Tree1=Spam, Tree2=Spam, Tree3=Spam',
          result: 'Final prediction: Spam (3/3 votes)',
          explanation: 'Combine predictions from all trees using majority voting.'
        },
        {
          step: 6,
          title: 'Calculate Feature Importance',
          formula: 'Average importance across all trees',
          calculation: 'word_free: 0.6, word_money: 0.3, word_urgent: 0.1',
          result: 'Most important feature: word_free',
          explanation: 'Feature importance helps understand which features contribute most to predictions.'
        }
      ],
      result: {
        description: 'Random Forest Ensemble Model',
        value: 'Accuracy: 95%, Feature Importance: [word_free: 60%, word_money: 30%, word_urgent: 10%]',
        interpretation: 'The ensemble model achieves high accuracy by combining multiple decision trees, with "free" being the most predictive word for spam detection.'
      }
    },
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
    mathematicalExample: {
      title: 'Customer Segmentation Example',
      dataset: {
        description: 'Segmenting customers based on annual spending and age',
        data: [
          { age: 25, spending: 20000 },
          { age: 30, spending: 25000 },
          { age: 35, spending: 30000 },
          { age: 45, spending: 60000 },
          { age: 50, spending: 65000 },
          { age: 55, spending: 70000 }
        ],
        features: ['Age', 'Annual Spending ($)'],
        target: 'Customer Segment'
      },
      calculations: [
        {
          step: 1,
          title: 'Initialize Centroids (k=2)',
          formula: 'Randomly place k centroids in feature space',
          calculation: 'C1 = (30, 30000), C2 = (50, 50000)',
          result: 'Initial centroids placed',
          explanation: 'Start with random centroid positions or use K-means++ for better initialization.'
        },
        {
          step: 2,
          title: 'Assign Points to Nearest Centroid (Iteration 1)',
          formula: 'distance = √((x₁-c₁)² + (x₂-c₂)²)',
          calculation: 'Point (25,20000): d(C1)=10000, d(C2)=36056 → Cluster 1, Point (45,60000): d(C1)=33541, d(C2)=14142 → Cluster 2',
          result: 'Cluster 1: [(25,20000), (30,25000), (35,30000)], Cluster 2: [(45,60000), (50,65000), (55,70000)]',
          explanation: 'Each point is assigned to the cluster with the nearest centroid.'
        },
        {
          step: 3,
          title: 'Update Centroids (Iteration 1)',
          formula: 'New centroid = mean of assigned points',
          calculation: 'C1_new = ((25+30+35)/3, (20000+25000+30000)/3) = (30, 25000), C2_new = ((45+50+55)/3, (60000+65000+70000)/3) = (50, 65000)',
          result: 'Updated centroids: C1 = (30, 25000), C2 = (50, 65000)',
          explanation: 'Move centroids to the center of their assigned points.'
        },
        {
          step: 4,
          title: 'Check Convergence',
          formula: 'Convergence if centroid movement < threshold',
          calculation: 'C1 movement: √((30-30)² + (25000-30000)²) = 5000, C2 movement: √((50-50)² + (65000-50000)²) = 15000',
          result: 'Not converged (movement > threshold)',
          explanation: 'Continue iterations until centroids stop moving significantly.'
        },
        {
          step: 5,
          title: 'Final Assignment (After Convergence)',
          formula: 'Repeat steps 2-4 until convergence',
          calculation: 'After 3 iterations: C1 = (30, 25000), C2 = (50, 65000)',
          result: 'Final clusters: Young/Low-spending vs Mature/High-spending',
          explanation: 'Algorithm converges when centroids stabilize.'
        },
        {
          step: 6,
          title: 'Calculate Within-Cluster Sum of Squares (WCSS)',
          formula: 'WCSS = Σ(distance from points to their centroid)²',
          calculation: 'WCSS = Σ₁[(25-30)² + (20000-25000)²] + Σ₂[(45-50)² + (60000-65000)²] + ...',
          result: 'WCSS = 1.25 × 10⁸',
          explanation: 'Lower WCSS indicates tighter, more cohesive clusters.'
        }
      ],
      result: {
        description: 'Customer Segmentation Results',
        value: 'Cluster 1: Young customers (avg age 30, spending $25K), Cluster 2: Mature customers (avg age 50, spending $65K)',
        interpretation: 'The algorithm successfully identified two distinct customer segments based on age and spending patterns, enabling targeted marketing strategies.'
      }
    },
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
    mathematicalExample: {
      title: 'XOR Gate Classification',
      dataset: {
        description: 'Learning the XOR logical function using a simple neural network',
        data: [
          { x1: 0, x2: 0, output: 0 },
          { x1: 0, x2: 1, output: 1 },
          { x1: 1, x2: 0, output: 1 },
          { x1: 1, x2: 1, output: 0 }
        ],
        features: ['Input 1', 'Input 2'],
        target: 'XOR Output'
      },
      calculations: [
        {
          step: 1,
          title: 'Network Architecture',
          formula: 'Input layer (2) → Hidden layer (2) → Output layer (1)',
          calculation: 'Weights: W1[2×2], b1[2×1], W2[2×1], b2[1×1]',
          result: 'Total parameters: 9 (6 weights + 3 biases)',
          explanation: 'Simple network with one hidden layer to learn non-linear XOR function.'
        },
        {
          step: 2,
          title: 'Initialize Weights (Random)',
          formula: 'W ~ N(0, 0.5), b = 0',
          calculation: 'W1 = [[0.5, -0.3], [0.2, 0.8]], b1 = [0, 0], W2 = [[0.4], [-0.6]], b2 = [0]',
          result: 'Random weight initialization completed',
          explanation: 'Small random weights break symmetry and enable learning.'
        },
        {
          step: 3,
          title: 'Forward Pass (Input: [1,0])',
          formula: 'z = W·x + b, a = σ(z) where σ(z) = 1/(1+e^(-z))',
          calculation: 'Hidden: z1 = [0.5×1 + (-0.3)×0, 0.2×1 + 0.8×0] = [0.5, 0.2], a1 = [0.62, 0.55], Output: z2 = 0.4×0.62 + (-0.6)×0.55 = -0.082, a2 = 0.48',
          result: 'Predicted output: 0.48, Target: 1, Error: 0.52',
          explanation: 'Forward propagation computes prediction through network layers.'
        },
        {
          step: 4,
          title: 'Calculate Loss',
          formula: 'Loss = 0.5 × (predicted - target)²',
          calculation: 'Loss = 0.5 × (0.48 - 1)² = 0.5 × (-0.52)² = 0.135',
          result: 'Mean Squared Error: 0.135',
          explanation: 'Loss function measures prediction error for optimization.'
        },
        {
          step: 5,
          title: 'Backpropagation',
          formula: 'δ_output = (a - y) × σ\'(z), δ_hidden = (W^T × δ_output) × σ\'(z)',
          calculation: 'δ2 = (0.48-1) × 0.48×(1-0.48) = -0.13, δ1 = [0.4×(-0.13), (-0.6)×(-0.13)] × [0.62×0.38, 0.55×0.45] = [-0.012, 0.035]',
          result: 'Gradients computed for all weights',
          explanation: 'Backpropagation calculates gradients for weight updates.'
        },
        {
          step: 6,
          title: 'Update Weights',
          formula: 'W = W - α × ∇W, b = b - α × ∇b',
          calculation: 'α = 0.5, W2_new = [0.4 - 0.5×(-0.08), -0.6 - 0.5×(-0.07)] = [0.44, -0.565], W1 updated similarly',
          result: 'Weights updated using gradient descent',
          explanation: 'After many iterations, network learns to output correct XOR values.'
        }
      ],
      result: {
        description: 'Trained Neural Network for XOR',
        value: 'Final Accuracy: 100%, Learned non-linear decision boundary',
        interpretation: 'The neural network successfully learned the XOR function, demonstrating its ability to capture non-linear relationships that linear models cannot represent.'
      }
    },
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
    mathematicalExample: {
      title: 'Binary Text Classification',
      dataset: {
        description: 'Classifying documents as positive or negative sentiment',
        data: [
          { word_good: 3, word_bad: 0, sentiment: 1 },
          { word_good: 0, word_bad: 2, sentiment: -1 },
          { word_good: 2, word_bad: 1, sentiment: 1 },
          { word_good: 1, word_bad: 3, sentiment: -1 },
          { word_good: 4, word_bad: 0, sentiment: 1 }
        ],
        features: ['Good word count', 'Bad word count'],
        target: 'Sentiment (+1/-1)'
      },
      calculations: [
        {
          step: 1,
          title: 'Formulate Optimization Problem',
          formula: 'Minimize: ½||w||² subject to: yᵢ(w·xᵢ + b) ≥ 1',
          calculation: 'Find hyperplane w·x + b = 0 that maximizes margin 2/||w||',
          result: 'Quadratic optimization problem setup',
          explanation: 'SVM finds the hyperplane with maximum margin between classes.'
        },
        {
          step: 2,
          title: 'Identify Support Vectors',
          formula: 'Support vectors: points where yᵢ(w·xᵢ + b) = 1',
          calculation: 'Points closest to decision boundary: (3,0,+1), (0,2,-1), (1,3,-1)',
          result: '3 support vectors identified',
          explanation: 'Only support vectors determine the decision boundary.'
        },
        {
          step: 3,
          title: 'Solve Dual Problem',
          formula: 'Maximize: Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼ(xᵢ·xⱼ) subject to: Σαᵢyᵢ = 0, αᵢ ≥ 0',
          calculation: 'Using SMO algorithm: α₁ = 0.4, α₂ = 0.3, α₃ = 0.1, others = 0',
          result: 'Lagrange multipliers: α = [0.4, 0.3, 0, 0.1, 0]',
          explanation: 'Dual formulation allows kernel trick and efficient optimization.'
        },
        {
          step: 4,
          title: 'Calculate Weight Vector',
          formula: 'w = Σαᵢyᵢxᵢ (only for support vectors)',
          calculation: 'w = 0.4×(+1)×[3,0] + 0.3×(-1)×[0,2] + 0.1×(-1)×[1,3] = [1.1, -0.9]',
          result: 'Weight vector: w = [1.1, -0.9]',
          explanation: 'Weight vector is linear combination of support vectors.'
        },
        {
          step: 5,
          title: 'Calculate Bias Term',
          formula: 'b = yᵢ - w·xᵢ (using any support vector)',
          calculation: 'Using support vector (3,0,+1): b = 1 - [1.1,-0.9]·[3,0] = 1 - 3.3 = -2.3',
          result: 'Bias: b = -2.3',
          explanation: 'Bias ensures support vectors lie exactly on margin boundary.'
        },
        {
          step: 6,
          title: 'Final Decision Function',
          formula: 'f(x) = sign(w·x + b) = sign(1.1×x₁ - 0.9×x₂ - 2.3)',
          calculation: 'Test point [2,1]: f([2,1]) = sign(1.1×2 - 0.9×1 - 2.3) = sign(-1.0) = -1',
          result: 'Decision boundary: 1.1×x₁ - 0.9×x₂ - 2.3 = 0',
          explanation: 'Hyperplane separates positive and negative sentiment documents.'
        }
      ],
      result: {
        description: 'SVM Text Classifier',
        value: 'Decision boundary: 1.1×(good_words) - 0.9×(bad_words) - 2.3 = 0, Margin width: 1.82',
        interpretation: 'The SVM learned that documents with more "good" words relative to "bad" words are classified as positive sentiment, with a clear margin of separation.'
      }
    },
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