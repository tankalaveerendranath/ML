import React from 'react';
import { Brain, Zap, Target, TrendingUp, GitBranch, Circle, Trees, SeparatorVertical as Separator, ArrowRight, CheckCircle, Star, Users, Award } from 'lucide-react';

export const AboutML: React.FC = () => {
  const mlTypes = [
    {
      title: 'Supervised Learning',
      description: 'Learn from labeled examples to make predictions on new data',
      icon: Target,
      color: 'from-blue-500 to-blue-600',
      examples: ['Classification', 'Regression', 'Prediction'],
      algorithms: ['Linear Regression', 'Decision Trees', 'SVM', 'Neural Networks']
    },
    {
      title: 'Unsupervised Learning',
      description: 'Discover hidden patterns in data without labeled examples',
      icon: Circle,
      color: 'from-purple-500 to-purple-600',
      examples: ['Clustering', 'Dimensionality Reduction', 'Association Rules'],
      algorithms: ['K-Means', 'PCA', 'DBSCAN', 'Hierarchical Clustering']
    },
    {
      title: 'Reinforcement Learning',
      description: 'Learn through interaction with environment using rewards and penalties',
      icon: Zap,
      color: 'from-green-500 to-green-600',
      examples: ['Game Playing', 'Robotics', 'Autonomous Systems'],
      algorithms: ['Q-Learning', 'Policy Gradient', 'Actor-Critic', 'Deep Q-Networks']
    }
  ];

  const mlWorkflow = [
    {
      step: 1,
      title: 'Data Collection',
      description: 'Gather relevant data from various sources',
      icon: '📊',
      details: 'Quality data is the foundation of any successful ML project'
    },
    {
      step: 2,
      title: 'Data Preprocessing',
      description: 'Clean, transform, and prepare data for analysis',
      icon: '🔧',
      details: 'Handle missing values, outliers, and feature engineering'
    },
    {
      step: 3,
      title: 'Model Selection',
      description: 'Choose appropriate algorithm based on problem type',
      icon: '🎯',
      details: 'Consider data size, complexity, and interpretability requirements'
    },
    {
      step: 4,
      title: 'Training',
      description: 'Train the model using prepared dataset',
      icon: '🏋️',
      details: 'Optimize parameters to minimize prediction errors'
    },
    {
      step: 5,
      title: 'Evaluation',
      description: 'Assess model performance using metrics',
      icon: '📈',
      details: 'Use cross-validation and test sets for unbiased evaluation'
    },
    {
      step: 6,
      title: 'Deployment',
      description: 'Deploy model to production environment',
      icon: '🚀',
      details: 'Monitor performance and retrain as needed'
    }
  ];

  const applications = [
    {
      category: 'Healthcare',
      icon: '🏥',
      examples: ['Medical Diagnosis', 'Drug Discovery', 'Personalized Treatment', 'Medical Imaging'],
      color: 'bg-red-50 border-red-200'
    },
    {
      category: 'Finance',
      icon: '💰',
      examples: ['Fraud Detection', 'Credit Scoring', 'Algorithmic Trading', 'Risk Assessment'],
      color: 'bg-green-50 border-green-200'
    },
    {
      category: 'Technology',
      icon: '💻',
      examples: ['Recommendation Systems', 'Search Engines', 'Computer Vision', 'Natural Language Processing'],
      color: 'bg-blue-50 border-blue-200'
    },
    {
      category: 'Transportation',
      icon: '🚗',
      examples: ['Autonomous Vehicles', 'Route Optimization', 'Traffic Management', 'Predictive Maintenance'],
      color: 'bg-purple-50 border-purple-200'
    }
  ];

  const benefits = [
    {
      title: 'Automation',
      description: 'Automate complex decision-making processes',
      icon: Zap
    },
    {
      title: 'Insights',
      description: 'Discover hidden patterns in large datasets',
      icon: Brain
    },
    {
      title: 'Efficiency',
      description: 'Improve operational efficiency and reduce costs',
      icon: TrendingUp
    },
    {
      title: 'Personalization',
      description: 'Deliver personalized experiences to users',
      icon: Target
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-700 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-white bg-opacity-20 rounded-full">
                <Brain className="w-16 h-16 text-white" />
              </div>
            </div>
            <h1 className="text-5xl font-bold mb-6">
              What is Machine Learning?
            </h1>
            <p className="text-xl text-blue-100 max-w-3xl mx-auto mb-8 leading-relaxed">
              Machine Learning is a subset of artificial intelligence that enables computers to learn 
              and make decisions from data without being explicitly programmed for every scenario.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <div className="bg-white bg-opacity-20 px-6 py-3 rounded-lg">
                <div className="text-2xl font-bold">6+</div>
                <div className="text-sm text-blue-100">Algorithms Covered</div>
              </div>
              <div className="bg-white bg-opacity-20 px-6 py-3 rounded-lg">
                <div className="text-2xl font-bold">Interactive</div>
                <div className="text-sm text-blue-100">Learning Experience</div>
              </div>
              <div className="bg-white bg-opacity-20 px-6 py-3 rounded-lg">
                <div className="text-2xl font-bold">Step-by-Step</div>
                <div className="text-sm text-blue-100">Explanations</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Types of Machine Learning */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Types of Machine Learning</h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Understanding the three main categories of machine learning and their applications
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {mlTypes.map((type, index) => {
              const IconComponent = type.icon;
              return (
                <div key={index} className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-shadow">
                  <div className={`w-16 h-16 bg-gradient-to-r ${type.color} rounded-xl flex items-center justify-center mb-6`}>
                    <IconComponent className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">{type.title}</h3>
                  <p className="text-gray-600 mb-6">{type.description}</p>
                  
                  <div className="mb-6">
                    <h4 className="font-semibold text-gray-900 mb-2">Use Cases:</h4>
                    <div className="flex flex-wrap gap-2">
                      {type.examples.map((example, idx) => (
                        <span key={idx} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                          {example}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Popular Algorithms:</h4>
                    <ul className="space-y-1">
                      {type.algorithms.map((algo, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-center">
                          <CheckCircle className="w-3 h-3 text-green-500 mr-2" />
                          {algo}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* ML Workflow */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Machine Learning Workflow</h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              The typical process of developing and deploying machine learning solutions
            </p>
          </div>

          <div className="relative">
            {/* Workflow Steps */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {mlWorkflow.map((item, index) => (
                <div key={index} className="relative">
                  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
                    <div className="flex items-center mb-4">
                      <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold mr-4">
                        {item.step}
                      </div>
                      <div className="text-2xl">{item.icon}</div>
                    </div>
                    <h3 className="text-lg font-bold text-gray-900 mb-2">{item.title}</h3>
                    <p className="text-gray-600 mb-3">{item.description}</p>
                    <p className="text-sm text-gray-500">{item.details}</p>
                  </div>
                  
                  {/* Arrow for desktop */}
                  {index < mlWorkflow.length - 1 && (
                    <div className="hidden lg:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                      <ArrowRight className="w-8 h-8 text-blue-500" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Applications */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Real-World Applications</h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Machine learning is transforming industries and solving complex problems across various domains
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {applications.map((app, index) => (
              <div key={index} className={`${app.color} border-2 rounded-xl p-6`}>
                <div className="flex items-center mb-4">
                  <div className="text-3xl mr-4">{app.icon}</div>
                  <h3 className="text-xl font-bold text-gray-900">{app.category}</h3>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {app.examples.map((example, idx) => (
                    <div key={idx} className="bg-white bg-opacity-70 px-3 py-2 rounded-lg text-sm font-medium text-gray-700">
                      {example}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Benefits */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Why Machine Learning Matters</h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              The key benefits that make machine learning essential for modern businesses and research
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {benefits.map((benefit, index) => {
              const IconComponent = benefit.icon;
              return (
                <div key={index} className="text-center p-6 bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <IconComponent className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{benefit.title}</h3>
                  <p className="text-gray-600 text-sm">{benefit.description}</p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Getting Started */}
        <section className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-12 text-white text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Start Learning?</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
            Explore our interactive algorithm explanations and discover which machine learning 
            techniques work best for your data and use cases.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              Explore Algorithms
            </button>
            <button className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition-colors">
              Upload Dataset
            </button>
          </div>
        </section>
      </div>
    </div>
  );
};