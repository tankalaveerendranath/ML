import React, { useState } from 'react';
import { Code, Zap, Copy, CheckCircle, AlertCircle, Lightbulb } from 'lucide-react';

interface OptimizationSuggestion {
  type: 'performance' | 'readability' | 'best-practice' | 'bug-fix';
  title: string;
  description: string;
  before: string;
  after: string;
  impact: 'high' | 'medium' | 'low';
}

export const CodeOptimizer: React.FC = () => {
  const [inputCode, setInputCode] = useState('');
  const [optimizedCode, setOptimizedCode] = useState('');
  const [suggestions, setSuggestions] = useState<OptimizationSuggestion[]>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [language, setLanguage] = useState('python');

  const analyzeAndOptimizeCode = (code: string, lang: string) => {
    // Comprehensive code analysis and optimization
    const optimizations: OptimizationSuggestion[] = [];
    let optimized = code;

    // Python-specific optimizations
    if (lang === 'python') {
      // List comprehension optimization
      if (code.includes('for') && code.includes('append')) {
        optimizations.push({
          type: 'performance',
          title: 'Use List Comprehension',
          description: 'Replace explicit loops with list comprehensions for better performance',
          before: 'result = []\nfor item in items:\n    result.append(item * 2)',
          after: 'result = [item * 2 for item in items]',
          impact: 'medium'
        });
        optimized = optimized.replace(
          /(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+):\s*\n\s*\1\.append\(([^)]+)\)/g,
          '$1 = [$4 for $2 in $3]'
        );
      }

      // NumPy vectorization
      if (code.includes('for') && (code.includes('math.') || code.includes('**'))) {
        optimizations.push({
          type: 'performance',
          title: 'Use NumPy Vectorization',
          description: 'Replace loops with NumPy operations for significant speedup',
          before: 'for i in range(len(arr)):\n    arr[i] = math.sqrt(arr[i])',
          after: 'import numpy as np\narr = np.sqrt(arr)',
          impact: 'high'
        });
      }

      // String concatenation optimization
      if (code.includes('+=') && code.includes('str')) {
        optimizations.push({
          type: 'performance',
          title: 'Optimize String Concatenation',
          description: 'Use join() instead of += for string concatenation in loops',
          before: 'result = ""\nfor item in items:\n    result += str(item)',
          after: 'result = "".join(str(item) for item in items)',
          impact: 'medium'
        });
      }

      // Dictionary get() method
      if (code.includes('if') && code.includes('in') && code.includes('dict')) {
        optimizations.push({
          type: 'best-practice',
          title: 'Use dict.get() Method',
          description: 'Use dict.get() with default values instead of checking key existence',
          before: 'if key in my_dict:\n    value = my_dict[key]\nelse:\n    value = default',
          after: 'value = my_dict.get(key, default)',
          impact: 'low'
        });
      }

      // Machine Learning specific optimizations
      if (code.includes('sklearn') || code.includes('fit') || code.includes('predict')) {
        optimizations.push({
          type: 'performance',
          title: 'ML Pipeline Optimization',
          description: 'Use sklearn pipelines for better code organization and performance',
          before: 'scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nmodel = LogisticRegression()\nmodel.fit(X_scaled, y)',
          after: 'from sklearn.pipeline import Pipeline\npipeline = Pipeline([\n    ("scaler", StandardScaler()),\n    ("model", LogisticRegression())\n])\npipeline.fit(X, y)',
          impact: 'medium'
        });
      }

      // Pandas optimizations
      if (code.includes('pandas') || code.includes('df.')) {
        optimizations.push({
          type: 'performance',
          title: 'Pandas Vectorization',
          description: 'Use vectorized operations instead of iterrows() or apply() when possible',
          before: 'for index, row in df.iterrows():\n    df.at[index, "new_col"] = row["col1"] * row["col2"]',
          after: 'df["new_col"] = df["col1"] * df["col2"]',
          impact: 'high'
        });
      }
    }

    // JavaScript-specific optimizations
    if (lang === 'javascript') {
      // Array methods optimization
      if (code.includes('for') && code.includes('push')) {
        optimizations.push({
          type: 'readability',
          title: 'Use Array Methods',
          description: 'Replace explicit loops with array methods like map, filter, reduce',
          before: 'const result = [];\nfor (let i = 0; i < arr.length; i++) {\n    result.push(arr[i] * 2);\n}',
          after: 'const result = arr.map(item => item * 2);',
          impact: 'medium'
        });
      }

      // Async/await optimization
      if (code.includes('Promise') && code.includes('.then')) {
        optimizations.push({
          type: 'readability',
          title: 'Use Async/Await',
          description: 'Replace Promise chains with async/await for better readability',
          before: 'fetchData().then(data => {\n    return processData(data);\n}).then(result => {\n    console.log(result);\n});',
          after: 'async function handleData() {\n    const data = await fetchData();\n    const result = await processData(data);\n    console.log(result);\n}',
          impact: 'medium'
        });
      }

      // Object destructuring
      if (code.includes('obj.') && code.includes('=')) {
        optimizations.push({
          type: 'readability',
          title: 'Use Destructuring',
          description: 'Use object destructuring for cleaner variable assignment',
          before: 'const name = user.name;\nconst email = user.email;\nconst age = user.age;',
          after: 'const { name, email, age } = user;',
          impact: 'low'
        });
      }
    }

    // General optimizations for all languages
    
    // Variable naming
    if (/\b[a-z]\b/.test(code)) {
      optimizations.push({
        type: 'readability',
        title: 'Improve Variable Names',
        description: 'Use descriptive variable names instead of single letters',
        before: 'for i in range(n):\n    x = data[i]\n    y = process(x)',
        after: 'for index in range(data_length):\n    current_item = data[index]\n    processed_result = process(current_item)',
        impact: 'medium'
      });
    }

    // Function length
    const lines = code.split('\n');
    if (lines.length > 20) {
      optimizations.push({
        type: 'best-practice',
        title: 'Break Down Large Functions',
        description: 'Consider breaking this function into smaller, more focused functions',
        before: '# Large function with many responsibilities',
        after: '# Split into smaller functions:\n# - data_preprocessing()\n# - model_training()\n# - result_evaluation()',
        impact: 'medium'
      });
    }

    // Magic numbers
    if (/\b\d{2,}\b/.test(code)) {
      optimizations.push({
        type: 'best-practice',
        title: 'Replace Magic Numbers',
        description: 'Replace magic numbers with named constants',
        before: 'if score > 85:\n    grade = "A"',
        after: 'GRADE_A_THRESHOLD = 85\nif score > GRADE_A_THRESHOLD:\n    grade = "A"',
        impact: 'low'
      });
    }

    return { optimized, optimizations };
  };

  const handleOptimize = async () => {
    if (!inputCode.trim()) return;
    
    setIsOptimizing(true);
    
    // Simulate AI processing time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const { optimized, optimizations } = analyzeAndOptimizeCode(inputCode, language);
    
    setOptimizedCode(optimized);
    setSuggestions(optimizations);
    setIsOptimizing(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'performance': return <Zap className="w-4 h-4 text-orange-500" />;
      case 'readability': return <Code className="w-4 h-4 text-blue-500" />;
      case 'best-practice': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'bug-fix': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <Lightbulb className="w-4 h-4 text-purple-500" />;
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-3 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">AI Code Optimizer</h2>
            <p className="text-gray-600">Analyze and optimize your code for better performance and readability</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Your Code</h3>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="java">Java</option>
                <option value="cpp">C++</option>
              </select>
            </div>
            
            <textarea
              value={inputCode}
              onChange={(e) => setInputCode(e.target.value)}
              placeholder={`Paste your ${language} code here...

Example:
# Linear regression implementation
import numpy as np

def linear_regression(X, y):
    result = []
    for i in range(len(X)):
        result.append(X[i] * 2 + 1)
    
    error = 0
    for i in range(len(y)):
        error += (result[i] - y[i]) ** 2
    
    return result, error`}
              className="w-full h-80 p-4 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            />
            
            <button
              onClick={handleOptimize}
              disabled={!inputCode.trim() || isOptimizing}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all duration-200 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isOptimizing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Optimizing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Optimize Code</span>
                </>
              )}
            </button>
          </div>

          {/* Output Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Optimized Code</h3>
              {optimizedCode && (
                <button
                  onClick={() => copyToClipboard(optimizedCode)}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <Copy className="w-4 h-4" />
                </button>
              )}
            </div>
            
            <div className="relative">
              <textarea
                value={optimizedCode}
                readOnly
                placeholder="Optimized code will appear here..."
                className="w-full h-80 p-4 border border-gray-300 rounded-lg font-mono text-sm bg-gray-50 resize-none"
              />
              {!optimizedCode && !isOptimizing && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    <Code className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Enter code and click optimize to see improvements</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Optimization Suggestions */}
        {suggestions.length > 0 && (
          <div className="mt-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization Suggestions</h3>
            <div className="space-y-4">
              {suggestions.map((suggestion, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-6 hover:border-blue-300 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      {getTypeIcon(suggestion.type)}
                      <div>
                        <h4 className="font-semibold text-gray-900">{suggestion.title}</h4>
                        <p className="text-sm text-gray-600">{suggestion.description}</p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getImpactColor(suggestion.impact)}`}>
                      {suggestion.impact} impact
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="text-sm font-medium text-red-700 mb-2">❌ Before:</h5>
                      <pre className="bg-red-50 p-3 rounded text-xs overflow-x-auto">
                        <code>{suggestion.before}</code>
                      </pre>
                    </div>
                    <div>
                      <h5 className="text-sm font-medium text-green-700 mb-2">✅ After:</h5>
                      <pre className="bg-green-50 p-3 rounded text-xs overflow-x-auto">
                        <code>{suggestion.after}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tips Section */}
        <div className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-3">
            <Lightbulb className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">Optimization Tips</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
            <div>
              <h4 className="font-medium mb-2">Performance:</h4>
              <ul className="space-y-1 text-xs">
                <li>• Use vectorized operations (NumPy, Pandas)</li>
                <li>• Avoid nested loops when possible</li>
                <li>• Use appropriate data structures</li>
                <li>• Profile your code to find bottlenecks</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Readability:</h4>
              <ul className="space-y-1 text-xs">
                <li>• Use descriptive variable names</li>
                <li>• Keep functions small and focused</li>
                <li>• Add comments for complex logic</li>
                <li>• Follow language conventions</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};