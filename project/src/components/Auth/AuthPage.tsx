import React, { useState } from 'react';
import { Brain } from 'lucide-react';
import { Login } from './Login';
import { Signup } from './Signup';

export const AuthPage: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl mx-auto flex items-center justify-center">
        <div className="hidden lg:flex lg:w-1/2 items-center justify-center">
          <div className="text-center">
            <div className="mb-8">
              <Brain className="w-24 h-24 text-blue-600 mx-auto mb-4 animate-pulse" />
              <h1 className="text-4xl font-bold text-gray-900 mb-4">
                ML Algorithms Hub
              </h1>
              <p className="text-xl text-gray-600 mb-8">
                Discover, Learn, and Master Machine Learning Algorithms
              </p>
            </div>
            <div className="space-y-4 text-left max-w-md">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <span className="text-gray-700">Interactive algorithm explanations</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-emerald-600 rounded-full"></div>
                <span className="text-gray-700">Step-by-step animated tutorials</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
                <span className="text-gray-700">Smart dataset analysis</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-orange-600 rounded-full"></div>
                <span className="text-gray-700">Algorithm recommendations</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="w-full lg:w-1/2">
          <div className="bg-white rounded-2xl shadow-2xl p-8 lg:p-12">
            <div className="lg:hidden text-center mb-8">
              <Brain className="w-16 h-16 text-blue-600 mx-auto mb-4" />
              <h1 className="text-2xl font-bold text-gray-900">ML Algorithms Hub</h1>
            </div>
            
            {isLogin ? (
              <Login onSwitch={() => setIsLogin(false)} />
            ) : (
              <Signup onSwitch={() => setIsLogin(true)} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};