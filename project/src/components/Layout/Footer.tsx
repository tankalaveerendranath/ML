import React from 'react';
import { Brain, Github, Twitter, Linkedin, Mail, Heart, ExternalLink } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const quickLinks = [
    { label: 'Home', href: '#home' },
    { label: 'About ML', href: '#about' },
    { label: 'Algorithms', href: '#algorithms' },
    { label: 'Dataset Upload', href: '#dataset' },
  ];

  const algorithms = [
    { label: 'Linear Regression', href: '#linear-regression' },
    { label: 'Decision Trees', href: '#decision-tree' },
    { label: 'Random Forest', href: '#random-forest' },
    { label: 'Neural Networks', href: '#neural-networks' },
  ];

  const resources = [
    { label: 'Documentation', href: '#docs', external: true },
    { label: 'Tutorials', href: '#tutorials', external: true },
    { label: 'Research Papers', href: '#papers', external: true },
    { label: 'Community', href: '#community', external: true },
  ];

  return (
    <footer className="bg-gray-900 text-white">
      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="lg:col-span-1">
            <div className="flex items-center space-x-3 mb-4">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold">ML Algorithms Hub</h3>
                <p className="text-sm text-gray-400">Learn • Explore • Discover</p>
              </div>
            </div>
            <p className="text-gray-300 text-sm mb-6 leading-relaxed">
              Your comprehensive platform for understanding machine learning algorithms through 
              interactive visualizations, step-by-step explanations, and intelligent dataset analysis.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-400 hover:text-blue-400 transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-blue-400 transition-colors">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-blue-400 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-blue-400 transition-colors">
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-white">Quick Links</h4>
            <ul className="space-y-3">
              {quickLinks.map((link, index) => (
                <li key={index}>
                  <a 
                    href={link.href}
                    className="text-gray-300 hover:text-blue-400 transition-colors text-sm flex items-center"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Algorithms */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-white">Algorithms</h4>
            <ul className="space-y-3">
              {algorithms.map((algo, index) => (
                <li key={index}>
                  <a 
                    href={algo.href}
                    className="text-gray-300 hover:text-blue-400 transition-colors text-sm flex items-center"
                  >
                    {algo.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-white">Resources</h4>
            <ul className="space-y-3">
              {resources.map((resource, index) => (
                <li key={index}>
                  <a 
                    href={resource.href}
                    className="text-gray-300 hover:text-blue-400 transition-colors text-sm flex items-center"
                  >
                    {resource.label}
                    {resource.external && <ExternalLink className="w-3 h-3 ml-1" />}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Newsletter Signup */}
        <div className="mt-12 pt-8 border-t border-gray-800">
          <div className="max-w-md mx-auto text-center">
            <h4 className="text-lg font-semibold mb-2">Stay Updated</h4>
            <p className="text-gray-400 text-sm mb-4">
              Get the latest updates on new algorithms and features
            </p>
            <div className="flex space-x-2">
              <input
                type="email"
                placeholder="Enter your email"
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 font-medium">
                Subscribe
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <span>© {currentYear} ML Algorithms Hub. Made with</span>
              <Heart className="w-4 h-4 text-red-500" />
              <span>for ML enthusiasts</span>
            </div>
            <div className="flex items-center space-x-6 text-sm text-gray-400">
              <a href="#privacy" className="hover:text-blue-400 transition-colors">Privacy Policy</a>
              <a href="#terms" className="hover:text-blue-400 transition-colors">Terms of Service</a>
              <a href="#cookies" className="hover:text-blue-400 transition-colors">Cookie Policy</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};