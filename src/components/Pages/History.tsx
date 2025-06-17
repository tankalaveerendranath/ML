import React, { useState, useEffect } from 'react';
import { History as HistoryIcon, Calendar, Database, TrendingUp, Trash2, Eye, Download } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { HistoryEntry } from '../../types';

export const History: React.FC = () => {
  const { user } = useAuth();
  const [historyEntries, setHistoryEntries] = useState<HistoryEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);

  useEffect(() => {
    if (user) {
      const history = JSON.parse(localStorage.getItem('ml-dataset-history') || '[]');
      const userHistory = history.filter((entry: HistoryEntry) => entry.userId === user.id);
      setHistoryEntries(userHistory.sort((a: HistoryEntry, b: HistoryEntry) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ));
    }
  }, [user]);

  const deleteEntry = (entryId: string) => {
    const allHistory = JSON.parse(localStorage.getItem('ml-dataset-history') || '[]');
    const updatedHistory = allHistory.filter((entry: HistoryEntry) => entry.id !== entryId);
    localStorage.setItem('ml-dataset-history', JSON.stringify(updatedHistory));
    setHistoryEntries(prev => prev.filter(entry => entry.id !== entryId));
    if (selectedEntry?.id === entryId) {
      setSelectedEntry(null);
    }
  };

  const exportHistory = () => {
    const dataStr = JSON.stringify(historyEntries, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ml-history-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-100';
    if (confidence >= 80) return 'text-blue-600 bg-blue-100';
    if (confidence >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <HistoryIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Please Sign In</h2>
          <p className="text-gray-600">You need to be signed in to view your history.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Dataset Analysis History</h1>
              <p className="text-blue-100">
                View your previous dataset uploads and algorithm recommendations
              </p>
            </div>
            {historyEntries.length > 0 && (
              <button
                onClick={exportHistory}
                className="flex items-center space-x-2 bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {historyEntries.length === 0 ? (
          <div className="text-center py-16">
            <Database className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No History Yet</h2>
            <p className="text-gray-600 mb-6">
              Upload your first dataset to start building your analysis history.
            </p>
            <button className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
              Upload Dataset
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* History List */}
            <div className="lg:col-span-1 space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Analysis History ({historyEntries.length})
              </h2>
              
              <div className="space-y-3">
                {historyEntries.map((entry) => (
                  <div
                    key={entry.id}
                    className={`bg-white rounded-lg p-4 cursor-pointer transition-all duration-200 border-2 ${
                      selectedEntry?.id === entry.id
                        ? 'border-blue-500 shadow-md'
                        : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                    }`}
                    onClick={() => setSelectedEntry(entry)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-medium text-gray-900 truncate">
                        {entry.dataset.name}
                      </h3>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteEntry(entry.id);
                        }}
                        className="text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    
                    <div className="flex items-center space-x-2 text-sm text-gray-600 mb-2">
                      <Calendar className="w-4 h-4" />
                      <span>{formatDate(entry.timestamp)}</span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-blue-50 px-2 py-1 rounded">
                        <span className="text-blue-700">{entry.dataset.samples.toLocaleString()} samples</span>
                      </div>
                      <div className="bg-green-50 px-2 py-1 rounded">
                        <span className="text-green-700">{entry.dataset.features} features</span>
                      </div>
                    </div>
                    
                    <div className="mt-2 text-xs text-gray-500">
                      {entry.recommendations.length} recommendation{entry.recommendations.length !== 1 ? 's' : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed View */}
            <div className="lg:col-span-2">
              {selectedEntry ? (
                <div className="bg-white rounded-xl shadow-lg p-8">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-gray-900">
                      {selectedEntry.dataset.name}
                    </h2>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <Calendar className="w-4 h-4" />
                      <span>{formatDate(selectedEntry.timestamp)}</span>
                    </div>
                  </div>

                  {/* Dataset Overview */}
                  <div className="mb-8">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Overview</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {selectedEntry.dataset.samples.toLocaleString()}
                        </div>
                        <div className="text-sm text-blue-700">Samples</div>
                      </div>
                      <div className="text-center p-4 bg-emerald-50 rounded-lg">
                        <div className="text-2xl font-bold text-emerald-600">
                          {selectedEntry.dataset.features}
                        </div>
                        <div className="text-sm text-emerald-700">Features</div>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <div className="text-sm font-medium text-purple-600 capitalize">
                          {selectedEntry.dataset.type}
                        </div>
                        <div className="text-xs text-purple-700">Data Type</div>
                      </div>
                      <div className="text-center p-4 bg-orange-50 rounded-lg">
                        <div className="text-sm font-medium text-orange-600 capitalize">
                          {selectedEntry.dataset.target}
                        </div>
                        <div className="text-xs text-orange-700">Problem Type</div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Algorithm Recommendations
                    </h3>
                    <div className="space-y-4">
                      {selectedEntry.recommendations.map((rec, index) => (
                        <div key={index} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-start justify-between mb-2">
                            <h4 className="font-medium text-gray-900">{rec.algorithm}</h4>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(rec.confidence)}`}>
                              {rec.confidence}% match
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 mb-3">{rec.reasoning}</p>
                          
                          <div className="flex items-center justify-between">
                            <div className="w-full bg-gray-200 rounded-full h-2 mr-3">
                              <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${rec.confidence}%` }}
                              ></div>
                            </div>
                            <button className="text-blue-600 hover:text-blue-700 text-sm font-medium whitespace-nowrap">
                              Learn More
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                  <Eye className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Select an Entry
                  </h3>
                  <p className="text-gray-600">
                    Choose a dataset analysis from the list to view detailed information and recommendations.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};