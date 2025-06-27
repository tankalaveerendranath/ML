import { useState, useEffect } from 'react';
import { HistoryEntry, Dataset, DatasetRecommendation } from '../types';
import { useAuth } from './useAuth';

export const useHistory = () => {
  const { user } = useAuth();
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    if (user) {
      loadHistory();
    }
  }, [user]);

  const loadHistory = () => {
    if (!user) return;
    
    const storedHistory = JSON.parse(localStorage.getItem('ml-website-history') || '[]');
    const userHistory = storedHistory.filter((entry: HistoryEntry) => entry.userId === user.id);
    setHistory(userHistory.sort((a: HistoryEntry, b: HistoryEntry) => 
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    ));
  };

  const addToHistory = (dataset: Dataset, recommendations: DatasetRecommendation[]) => {
    if (!user) return;

    const newEntry: HistoryEntry = {
      id: Date.now().toString(),
      userId: user.id,
      dataset,
      recommendations,
      createdAt: new Date().toISOString()
    };

    const storedHistory = JSON.parse(localStorage.getItem('ml-website-history') || '[]');
    storedHistory.push(newEntry);
    localStorage.setItem('ml-website-history', JSON.stringify(storedHistory));
    
    setHistory(prev => [newEntry, ...prev]);
  };

  const clearHistory = () => {
    if (!user) return;
    
    const storedHistory = JSON.parse(localStorage.getItem('ml-website-history') || '[]');
    const filteredHistory = storedHistory.filter((entry: HistoryEntry) => entry.userId !== user.id);
    localStorage.setItem('ml-website-history', JSON.stringify(filteredHistory));
    
    setHistory([]);
  };

  return {
    history,
    addToHistory,
    clearHistory,
    loadHistory
  };
};