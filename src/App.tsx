import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useParams, useNavigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './hooks/useAuth';
import { AuthPage } from './components/Auth/AuthPage';
import { Dashboard } from './components/Dashboard/Dashboard';
import { AlgorithmExplainer } from './components/Algorithms/AlgorithmExplainer';
import { DatasetUpload } from './components/DatasetUpload/DatasetUpload';
import { AboutML } from './components/Pages/AboutML';
import { Contact } from './components/Pages/Contact';
import { Header } from './components/Layout/Header';
import { Footer } from './components/Layout/Footer';
import { algorithms } from './data/algorithms';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return user ? <>{children}</> : <Navigate to="/auth" replace />;
};

const AlgorithmRoute: React.FC = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const algorithm = algorithms.find(algo => algo.id === id);
  
  if (!algorithm) {
    return <Navigate to="/" replace />;
  }
  
  return (
    <AlgorithmExplainer 
      algorithm={algorithm} 
      onBack={() => navigate('/')}
    />
  );
};

const AppContent: React.FC = () => {
  const { user } = useAuth();

  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Routes>
          <Route path="/auth" element={!user ? <AuthPage /> : <Navigate to="/" replace />} />
          
          <Route path="/*" element={
            <ProtectedRoute>
              <Header />
              <div className="flex-1">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/algorithm/:id" element={<AlgorithmRoute />} />
                  <Route path="/dataset" element={<DatasetUpload />} />
                  <Route path="/about" element={<AboutML />} />
                  <Route path="/contact" element={<Contact />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </div>
              <Footer />
            </ProtectedRoute>
          } />
        </Routes>
      </div>
    </Router>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;