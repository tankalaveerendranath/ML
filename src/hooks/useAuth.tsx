import { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import { User } from '../types';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<{ success: boolean; message?: string }>;
  signup: (name: string, email: string, password: string) => Promise<{ success: boolean; message?: string }>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for stored user session
    const storedUser = localStorage.getItem('ml-website-user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (error) {
        localStorage.removeItem('ml-website-user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string): Promise<{ success: boolean; message?: string }> => {
    setIsLoading(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Get stored users
    const storedUsers = JSON.parse(localStorage.getItem('ml-website-users') || '[]');
    
    // Find user with matching email and password
    const existingUser = storedUsers.find((u: any) => u.email === email && u.password === password);
    
    if (existingUser) {
      const user: User = {
        id: existingUser.id,
        name: existingUser.name,
        email: existingUser.email,
        createdAt: existingUser.createdAt
      };
      setUser(user);
      localStorage.setItem('ml-website-user', JSON.stringify(user));
      setIsLoading(false);
      return { success: true };
    }
    
    setIsLoading(false);
    return { success: false, message: 'Invalid email or password' };
  };

  const signup = async (name: string, email: string, password: string): Promise<{ success: boolean; message?: string }> => {
    setIsLoading(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Get stored users
    const storedUsers = JSON.parse(localStorage.getItem('ml-website-users') || '[]');
    
    // Check if user already exists
    const existingUser = storedUsers.find((u: any) => u.email === email);
    if (existingUser) {
      setIsLoading(false);
      return { success: false, message: 'User with this email already exists' };
    }
    
    // Create new user
    const newUser = {
      id: Date.now().toString(),
      name,
      email,
      password,
      createdAt: new Date().toISOString()
    };
    
    // Store user
    storedUsers.push(newUser);
    localStorage.setItem('ml-website-users', JSON.stringify(storedUsers));
    
    // Auto-login after signup
    const user: User = {
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      createdAt: newUser.createdAt
    };
    setUser(user);
    localStorage.setItem('ml-website-user', JSON.stringify(user));
    
    setIsLoading(false);
    return { success: true };
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('ml-website-user');
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};