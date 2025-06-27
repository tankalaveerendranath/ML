export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
}

export interface Algorithm {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: string;
  complexity: 'Beginner' | 'Intermediate' | 'Advanced';
  useCase: string;
  pros: string[];
  cons: string[];
  steps: AlgorithmStep[];
  exampleDataset: ExampleDataset;
  mathematicalFormulas: MathematicalFormula[];
}

export interface AlgorithmStep {
  id: number;
  title: string;
  description: string;
  animation: string;
  code?: string;
  mathematicalStep?: string;
  exampleCalculation?: string;
}

export interface ExampleDataset {
  name: string;
  description: string;
  features: string[];
  data: number[][];
  target: number[] | string[];
  featureNames: string[];
}

export interface MathematicalFormula {
  name: string;
  formula: string;
  description: string;
  variables: { [key: string]: string };
}

export interface Dataset {
  id: string;
  name: string;
  size: number;
  features: number;
  samples: number;
  type: 'numerical' | 'categorical' | 'mixed';
  target: 'classification' | 'regression' | 'clustering';
  uploadedAt: string;
  userId: string;
}

export interface DatasetRecommendation {
  algorithm: string;
  confidence: number;
  reasoning: string;
}

export interface HistoryEntry {
  id: string;
  userId: string;
  dataset: Dataset;
  recommendations: DatasetRecommendation[];
  createdAt: string;
}

export interface CalculationStep {
  step: number;
  description: string;
  formula: string;
  calculation: string;
  result: string | number;
}