import React, { useState, useEffect } from 'react';

interface Point {
  x: number;
  y: number;
  cluster: number;
  id: number;
}

interface Centroid {
  x: number;
  y: number;
  id: number;
}

interface KMeansVizProps {
  isActive: boolean;
  step: number;
}

export const KMeansViz: React.FC<KMeansVizProps> = ({ isActive, step }) => {
  const [points] = useState<Point[]>([
    { x: 60, y: 80, cluster: -1, id: 1 }, { x: 80, y: 60, cluster: -1, id: 2 },
    { x: 70, y: 90, cluster: -1, id: 3 }, { x: 90, y: 70, cluster: -1, id: 4 },
    { x: 180, y: 160, cluster: -1, id: 5 }, { x: 200, y: 140, cluster: -1, id: 6 },
    { x: 190, y: 170, cluster: -1, id: 7 }, { x: 210, y: 150, cluster: -1, id: 8 },
    { x: 120, y: 200, cluster: -1, id: 9 }, { x: 140, y: 180, cluster: -1, id: 10 },
    { x: 130, y: 210, cluster: -1, id: 11 }, { x: 150, y: 190, cluster: -1, id: 12 }
  ]);

  const [centroids, setCentroids] = useState<Centroid[]>([
    { x: 100, y: 100, id: 0 },
    { x: 200, y: 100, id: 1 },
    { x: 150, y: 200, id: 2 }
  ]);

  const [assignedPoints, setAssignedPoints] = useState<Point[]>(points);
  const [iteration, setIteration] = useState(0);

  const colors = ['#3b82f6', '#ef4444', '#10b981'];

  useEffect(() => {
    if (isActive && step >= 3) {
      const interval = setInterval(() => {
        setIteration(prev => {
          const newIteration = prev + 1;
          
          // Assign points to nearest centroids
          const newAssignedPoints = points.map(point => {
            let minDistance = Infinity;
            let nearestCluster = -1;
            
            centroids.forEach(centroid => {
              const distance = Math.sqrt(
                Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2)
              );
              if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = centroid.id;
              }
            });
            
            return { ...point, cluster: nearestCluster };
          });
          
          setAssignedPoints(newAssignedPoints);
          
          // Update centroids
          if (step >= 4) {
            const newCentroids = centroids.map(centroid => {
              const clusterPoints = newAssignedPoints.filter(p => p.cluster === centroid.id);
              if (clusterPoints.length > 0) {
                const avgX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
                const avgY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
                return { ...centroid, x: avgX, y: avgY };
              }
              return centroid;
            });
            setCentroids(newCentroids);
          }
          
          return newIteration > 8 ? 0 : newIteration;
        });
      }, 1500);
      
      return () => clearInterval(interval);
    }
  }, [isActive, step, points, centroids]);

  return (
    <div className="bg-white rounded-lg p-6 border-2 border-gray-200">
      <h4 className="text-lg font-semibold mb-4 text-center">K-Means Clustering Visualization</h4>
      
      <svg width="300" height="280" className="mx-auto border border-gray-300 rounded">
        {/* Grid background */}
        <defs>
          <pattern id="kmeans-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f5f5f5" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#kmeans-grid)" />
        
        {/* Data points */}
        {assignedPoints.map((point, index) => (
          <circle
            key={point.id}
            cx={point.x}
            cy={point.y}
            r="5"
            fill={point.cluster >= 0 ? colors[point.cluster] : '#6b7280'}
            stroke="#fff"
            strokeWidth="2"
            className={`transition-all duration-500 ${
              isActive && step >= 1 ? 'opacity-100' : 'opacity-0'
            }`}
            style={{ transitionDelay: `${index * 50}ms` }}
          />
        ))}
        
        {/* Centroids */}
        {centroids.map((centroid, index) => (
          <g key={centroid.id}>
            <circle
              cx={centroid.x}
              cy={centroid.y}
              r="8"
              fill={colors[centroid.id]}
              stroke="#000"
              strokeWidth="3"
              className={`transition-all duration-700 ${
                isActive && step >= 2 ? 'opacity-100' : 'opacity-0'
              }`}
              style={{ transitionDelay: `${index * 200}ms` }}
            />
            <text
              x={centroid.x}
              y={centroid.y + 3}
              textAnchor="middle"
              className="text-xs font-bold fill-white"
            >
              C{centroid.id + 1}
            </text>
          </g>
        ))}
        
        {/* Distance lines (when assigning points) */}
        {isActive && step >= 3 && assignedPoints.map(point => {
          const centroid = centroids.find(c => c.id === point.cluster);
          return centroid ? (
            <line
              key={`line-${point.id}`}
              x1={point.x}
              y1={point.y}
              x2={centroid.x}
              y2={centroid.y}
              stroke={colors[point.cluster]}
              strokeWidth="1"
              strokeDasharray="2,2"
              opacity="0.5"
              className="transition-all duration-300"
            />
          ) : null;
        })}
      </svg>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center space-x-4 text-sm mb-2">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>Cluster 1</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
            <span>Cluster 2</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
            <span>Cluster 3</span>
          </div>
        </div>
        <div className="flex justify-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 border-2 border-black rounded-full mr-2"></div>
            <span>Centroids</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-1 bg-gray-400 mr-2" style={{ borderStyle: 'dashed' }}></div>
            <span>Assignments</span>
          </div>
        </div>
        {isActive && step >= 3 && (
          <div className="mt-2 text-xs text-gray-600">
            Iteration: {iteration} | Reassigning points to nearest centroids
          </div>
        )}
      </div>
    </div>
  );
};