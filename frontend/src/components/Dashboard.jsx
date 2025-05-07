import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { dashboardApi, camerasApi, alertsApi } from '../api/api';
import { FaVideo, FaBell, FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [overview, setOverview] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [timeline, setTimeline] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch dashboard data
  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setLoading(true);
        
        // Fetch overview data
        const overviewData = await dashboardApi.getOverview();
        setOverview(overviewData);
        
        // Fetch cameras
        const camerasData = await camerasApi.getAll();
        setCameras(camerasData);
        
        // Fetch recent alerts
        const alertsData = await alertsApi.getAll({ limit: 5 });
        setAlerts(alertsData);
        
        // Fetch timeline data
        const timelineData = await dashboardApi.getTimeline();
        setTimeline(timelineData);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchDashboard();
    
    // Refresh data every 30 seconds
    const interval = setInterval(() => {
      fetchDashboard();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Prepare chart data
  const prepareChartData = () => {
    if (!timeline) return null;
    
    const labels = timeline.timeline.map(item => item.time);
    
    // Get all violation types
    const violationTypes = new Set();
    timeline.timeline.forEach(item => {
      Object.keys(item.by_type || {}).forEach(type => {
        violationTypes.add(type);
      });
    });
    
    // Prepare datasets
    const datasets = Array.from(violationTypes).map((type, index) => {
      // Generate color based on index
      const hue = (index * 137) % 360;
      const color = `hsl(${hue}, 70%, 50%)`;
      
      return {
        label: type,
        data: timeline.timeline.map(item => item.by_type?.[type] || 0),
        borderColor: color,
        backgroundColor: `hsla(${hue}, 70%, 50%, 0.2)`,
        tension: 0.3,
      };
    });
    
    // Add total line
    datasets.push({
      label: 'Total',
      data: timeline.timeline.map(item => item.total),
      borderColor: 'rgb(75, 192, 192)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderWidth: 2,
      tension: 0.3,
    });
    
    return {
      labels,
      datasets,
    };
  };
  
  const chartData = prepareChartData();
  
  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Alert Trends',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Alerts',
        },
      },
    },
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="text-center p-6 bg-danger-50 text-danger-700 rounded-lg">
        <FaExclamationTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p>{error}</p>
        <button
          className="mt-4 btn-primary"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Cameras card */}
        <div className="card p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Cameras</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-semibold text-gray-900">
                  {overview?.cameras?.monitoring || 0} / {overview?.cameras?.total || 0}
                </p>
                <p className="ml-2 text-sm text-gray-600">active</p>
              </div>
            </div>
            <div className="rounded-full bg-primary-100 p-3">
              <FaVideo className="h-6 w-6 text-primary-600" />
            </div>
          </div>
          <div className="mt-6">
            <Link
              to="/cameras"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all cameras
            </Link>
          </div>
        </div>
        
        {/* Alerts card */}
        <div className="card p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Recent Alerts</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-semibold text-gray-900">
                  {overview?.alerts?.total || 0}
                </p>
                <p className="ml-2 text-sm text-gray-600">
                  in the last {overview?.time_range_hours || 24}h
                </p>
              </div>
            </div>
            <div className="rounded-full bg-danger-100 p-3">
              <FaBell className="h-6 w-6 text-danger-600" />
            </div>
          </div>
          <div className="mt-6">
            <Link
              to="/alerts"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all alerts
            </Link>
          </div>
        </div>
        
        {/* Compliance score card */}
        <div className="card p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Compliance Score</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-semibold text-gray-900">
                  {overview?.compliance_score || 'N/A'}%
                </p>
              </div>
            </div>
            <div className={`rounded-full p-3 ${
              (overview?.compliance_score || 0) > 80 
                ? 'bg-success-100' 
                : (overview?.compliance_score || 0) > 50 
                  ? 'bg-warning-100' 
                  : 'bg-danger-100'
            }`}>
              <FaCheckCircle className={`h-6 w-6 ${
                (overview?.compliance_score || 0) > 80 
                  ? 'text-success-600' 
                  : (overview?.compliance_score || 0) > 50 
                    ? 'text-warning-600' 
                    : 'text-danger-600'
              }`} />
            </div>
          </div>
          <div className="mt-4">
            {overview?.alerts?.by_type && (
              <div className="space-y-2">
                <p className="text-sm font-medium text-gray-600">Violation Types:</p>
                <div className="space-y-1">
                  {Object.entries(overview.alerts.by_type).map(([type, count]) => (
                    <div key={type} className="flex justify-between text-sm">
                      <span>{type.replace('_', ' ')}</span>
                      <span className="font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Chart and recent alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="card p-6 lg:col-span-2">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Alert Trends</h2>
          {chartData ? (
            <div className="h-64">
              <Line data={chartData} options={chartOptions} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
              <p className="text-gray-500">No trend data available</p>
            </div>
          )}
        </div>
        
        {/* Recent alerts */}
        <div className="card p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Alerts</h2>
          <div className="space-y-4">
            {alerts.length > 0 ? (
              alerts.map(alert => (
                <Link
                  key={alert.id}
                  to={`/alerts/${alert.id}`}
                  className="block"
                >
                  <div className="border border-gray-200 rounded-md p-3 hover:bg-gray-50">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-gray-900">
                          {alert.violation_type.replace('_', ' ')}
                        </p>
                        <p className="text-sm text-gray-500">
                          Camera ID: {alert.camera_id}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${
                          alert.resolved
                            ? 'bg-success-100 text-success-800'
                            : 'bg-danger-100 text-danger-800'
                        }`}
                      >
                        {alert.resolved ? 'Resolved' : 'Active'}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      {new Date(alert.created_at).toLocaleString()}
                    </p>
                  </div>
                </Link>
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No recent alerts</p>
            )}
            
            {alerts.length > 0 && (
              <div className="mt-2 text-center">
                <Link
                  to="/alerts"
                  className="text-sm font-medium text-primary-600 hover:text-primary-500"
                >
                  View all alerts
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Camera previews */}
      <div className="card p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Camera Previews</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {cameras.length > 0 ? (
            cameras.slice(0, 3).map(camera => (
              <Link
                key={camera.id}
                to={`/cameras/${camera.id}`}
                className="block"
              >
                <div className="border border-gray-200 rounded-md p-2 hover:bg-gray-50">
                  <div className="aspect-video bg-gray-200 rounded-md flex items-center justify-center mb-2">
                    <FaVideo className="h-8 w-8 text-gray-400" />
                  </div>
                  <div className="px-2">
                    <p className="font-medium text-gray-900 truncate">
                      {camera.name}
                    </p>
                    <p className="text-xs text-gray-500 truncate">
                      {camera.location || 'No location set'}
                    </p>
                    <div className="flex justify-between items-center mt-2">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${
                          camera.is_active
                            ? 'bg-success-100 text-success-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {camera.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <div className="col-span-3 text-center py-8">
              <p className="text-gray-500 mb-4">No cameras configured</p>
              <Link to="/cameras" className="btn-primary">
                Add Camera
              </Link>
            </div>
          )}
        </div>
        
        {cameras.length > 3 && (
          <div className="mt-4 text-center">
            <Link
              to="/cameras"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all cameras
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;