import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { dashboardApi, camerasApi, alertsApi } from '../api/api';
import { 
  FaVideo, 
  FaBell, 
  FaExclamationTriangle, 
  FaCheckCircle, 
  FaHardHat, 
  FaEye, 
  FaShieldAlt, 
  FaClock,
  FaCalendarAlt,
  FaArrowUp,
  FaArrowDown,
  FaTimes,
  FaExclamationCircle
} from 'react-icons/fa';
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
  const [timeRange, setTimeRange] = useState('24h'); // Default to 24 hours
  
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
  }, [timeRange]);
  
  // Prepare chart data
  const prepareChartData = () => {
    if (!timeline || !Array.isArray(timeline.timeline) || timeline.timeline.length === 0) return null;

    // Only use valid dates for labels
    const labels = timeline.timeline.map(item => {
      const timeDate = new Date(item.time);
      return isNaN(timeDate.getTime())
        ? ''
        : timeDate.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    });

    // Get all violation types
    const violationTypes = new Set();
    timeline.timeline.forEach(item => {
      Object.keys(item.by_type || {}).forEach(type => {
        violationTypes.add(type);
      });
    });

    // Modern color palette
    const palette = [
      '#10b981', // emerald
      '#6366f1', // indigo
      '#f59e42', // orange
      '#ef4444', // red
      '#3b82f6', // blue
      '#a855f7', // purple
      '#fbbf24', // yellow
    ];

    // Prepare datasets
    const datasets = Array.from(violationTypes).map((type, index) => {
      const color = palette[index % palette.length];
      return {
        label: type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
        data: timeline.timeline.map(item => item.by_type?.[type] || 0),
        borderColor: color,
        backgroundColor: color + '22', // subtle fill
        pointBackgroundColor: color,
        pointBorderColor: '#fff',
        pointRadius: 5,
        pointHoverRadius: 8,
        tension: 0.4,
        borderWidth: 3,
        fill: false,
      };
    });

    // Add total line
    datasets.push({
      label: 'Total',
      data: timeline.timeline.map(item => item.total),
      borderColor: '#10b981',
      backgroundColor: '#10b98122',
      pointBackgroundColor: '#10b981',
      pointBorderColor: '#fff',
      pointRadius: 6,
      pointHoverRadius: 10,
      borderWidth: 4,
      tension: 0.4,
      fill: false,
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
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          padding: 20,
          usePointStyle: true,
          pointStyle: 'circle',
          font: { size: 15, weight: '500' },
        },
      },
      title: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: '#111827',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#10b981',
        borderWidth: 1,
        padding: 12,
        caretSize: 8,
        cornerRadius: 8,
        displayColors: true,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0,0,0,0.06)',
        },
        title: {
          display: true,
          text: 'Number of Alerts',
          font: { size: 13, weight: '500' },
        },
        ticks: {
          color: '#6b7280',
          font: { size: 13 },
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          color: '#6b7280',
          font: { size: 13 },
          maxRotation: 30,
          minRotation: 0,
          autoSkip: true,
          maxTicksLimit: 10,
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index',
    },
    elements: {
      line: {
        borderJoinStyle: 'round',
      },
    },
  };
  
  // Time range selection handler
  const handleTimeRangeChange = (range) => {
    setTimeRange(range);
    // The useEffect will handle data refetching when timeRange changes
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="text-center p-8 bg-red-50 text-red-700 rounded-xl">
        <FaExclamationTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p>{error}</p>
        <button
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Time range selector */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 leading-tight mb-2">Safety Monitoring Dashboard</h1>
          <p className="text-lg text-gray-600">Real-time Safety and Compliance Monitoring</p>
        </div>
        <div className="flex space-x-2 bg-white p-1 rounded-lg shadow-sm border border-gray-200">
          <button 
            onClick={() => handleTimeRangeChange('24h')} 
            className={`flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
              timeRange === '24h' 
                ? 'bg-primary-50 text-primary-600' 
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <FaClock className="mr-1.5 h-4 w-4" />
            24h
          </button>
          <button 
            onClick={() => handleTimeRangeChange('7d')} 
            className={`flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
              timeRange === '7d' 
                ? 'bg-primary-50 text-primary-600' 
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <FaCalendarAlt className="mr-1.5 h-4 w-4" />
            7d
          </button>
          <button 
            onClick={() => handleTimeRangeChange('30d')} 
            className={`flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
              timeRange === '30d' 
                ? 'bg-primary-50 text-primary-600' 
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <FaCalendarAlt className="mr-1.5 h-4 w-4" />
            30d
          </button>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Cameras card */}
        <section className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-6 flex flex-col gap-3 border border-gray-200" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Cameras</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-bold text-gray-900">
                  {overview?.cameras?.monitoring || 0} / {overview?.cameras?.total || 0}
                </p>
                <p className="ml-2 text-sm text-gray-500">active</p>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-primary-50">
              <FaVideo className="h-6 w-6 text-primary-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-center text-sm">
              <span className="text-gray-600">Status:</span>
              <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700">
                {overview?.cameras?.status || 'Unknown'}
              </span>
            </div>
          </div>
        </section>
        
        {/* Alerts card */}
        <section className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-6 flex flex-col gap-3 border border-gray-200" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Alerts</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-bold text-gray-900">
                  {overview?.alerts?.total || 0}
                </p>
                <p className="ml-2 text-sm text-gray-500">total</p>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-red-50">
              <FaBell className="h-6 w-6 text-red-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-center text-sm">
              <span className="text-gray-600">Active:</span>
              <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-50 text-yellow-700">
                {overview?.alerts?.active || 0} alerts
              </span>
            </div>
          </div>
        </section>
        
        {/* Compliance card */}
        <section className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-6 flex flex-col gap-3 border border-gray-200" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Compliance</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-3xl font-bold text-gray-900">
                  {overview?.compliance?.rate || 0}%
                </p>
                <p className="ml-2 text-sm text-gray-500">rate</p>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-emerald-50">
              <FaHardHat className="h-6 w-6 text-emerald-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-center text-sm">
              <span className="text-gray-600">Trend:</span>
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-medium ${
                overview?.compliance?.trend > 0 
                  ? 'bg-green-50 text-green-700' 
                  : 'bg-red-50 text-red-700'
              }`}>
                {overview?.compliance?.trend > 0 ? '↑' : '↓'} {Math.abs(overview?.compliance?.trend || 0)}%
              </span>
            </div>
          </div>
        </section>
      </div>
      
      {/* Chart section */}
      <section className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Alert Trends</h2>
          <p className="text-sm text-gray-600">Violation trends over time</p>
        </div>
        <div className="h-[400px] md:h-[420px] w-full flex items-center justify-center">
          {chartData && chartData.labels.filter(Boolean).length > 0 && chartData.datasets.some(ds => ds.data.some(val => val > 0)) ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="text-center text-gray-400 text-lg font-medium">No alert trend data available</div>
          )}
        </div>
      </section>

      {/* Recent alerts section */}
      <section className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Recent Alerts</h2>
            <p className="text-sm text-gray-600">Latest safety violations</p>
          </div>
          <Link 
            to="/alerts" 
            className="text-sm font-medium text-primary-600 hover:text-primary-700 flex items-center gap-1"
          >
            View all
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
          </Link>
        </div>
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div 
              key={alert.id} 
              className="flex items-center justify-between p-4 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className={`p-2 rounded-lg ${
                  alert.severity === 'high' ? 'bg-red-50' :
                  alert.severity === 'medium' ? 'bg-yellow-50' :
                  'bg-blue-50'
                }`}>
                  <FaExclamationTriangle className={`h-5 w-5 ${
                    alert.severity === 'high' ? 'text-red-600' :
                    alert.severity === 'medium' ? 'text-yellow-600' :
                    'text-blue-600'
                  }`} />
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-900">{alert.type}</h3>
                  <p className="text-sm text-gray-600">{alert.location}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  alert.severity === 'high' ? 'bg-red-50 text-red-700' :
                  alert.severity === 'medium' ? 'bg-yellow-50 text-yellow-700' :
                  'bg-blue-50 text-blue-700'
                }`}>
                  {alert.severity}
                </span>
                <span className="text-sm text-gray-500">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>
      
      {/* Camera previews */}
      <section className="bg-white rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 p-6 flex flex-col gap-3 border border-green-main" aria-label="Dashboard Card">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-medium text-gray-900">Camera Previews</h2>
          <Link to="/cameras" className="text-xs text-green-main hover:text-green-main flex items-center">
            <span>Manage cameras</span>
            <svg className="w-4 h-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </Link>
        </div>
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
              className="text-sm font-medium text-green-main hover:underline transition-colors"
            >
              View all cameras
            </Link>
          </div>
        )}
      </section>
    </div>
  );
};

export default Dashboard;