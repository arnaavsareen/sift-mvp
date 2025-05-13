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
  FaExclamationCircle,
  FaPlus,
  FaCamera
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
        
        // Calculate hours based on time range
        const hours = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720; // 24h, 7d or 30d
        const intervalMinutes = timeRange === '24h' ? 60 : timeRange === '7d' ? 360 : 1440; // Adjust interval based on range
        
        // Fetch overview data
        const overviewData = await dashboardApi.getOverview({ hours });
        setOverview(overviewData);
        
        // Fetch cameras
        const camerasData = await camerasApi.getAll();
        setCameras(camerasData);
        
        // Fetch recent alerts
        const alertsData = await alertsApi.getAll({ limit: 5 });
        setAlerts(alertsData);
        
        // Fetch timeline data with appropriate interval
        const timelineData = await dashboardApi.getTimeline({ 
          hours, 
          interval_minutes: intervalMinutes 
        });
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
      // Use timestamp if available, otherwise time
      const timeString = item.timestamp || item.time;
      const timeDate = new Date(timeString);
      return isNaN(timeDate.getTime())
        ? ''
        : timeDate.toLocaleString([], { 
            month: 'short', 
            day: 'numeric', 
            hour: timeRange === '24h' ? '2-digit' : undefined, 
            minute: timeRange === '24h' ? '2-digit' : undefined 
          });
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
      '#ef4444', // red
      '#f59e42', // orange
      '#fbbf24', // yellow
      '#10b981', // emerald
      '#3b82f6', // blue
      '#6366f1', // indigo
      '#a855f7', // purple
    ];

    // Prepare datasets
    const datasets = Array.from(violationTypes).map((type, index) => {
      const color = palette[index % palette.length];
      return {
        label: type.replace(/_/g, ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
        data: timeline.timeline.map(item => item.by_type?.[type] || 0),
        borderColor: color,
        backgroundColor: color + '22', // subtle fill
        pointBackgroundColor: color,
        pointBorderColor: '#fff',
        pointRadius: 3,
        pointHoverRadius: 6,
        tension: 0.4,
        borderWidth: 2,
        fill: true,
      };
    });

    // Add total line if there are multiple violation types
    if (violationTypes.size > 1) {
      datasets.push({
        label: 'Total',
        data: timeline.timeline.map(item => item.total),
        borderColor: '#10b981',
        backgroundColor: 'transparent',
        pointBackgroundColor: '#10b981',
        pointBorderColor: '#fff',
        pointRadius: 4,
        pointHoverRadius: 8,
        borderWidth: 3,
        tension: 0.4,
        fill: false,
        borderDash: violationTypes.size > 0 ? [5, 5] : undefined,
      });
    }

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
  
  // Helper to get camera name
  const getCameraName = (cameraId) => {
    const camera = cameras.find(c => c.id === cameraId);
    return camera ? camera.name : `Camera ${cameraId}`;
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
              <span className="text-gray-600">Status:</span>
              {overview?.alerts?.active > 0 ? (
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700 flex items-center">
                  <span className="w-2 h-2 bg-red-500 rounded-full mr-1.5 animate-pulse"></span>
                  {overview?.alerts?.active} Active
                </span>
              ) : (
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 flex items-center">
                  <span className="w-2 h-2 bg-green-500 rounded-full mr-1.5"></span>
                  All Clear
                </span>
              )}
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
                  {overview?.compliance_score ? Math.round(overview.compliance_score) : 0}%
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
              {typeof overview?.compliance_score === 'number' ? (
                <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-medium ${
                  (overview?.alerts?.total || 0) === 0 
                    ? 'bg-gray-50 text-gray-700' 
                    : overview?.compliance_score >= 90
                    ? 'bg-green-50 text-green-700'
                    : overview?.compliance_score >= 70
                    ? 'bg-yellow-50 text-yellow-700'
                    : 'bg-red-50 text-red-700'
                }`}>
                  {(overview?.alerts?.total || 0) === 0 
                    ? 'No data'
                    : overview?.compliance_score >= 90
                    ? 'Excellent'
                    : overview?.compliance_score >= 70
                    ? 'Average'
                    : 'Need attention'
                  }
                </span>
              ) : (
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-medium bg-gray-50 text-gray-700">
                  Unknown
                </span>
              )}
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

      {/* Camera previews */}
      <section className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Camera Monitoring</h2>
            <p className="text-sm text-gray-600">Active surveillance cameras</p>
          </div>
          <Link 
            to="/cameras" 
            className="text-sm font-medium text-primary-600 hover:text-primary-700 flex items-center gap-1"
          >
            View all
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
          </Link>
        </div>
        
        {cameras.length === 0 ? (
          <div className="text-center py-12 border-2 border-dashed border-gray-200 rounded-lg">
            <FaVideo className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No cameras configured</h3>
            <p className="text-sm text-gray-500 mb-4 max-w-md mx-auto">
              Add cameras to begin monitoring for safety violations
            </p>
            <Link 
              to="/cameras/add" 
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
            >
              <FaPlus className="mr-2 h-4 w-4" />
              Add Camera
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {cameras.slice(0, 8).map((camera) => (
              <Link
                to={`/cameras/${camera.id}`}
                key={camera.id}
                className="block group"
              >
                <div className="bg-gray-800 rounded-lg overflow-hidden shadow-sm border border-gray-200 transition-all duration-200 group-hover:shadow-md">
                  <div className="aspect-video relative bg-gray-900 overflow-hidden">
                    {/* Camera preview image */}
                    <img
                      src={`${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/dashboard/cameras/${camera.id}/latest-frame?format=jpeg&t=${new Date().getTime()}`}
                      alt={camera.name}
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    
                    {/* Camera details overlay */}
                    <div className="absolute inset-0 flex flex-col justify-between p-4 bg-gradient-to-t from-black/70 to-transparent">
                      <div>
                        <div className="flex items-center space-x-2">
                          <div className={`h-2 w-2 rounded-full ${camera.is_active ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                          <span className="text-white text-xs font-medium">
                            {camera.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </div>
                      </div>
                      <div>
                        <h3 className="text-white font-medium truncate">
                          {camera.name}
                        </h3>
                        {camera.location && (
                          <p className="text-gray-300 text-xs truncate">
                            {camera.location}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
        
        {cameras.length > 8 && (
          <div className="mt-4 text-center">
            <Link 
              to="/cameras" 
              className="text-sm font-medium text-primary-600 hover:text-primary-700"
            >
              View all {cameras.length} cameras
            </Link>
          </div>
        )}
      </section>
      
      {/* Recent alerts section with screenshots */}
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
        
        {alerts.length === 0 ? (
          <div className="text-center py-12 border-2 border-dashed border-gray-200 rounded-lg">
            <FaBell className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No recent alerts</h3>
            <p className="text-sm text-gray-500 max-w-md mx-auto">
              Safety violation alerts will appear here when detected
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {alerts.map((alert) => (
              <Link 
                key={alert.id} 
                to={`/alerts/${alert.id}`}
                className="block rounded-lg overflow-hidden shadow-sm border border-gray-200 hover:shadow-md transition-all duration-200"
              >
                <div className="flex flex-col h-full">
                  {alert.screenshot_path ? (
                    <div className="relative bg-gray-900">
                      <img
                        src={`${process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:8000'}${alert.screenshot_path.startsWith('/') ? '' : '/screenshots/'}${alert.screenshot_path}`}
                        alt={`${alert.violation_type.replace(/_/g, ' ')} violation`}
                        className="w-full h-48 object-cover"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSIyMDAiIGZpbGw9IiNFNUU3RUIiLz48cGF0aCBkPSJNMTAwIDcwQzEwMCA4MS4wNDU3IDkxLjA0NTcgOTAgODAgOTBDNjguOTU0MyA5MCA2MCA4MS4wNDU3IDYwIDcwQzYwIDU4Ljk1NDMgNjguOTU0MyA1MCA4MCA1MEM5MS4wNDU3IDUwIDEwMCA1OC45NTQzIDEwMCA3MFoiIGZpbGw9IiNBMUExQUEiLz48cGF0aCBkPSJNMTQwIDEzMEMxNDAgMTUyLjA5MSAxMjIuMDkxIDE3MCAxMDAgMTcwQzc3LjkwODYgMTcwIDYwIDE1Mi4wOTEgNjAgMTMwQzYwIDEwNy45MDkgNzcuOTA4NiA5MCAxMDAgOTBDMTIyLjA5MSA5MCAxNDAgMTA3LjkwOSAxNDAgMTMwWiIgZmlsbD0iI0ExQTFBQSIvPjwvc3ZnPg==';
                        }}
                      />
                      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/70 to-transparent text-white p-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <FaExclamationTriangle className="text-yellow-400" />
                            <span className="font-medium text-sm">
                              {alert.violation_type.replace(/_/g, ' ').toUpperCase()}
                            </span>
                          </div>
                          <div className="flex items-center gap-1 bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                            <span>{Math.round(alert.confidence * 100)}%</span>
                          </div>
                        </div>
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent text-white p-3">
                        <div className="flex items-center justify-between">
                          <div className="text-xs">
                            {new Date(alert.created_at).toLocaleString([], {
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </div>
                          <div className="flex items-center gap-1">
                            <FaCamera className="h-3 w-3" />
                            <span className="text-xs">{getCameraName(alert.camera_id)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-gray-100 h-48 flex items-center justify-center">
                      <div className="text-center">
                        <FaBell className="h-10 w-10 text-gray-300 mx-auto mb-2" />
                        <p className="text-sm text-gray-500">No image available</p>
                      </div>
                    </div>
                  )}
                  <div className="p-3 bg-white flex-grow flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <FaBell className="h-4 w-4 text-danger-500 mr-2 flex-shrink-0" />
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {alert.violation_type.replace(/_/g, " ")}
                          </p>
                        </div>
                        <div>
                          <span className={`px-2 py-0.5 text-xs rounded-full ${alert.resolved ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                            {alert.resolved ? 'Resolved' : 'Active'}
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">
                        {(() => {
                          const type = alert.violation_type.toLowerCase();
                          if (type.includes('hardhat') && type.includes('vest')) {
                            return 'Worker without hard hat and safety vest';
                          } else if (type.includes('hardhat')) {
                            return 'Worker without required hard hat';
                          } else if (type.includes('vest')) {
                            return 'Worker without required safety vest';
                          } else {
                            return 'Safety violation detected';
                          }
                        })()}
                      </p>
                    </div>
                    <div className="mt-3 pt-2 border-t border-gray-100 flex items-center justify-between">
                      <p className="text-xs text-gray-500">
                        {new Date(alert.created_at).toLocaleString()}
                      </p>
                      <p className="text-xs text-primary-600 font-medium flex items-center">
                        <FaEye className="h-3 w-3 mr-1" />
                        View Details
                      </p>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default Dashboard;