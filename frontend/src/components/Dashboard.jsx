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
    if (!timeline) return null;
    
    const labels = timeline.timeline.map(item => {
      // Format the time string for better readability
      const timeDate = new Date(item.time);
      return timeDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });
    
    // Get all violation types
    const violationTypes = new Set();
    timeline.timeline.forEach(item => {
      Object.keys(item.by_type || {}).forEach(type => {
        violationTypes.add(type);
      });
    });
    
    // Prepare datasets
    const datasets = Array.from(violationTypes).map((type, index) => {
      // Generate color based on violation type for consistency
      const violationColors = {
        no_hardhat: 'rgb(235, 87, 87)',
        no_vest: 'rgb(242, 153, 74)',
        no_gloves: 'rgb(168, 92, 249)',
        no_goggles: 'rgb(106, 171, 218)',
        no_boots: 'rgb(134, 207, 59)',
      };
      
      const color = violationColors[type] || `hsl(${(index * 137) % 360}, 70%, 50%)`;
      
      return {
        label: type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
        data: timeline.timeline.map(item => item.by_type?.[type] || 0),
        borderColor: color,
        backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
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
  
  // Time range selection handler
  const handleTimeRangeChange = (range) => {
    setTimeRange(range);
    // The useEffect will handle data refetching when timeRange changes
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-main mx-auto"></div>
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
      {/* Time range selector */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-extrabold text-green-main leading-tight mb-1">Safety Monitoring Dashboard</h1>
<p className="text-lg text-gray-700 mb-6">Real-time PPE compliance monitoring</p>
        </div>
        <div className="flex space-x-2 bg-green-main text-white p-1 rounded-xl shadow-sm">
          <button 
            onClick={() => handleTimeRangeChange('24h')} 
            className={`flex items-center px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${timeRange === '24h' ? 'bg-green-main text-white shadow' : 'text-green-main bg-white border border-green-main hover:bg-green-main/10'}`}
          >
            <FaClock className="mr-1.5 h-3.5 w-3.5" />
            24h
          </button>
          <button 
            onClick={() => handleTimeRangeChange('7d')} 
            className={`flex items-center px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${timeRange === '7d' ? 'bg-green-main text-white shadow' : 'text-green-main bg-white border border-green-main hover:bg-green-main/10'}`}
          >
            <FaCalendarAlt className="mr-1.5 h-3.5 w-3.5" />
            7d
          </button>
          <button 
            onClick={() => handleTimeRangeChange('30d')} 
            className={`flex items-center px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200 ${timeRange === '30d' ? 'bg-green-main text-white shadow' : 'text-green-main bg-white border border-green-main hover:bg-green-main/10'}`}
          >
            <FaCalendarAlt className="mr-1.5 h-3.5 w-3.5" />
            30d
          </button>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Cameras card */}
        <section className="bg-white rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 p-6 flex flex-col gap-3 border border-green-main" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-semibold text-green-main">Cameras</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-4xl font-bold text-green-main">
                  {overview?.cameras?.monitoring || 0} / {overview?.cameras?.total || 0}
                </p>
                <p className="ml-2 text-sm text-green-main">active</p>
              </div>
            </div>
            <div className="rounded-full bg-green-main text-white p-3 shadow-lg">
              <FaVideo className="h-7 w-7 text-white" />
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Stream Health</span>
              <div className="flex items-center">
                <span className="inline-block w-2 h-2 rounded-full bg-success-500 mr-1"></span>
                <span className="text-xs font-medium text-success-600">All Online</span>
              </div>
            </div>
          </div>
          <div className="mt-4">
            <Link
              to="/cameras"
              className="text-sm font-bold text-white bg-green-main text-white hover:bg-green-main text-white px-4 py-2 rounded-lg shadow transition-colors flex items-center"
            >
              View all cameras
              <svg className="w-4 h-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </Link>
          </div>
        </section>
        
        {/* Alerts card */}
        <section className="bg-white rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 p-6 flex flex-col gap-3 border border-green-main" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-semibold text-green-main">Recent Alerts</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-4xl font-bold text-green-main">
                  {overview?.alerts?.total || 0}
                </p>
                <p className="ml-2 text-sm text-green-main">
                  in the last {overview?.time_range_hours || 24}h
                </p>
              </div>
            </div>
            <div className="rounded-full bg-danger-600 p-3 shadow-lg">
              <FaBell className="h-7 w-7 text-white" />
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">Recent Activity</span>
              {(overview?.alerts?.total || 0) > 0 ? (
                <div className="flex items-center">
                  <FaArrowUp className="h-3 w-3 text-danger-500 mr-1" />
                  <span className="text-xs font-medium text-danger-600">Alert trending</span>
                </div>
              ) : (
                <div className="flex items-center">
                  <FaArrowDown className="h-3 w-3 text-success-500 mr-1" />
                  <span className="text-xs font-medium text-success-600">No recent alerts</span>
                </div>
              )}
            </div>
          </div>
          <div className="mt-4">
            <Link
              to="/alerts"
              className="text-sm font-bold text-white bg-green-main text-white hover:bg-green-main text-white px-4 py-2 rounded-lg shadow transition-colors flex items-center"
            >
              View all alerts
              <svg className="w-4 h-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </Link>
          </div>
        </section>
        
        {/* Compliance score card */}
        <section className="bg-white rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 p-6 flex flex-col gap-3 border border-green-main" aria-label="Dashboard Card">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-semibold text-green-main">Compliance Score</p>
              <div className="mt-2 flex items-baseline">
                <p className="text-4xl font-bold text-white">
                  {overview?.compliance_score || 'N/A'}%
                </p>
              </div>
            </div>
            <div className={`rounded-full p-3 ${
              (overview?.compliance_score || 0) > 80 
                ? 'bg-success-600' 
                : (overview?.compliance_score || 0) > 50 
                  ? 'bg-warning-600' 
                  : 'bg-danger-600'
            }`}>
              {(overview?.compliance_score || 0) > 80 ? (
                <FaCheckCircle className="h-6 w-6 text-success-600" />
              ) : (overview?.compliance_score || 0) > 50 ? (
                <FaExclamationCircle className="h-6 w-6 text-warning-600" />
              ) : (
                <FaTimes className="h-7 w-7 text-white" />
              )}
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-100">
            {overview?.alerts?.by_type && (
              <div className="space-y-2">
                <p className="text-sm font-semibold text-green-main">Violation Breakdown:</p>
                <div className="space-y-2">
                  {Object.entries(overview.alerts.by_type).map(([type, count]) => {
                    const iconMap = {
                      no_hardhat: <FaHardHat className="h-3.5 w-3.5 mr-2" />,
                      no_vest: <FaShieldAlt className="h-3.5 w-3.5 mr-2" />,
                      no_goggles: <FaEye className="h-3.5 w-3.5 mr-2" />
                    };
                    
                    return (
                      <div key={type} className="flex justify-between items-center text-sm bg-surface-light rounded-md p-2">
                        <div className="flex items-center">
                          {iconMap[type] || <FaExclamationTriangle className="h-3.5 w-3.5 mr-2" />}
                          <span>{type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                        </div>
                        <span className="font-medium bg-surface px-2 py-0.5 rounded-full text-xs">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
      
      {/* Chart and recent alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <section className="bg-white rounded-2xl shadow-2xl p-8 flex flex-col gap-4 border border-green-main lg:col-span-2" aria-label="Alert Trends Chart">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-gray-900">Alert Trends</h2>
            <div className="flex items-center text-xs text-gray-500">
              <div className="w-3 h-3 rounded-full bg-green-main text-white mr-1"></div>
              <span className="mr-3">Total</span>
              {Array.from(new Set(timeline?.timeline.flatMap(item => Object.keys(item.by_type || {})) || [])).slice(0, 2).map((type, index) => (
                <div key={type} className="flex items-center ml-2">
                  <div className="w-3 h-3 rounded-full mr-1" style={{
                    backgroundColor: [`rgb(235, 87, 87)`, `rgb(106, 171, 218)`][index] || `hsl(${(index * 137) % 360}, 70%, 50%)`
                  }}></div>
                  <span>{type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                </div>
              ))}
            </div>
          </div>
          {chartData ? (
            <div className="h-80">
              <Line data={chartData} options={{
                ...chartOptions,
                plugins: {
                  legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                      color: '#2B3933',
                      font: { size: 16, weight: 'bold' },
                      usePointStyle: true,
                      boxWidth: 16,
                      padding: 20,
                    },
                  },
                },
                scales: {
                  x: {
                    grid: { display: false },
                    ticks: { color: '#2B3933', font: { weight: 'bold', size: 14 } },
                  },
                  y: {
                    grid: { color: '#F4F5F2', lineWidth: 2 },
                    ticks: { color: '#2B3933', font: { weight: 'bold', size: 14 } },
                  },
                },
                elements: {
                  line: {
                    borderWidth: 4,
                    tension: 0.5,
                  },
                  point: {
                    radius: 7,
                    backgroundColor: '#65704F',
                    borderColor: '#fff',
                    borderWidth: 3,
                    hoverRadius: 10,
                  },
                },
                animation: { duration: 1000, easing: 'easeOutQuart' },
                backgroundColor: '#fff',
              }} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 bg-surface-light rounded-lg">
              <p className="text-gray-500">No trend data available</p>
            </div>
          )}
        </section>
        
        {/* Recent alerts */}
        <section className="bg-white rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 p-6 flex flex-col gap-3 border border-green-main" aria-label="Dashboard Card">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-gray-900">Recent Alerts</h2>
            <Link to="/alerts" className="text-xs text-green-main hover:underline transition-colors">View all</Link>
          </div>
          <div className="space-y-3">
            {alerts.length > 0 ? (
              alerts.map(alert => (
                <Link
                  key={alert.id}
                  to={`/alerts/${alert.id}`}
                  className="block transition-all hover:translate-x-1 duration-200"
                >
                  <div className="border border-gray-200 rounded-lg p-3 hover:bg-surface-light transition-colors group">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-3">
                        <div className="bg-surface-light p-2 rounded-full group-hover:bg-green-main text-white transition-colors">
                          {alert.violation_type === 'no_hardhat' ? (
                            <FaHardHat className="h-4 w-4 text-danger-600" />
                          ) : alert.violation_type === 'no_vest' ? (
                            <FaShieldAlt className="h-4 w-4 text-danger-600" />
                          ) : alert.violation_type === 'no_goggles' ? (
                            <FaEye className="h-4 w-4 text-danger-600" />
                          ) : (
                            <FaExclamationCircle className="h-4 w-4 text-danger-600" />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900 group-hover:text-green-main transition-colors">
                            {alert.camera_name || 'Unknown camera'}
                          </p>
                          <div className="flex items-center text-xs text-gray-500">
                            <FaClock className="h-3 w-3 mr-1" />
                            {new Date(alert.created_at || alert.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </div>
                      </div>
                      <span className="px-2 py-1 text-xs font-medium rounded-full bg-danger-100 text-danger-800 group-hover:bg-danger-200 transition-colors">
                        {alert.violation_type ? alert.violation_type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') : 'Unknown'}
                      </span>
                    </div>
                  </div>
                </Link>
              ))
            ) : (
              <div className="text-center py-8 bg-surface-light rounded-lg">
                <FaTimes className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500">No recent alerts</p>
              </div>
            )}
          </div>
        </section>
      </div>
      
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