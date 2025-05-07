import React, { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { alertsApi, camerasApi } from '../api/api';
import { 
  FaBell, 
  FaExclamationTriangle, 
  FaCheck, 
  FaFilter,
  FaTrash,
  FaEye
} from 'react-icons/fa';

const AlertsList = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [alerts, setAlerts] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  
  // Filter state
  const [filters, setFilters] = useState({
    camera_id: searchParams.get('camera') || '',
    violation_type: searchParams.get('type') || '',
    unresolved_only: searchParams.get('unresolved') === 'true',
    hours: parseInt(searchParams.get('hours') || '24'),
  });
  
  // Pagination state
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const itemsPerPage = 10;
  
  // Load cameras for filter dropdown
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const camerasData = await camerasApi.getAll();
        setCameras(camerasData);
      } catch (err) {
        console.error('Error fetching cameras:', err);
      }
    };
    
    fetchCameras();
  }, []);
  
  // Load alerts and stats
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        setLoading(true);
        
        // Apply filters to API request
        const params = {
          skip: (page - 1) * itemsPerPage,
          limit: itemsPerPage,
          ...filters
        };
        
        // Fetch alerts
        const alertsData = await alertsApi.getAll(params);
        
        // Check if there are more alerts
        setHasMore(alertsData.length === itemsPerPage);
        
        // Update alerts (append if loading more, replace if filtering)
        if (page === 1) {
          setAlerts(alertsData);
        } else {
          setAlerts(prev => [...prev, ...alertsData]);
        }
        
        // Fetch stats
        const statsData = await alertsApi.getStats({
          hours: filters.hours,
          camera_id: filters.camera_id || undefined
        });
        setStats(statsData);
        
        setLoading(false);
        setError(null);
      } catch (err) {
        console.error('Error fetching alerts:', err);
        setError('Failed to load alerts. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchAlerts();
  }, [filters, page]);
  
  // Handle filter changes
  const handleFilterChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    const newValue = type === 'checkbox' 
      ? checked 
      : type === 'number'
        ? parseInt(value)
        : value;
        
    setFilters(prev => ({
      ...prev,
      [name]: newValue
    }));
    
    // Reset to page 1 when filters change
    setPage(1);
    
    // Update URL search params
    const newParams = new URLSearchParams();
    if (name === 'camera_id' && value) newParams.set('camera', value);
    if (name === 'violation_type' && value) newParams.set('type', value);
    if (name === 'unresolved_only' && checked) newParams.set('unresolved', 'true');
    if (name === 'hours') newParams.set('hours', value);
    
    setSearchParams(newParams);
  };
  
  // Load more alerts
  const loadMore = () => {
    setPage(prev => prev + 1);
  };
  
  // Resolve an alert
  const resolveAlert = async (alertId, currentStatus) => {
    try {
      await alertsApi.resolve(alertId, !currentStatus);
      
      // Update the alert in the list
      setAlerts(prev => 
        prev.map(alert => 
          alert.id === alertId 
            ? { ...alert, resolved: !currentStatus, resolved_at: new Date().toISOString() }
            : alert
        )
      );
    } catch (err) {
      console.error(`Error resolving alert ${alertId}:`, err);
      alert('Failed to update alert status. Please try again.');
    }
  };
  
  // Get camera name by ID
  const getCameraName = (cameraId) => {
    const camera = cameras.find(c => c.id === cameraId);
    return camera ? camera.name : `Camera ${cameraId}`;
  };
  
  if (loading && page === 1 && alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading alerts...</p>
        </div>
      </div>
    );
  }
  
  if (error && alerts.length === 0) {
    return (
      <div className="text-center p-6 bg-danger-50 text-danger-700 rounded-lg">
        <FaExclamationTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p>{error}</p>
        <button
          className="mt-4 btn-primary"
          onClick={() => setFilters({ ...filters })} // Trigger re-fetch
        >
          Retry
        </button>
      </div>
    );
  }
  
  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
      </div>
      
      {/* Filter section */}
      <div className="bg-white shadow rounded-lg p-4 mb-6">
        <div className="flex items-center mb-4">
          <FaFilter className="text-gray-400 mr-2" />
          <h2 className="text-lg font-medium text-gray-900">Filter Alerts</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Camera filter */}
          <div>
            <label htmlFor="camera_id" className="form-label">
              Camera
            </label>
            <select
              id="camera_id"
              name="camera_id"
              value={filters.camera_id}
              onChange={handleFilterChange}
              className="form-input"
            >
              <option value="">All Cameras</option>
              {cameras.map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </div>
          
          {/* Violation type filter */}
          <div>
            <label htmlFor="violation_type" className="form-label">
              Violation Type
            </label>
            <select
              id="violation_type"
              name="violation_type"
              value={filters.violation_type}
              onChange={handleFilterChange}
              className="form-input"
            >
              <option value="">All Types</option>
              {stats && stats.by_type && Object.keys(stats.by_type).map(type => (
                <option key={type} value={type}>
                  {type.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>
          
          {/* Time range filter */}
          <div>
            <label htmlFor="hours" className="form-label">
              Time Range
            </label>
            <select
              id="hours"
              name="hours"
              value={filters.hours}
              onChange={handleFilterChange}
              className="form-input"
            >
              <option value="4">Last 4 hours</option>
              <option value="8">Last 8 hours</option>
              <option value="24">Last 24 hours</option>
              <option value="48">Last 2 days</option>
              <option value="168">Last week</option>
            </select>
          </div>
          
          {/* Unresolved only toggle */}
          <div className="flex items-center pt-7">
            <input
              type="checkbox"
              id="unresolved_only"
              name="unresolved_only"
              checked={filters.unresolved_only}
              onChange={handleFilterChange}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="unresolved_only" className="ml-2 block text-sm text-gray-900">
              Show unresolved alerts only
            </label>
          </div>
        </div>
      </div>
      
      {/* Stats overview */}
      {stats && (
        <div className="bg-white shadow rounded-lg p-4 mb-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Statistics</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <p className="text-sm text-gray-500">Total Alerts</p>
              <p className="text-3xl font-bold text-gray-900">{stats.total_alerts}</p>
              <p className="text-xs text-gray-500">
                in the last {stats.time_range_hours} hours
              </p>
            </div>
            
            {stats.by_type && Object.keys(stats.by_type).length > 0 && (
              <div>
                <p className="text-sm text-gray-500 mb-2">By Violation Type</p>
                <div className="space-y-1">
                  {Object.entries(stats.by_type)
                    .sort((a, b) => b[1] - a[1]) // Sort by count (descending)
                    .map(([type, count]) => (
                      <div key={type} className="flex justify-between text-sm">
                        <span>{type.replace('_', ' ')}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                    ))
                  }
                </div>
              </div>
            )}
            
            {stats.by_camera && Object.keys(stats.by_camera).length > 0 && (
              <div>
                <p className="text-sm text-gray-500 mb-2">By Camera</p>
                <div className="space-y-1">
                  {Object.values(stats.by_camera)
                    .sort((a, b) => b.count - a.count) // Sort by count (descending)
                    .slice(0, 5) // Show top 5
                    .map(({ camera_id, name, count }) => (
                      <div key={camera_id} className="flex justify-between text-sm">
                        <span className="truncate">{name}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                    ))
                  }
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Alerts table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        {alerts.length === 0 ? (
          <div className="text-center p-8">
            <FaBell className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h2 className="text-xl font-medium text-gray-900 mb-2">No alerts found</h2>
            <p className="text-gray-500">
              No alerts match your current filter criteria.
              Try adjusting your filters or wait for new alerts.
            </p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Alert
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Camera
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {alerts.map((alert) => (
                    <tr key={alert.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10">
                            {alert.screenshot_path ? (
                              <img
                                src={`http://localhost:8000${alert.screenshot_path}`}
                                alt="Alert thumbnail"
                                className="h-10 w-10 rounded-full object-cover"
                              />
                            ) : (
                              <div className="h-10 w-10 rounded-full bg-danger-100 flex items-center justify-center">
                                <FaBell className="h-5 w-5 text-danger-600" />
                              </div>
                            )}
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {alert.violation_type.replace('_', ' ')}
                            </div>
                            <div className="text-xs text-gray-500">
                              Confidence: {Math.round(alert.confidence * 100)}%
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {getCameraName(alert.camera_id)}
                        </div>
                        <div className="text-xs text-gray-500">
                          ID: {alert.camera_id}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {new Date(alert.created_at).toLocaleString()}
                        </div>
                        <div className="text-xs text-gray-500">
                          {alert.resolved && alert.resolved_at && (
                            <>Resolved: {new Date(alert.resolved_at).toLocaleString()}</>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            alert.resolved
                              ? 'bg-success-100 text-success-800'
                              : 'bg-danger-100 text-danger-800'
                          }`}
                        >
                          {alert.resolved ? 'Resolved' : 'Active'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end space-x-2">
                          <Link
                            to={`/alerts/${alert.id}`}
                            className="text-primary-600 hover:text-primary-900"
                          >
                            <FaEye className="inline mr-1" />
                            View
                          </Link>
                          
                          <button
                            onClick={() => resolveAlert(alert.id, alert.resolved)}
                            className={`${
                              alert.resolved
                                ? 'text-warning-600 hover:text-warning-900'
                                : 'text-success-600 hover:text-success-900'
                            }`}
                          >
                            <FaCheck className="inline mr-1" />
                            {alert.resolved ? 'Unresolve' : 'Resolve'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Load more button */}
            {hasMore && (
              <div className="px-6 py-4 border-t flex justify-center">
                <button
                  onClick={loadMore}
                  className="btn-secondary"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500 mr-2"></div>
                      Loading...
                    </>
                  ) : (
                    'Load More'
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default AlertsList;