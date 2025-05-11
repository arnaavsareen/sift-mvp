import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { camerasApi, dashboardApi, alertsApi, websocketApi } from '../api/api';
import { 
  FaVideo, 
  FaPlay, 
  FaStop, 
  FaEdit, 
  FaArrowLeft,
  FaExclamationTriangle,
  FaBell
} from 'react-icons/fa';

const CameraDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const frameRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [camera, setCamera] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [wsError, setWsError] = useState(null);
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    url: '',
    location: '',
    is_active: true
  });
  
  // Fetch camera details
  const fetchCamera = async () => {
    try {
      const data = await camerasApi.getById(id);
      setCamera(data);
      
      // Initialize form data
      setFormData({
        name: data.name,
        url: data.url,
        location: data.location || '',
        is_active: data.is_active
      });
      
      return data;
    } catch (err) {
      console.error(`Error fetching camera ${id}:`, err);
      setError('Failed to load camera details. Please try again later.');
      return null;
    }
  };
  
  // Fetch camera status
  const fetchCameraStatus = async () => {
    try {
      const status = await camerasApi.getStatus(id);
      setIsProcessing(status.is_processing);
    } catch (err) {
      console.error(`Error fetching camera status for ${id}:`, err);
      setIsProcessing(false);
    }
  };
  
  // Fetch camera alerts
  const fetchAlerts = async (cameraId) => {
    try {
      const data = await alertsApi.getAll({ 
        camera_id: cameraId, 
        limit: 5
      });
      setAlerts(data);
    } catch (err) {
      console.error(`Error fetching alerts for camera ${cameraId}:`, err);
    }
  };
  
  // Load camera data
  useEffect(() => {
    const loadCamera = async () => {
      setLoading(true);
      
      const cameraData = await fetchCamera();
      if (cameraData) {
        await fetchCameraStatus();
        await fetchAlerts(cameraData.id);
      }
      
      setLoading(false);
    };
    
    loadCamera();
  }, [id]);
  
  // Frame refresh using WebSocket
  useEffect(() => {
    if (!isProcessing) {
      // Clean up any existing connection
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setWsError(null);
      return;
    }
    
    // Start WebSocket connection for streaming
    const onFrame = (frameData, timestamp) => {
      if (frameRef.current) {
        // Ensure we're correctly setting the image source with the base64 data
        if (typeof frameData === 'string') {
          frameRef.current.src = `data:image/jpeg;base64,${frameData}`;
          frameRef.current.style.display = 'block';
          
          // Clear any error indicators
          setWsError(null);
        }
      }
    };
    
    const onAlert = (alertData) => {
      console.log("Received alert via WebSocket:", alertData);
      
      // When a new alert is received via WebSocket, update the alerts list
      setAlerts(prevAlerts => {
        // Check if alert already exists
        const exists = prevAlerts.some(a => a.id === alertData.id);
        if (exists) return prevAlerts;
        
        // Add to beginning of list
        return [alertData, ...prevAlerts].slice(0, 5);
      });
    };
    
    const onError = (error) => {
      console.error("WebSocket error:", error);
      setWsError(error.toString());
      
      // If WebSocket fails, fall back to regular polling
      if (frameRef.current) {
        const timestamp = new Date().getTime();
        frameRef.current.src = `${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/dashboard/cameras/${id}/latest-frame?format=jpeg&t=${timestamp}`;
      }
    };
    
    const onClose = (event) => {
      // Connection closed - this is handled by the WebSocket library's automatic reconnection logic
      if (event.code !== 1000) { // Not a normal closure
        console.log("WebSocket connection closed abnormally:", event.code, event.reason);
      }
    };
    
    console.log("Creating WebSocket connection for camera:", id);
    const ws = websocketApi.createCameraStream(id, onFrame, onAlert, onError, onClose);
    wsRef.current = ws;
    
    return () => {
      if (wsRef.current) {
        console.log("Cleaning up WebSocket connection");
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [id, isProcessing]);
  
  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    });
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      await camerasApi.update(id, formData);
      setShowEditModal(false);
      fetchCamera(); // Refresh camera data
    } catch (err) {
      console.error('Error updating camera:', err);
      alert('Failed to update camera. Please try again.');
    }
  };
  
  // Handle camera actions
  const startCamera = async () => {
    try {
      await camerasApi.start(id);
      setIsProcessing(true);
    } catch (err) {
      console.error(`Error starting camera ${id}:`, err);
      alert('Failed to start camera processing. Please try again.');
    }
  };
  
  const stopCamera = async () => {
    try {
      await camerasApi.stop(id);
      setIsProcessing(false);
    } catch (err) {
      console.error(`Error stopping camera ${id}:`, err);
      alert('Failed to stop camera processing. Please try again.');
    }
  };
  
  const deleteCamera = async () => {
    if (!window.confirm('Are you sure you want to delete this camera?')) {
      return;
    }
    
    try {
      await camerasApi.delete(id);
      navigate('/cameras');
    } catch (err) {
      console.error(`Error deleting camera ${id}:`, err);
      alert('Failed to delete camera. Please try again.');
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading camera details...</p>
        </div>
      </div>
    );
  }
  
  if (error || !camera) {
    return (
      <div className="text-center p-6 bg-danger-50 text-danger-700 rounded-lg">
        <FaExclamationTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p>{error || 'Camera not found'}</p>
        <div className="mt-6">
          <Link to="/cameras" className="btn-primary">
            <FaArrowLeft className="mr-2" />
            Back to Cameras
          </Link>
        </div>
      </div>
    );
  }
  
  return (
    <div>
      {/* Header with back button */}
      <div className="flex items-center mb-6">
        <Link to="/cameras" className="mr-4 text-gray-600 hover:text-gray-900">
          <FaArrowLeft className="text-xl" />
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 flex-grow">{camera.name}</h1>
        <div className="space-x-2">
          {camera.is_active && (
            isProcessing ? (
              <button
                onClick={stopCamera}
                className="btn-danger"
              >
                <FaStop className="mr-2" />
                Stop Monitoring
              </button>
            ) : (
              <button
                onClick={startCamera}
                className="btn-success"
              >
                <FaPlay className="mr-2" />
                Start Monitoring
              </button>
            )
          )}
          <button
            onClick={() => setShowEditModal(true)}
            className="btn-secondary"
          >
            <FaEdit className="mr-2" />
            Edit
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera feed and details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Camera feed */}
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center justify-between">
              <span>Live Feed</span>
              {isProcessing && (
                <span className="flex items-center">
                  <span className="animate-pulse w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                  <span className="text-sm font-normal text-green-600">Live</span>
                </span>
              )}
            </h2>
            <div className="bg-gray-100 rounded-lg w-full overflow-hidden flex items-center justify-center shadow-inner relative">
              <div className="aspect-video w-full relative">
                {isProcessing ? (
                  <>
                    <img
                      ref={frameRef}
                      src={`${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/dashboard/cameras/${id}/latest-frame?format=jpeg&t=${new Date().getTime()}`}
                      alt="Camera feed"
                      className="w-full h-full object-contain bg-black"
                      onError={(e) => {
                        e.target.src = '';
                        e.target.className = 'hidden';
                        document.getElementById('camera-error-message').classList.remove('hidden');
                      }}
                    />
                    <div className="absolute top-0 left-0 p-3 flex flex-col space-y-1">
                      <div className="flex items-center space-x-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                        <FaVideo className="h-4 w-4" />
                        <span>Camera {camera.name}</span>
                      </div>
                      {camera.location && (
                        <div className="bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm flex items-center space-x-2">
                          <span>📍 {camera.location}</span>
                        </div>
                      )}
                    </div>
                    <div className="absolute bottom-3 right-3 bg-black bg-opacity-50 px-2 py-1 rounded text-xs text-white">
                      {new Date().toLocaleTimeString()}
                    </div>
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 text-white">
                    <FaVideo className="h-16 w-16 text-gray-400 mb-4" />
                    <p className="text-xl font-medium text-center mb-2">No Active Video Feed</p>
                    <p className="text-gray-400 max-w-md text-center mb-6">
                      Camera is not currently monitoring. 
                      Click "Start Monitoring" to begin processing the video feed.
                    </p>
                    <button
                      onClick={startCamera}
                      className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center space-x-2"
                    >
                      <FaPlay className="h-4 w-4" />
                      <span>Start Monitoring</span>
                    </button>
                  </div>
                )}
                <div id="camera-error-message" className="hidden absolute inset-0 flex flex-col items-center justify-center bg-gray-800 text-white">
                  <FaExclamationTriangle className="h-16 w-16 text-yellow-500 mb-4" />
                  <p className="text-xl font-medium text-center mb-2">Connection Error</p>
                  <p className="text-gray-400 max-w-md text-center mb-6">
                    Unable to load camera feed. 
                    The camera might be offline or the stream URL is invalid.
                  </p>
                  <div className="flex space-x-4">
                    <button
                      onClick={() => {
                        if (frameRef.current) {
                          frameRef.current.src = `${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/dashboard/cameras/${id}/latest-frame?format=jpeg&t=${new Date().getTime()}`;
                          frameRef.current.classList.remove('hidden');
                          document.getElementById('camera-error-message').classList.add('hidden');
                        }
                      }}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center space-x-2"
                    >
                      <span>Retry Connection</span>
                    </button>
                    <button
                      onClick={stopCamera}
                      className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center space-x-2"
                    >
                      <FaStop className="h-4 w-4" />
                      <span>Stop Monitoring</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            {isProcessing && (
              <div className="mt-4 text-sm flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`w-3 h-3 mr-2 rounded-full ${wsError ? 'bg-danger-500' : 'bg-success-500'}`}></div>
                  <p className={wsError ? 'text-danger-700' : 'text-success-700'}>
                    {wsError 
                      ? `Connection issue: ${wsError}` 
                      : 'Connected via WebSocket'}
                  </p>
                </div>
                <div className="text-gray-500 text-xs">
                  Last updated: {new Date().toLocaleTimeString()}
                </div>
              </div>
            )}
          </div>
          
          {/* Camera details */}
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Camera Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-500">Name</p>
                <p className="mt-1">{camera.name}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Status</p>
                <p className="mt-1">
                  <span
                    className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      !camera.is_active
                        ? 'bg-gray-100 text-gray-800'
                        : isProcessing
                        ? 'bg-success-100 text-success-800'
                        : 'bg-warning-100 text-warning-800'
                    }`}
                  >
                    {!camera.is_active
                      ? 'Inactive'
                      : isProcessing
                      ? 'Monitoring'
                      : 'Idle'}
                  </span>
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Stream URL</p>
                <p className="mt-1 text-sm break-all">{camera.url}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Location</p>
                <p className="mt-1">{camera.location || 'Not specified'}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Added on</p>
                <p className="mt-1">
                  {new Date(camera.created_at).toLocaleDateString()}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500">Camera ID</p>
                <p className="mt-1">{camera.id}</p>
              </div>
            </div>
            
            <div className="mt-6 border-t pt-4">
              <button
                onClick={deleteCamera}
                className="text-danger-600 hover:text-danger-800 text-sm font-medium"
              >
                Delete this camera
              </button>
            </div>
          </div>
        </div>
        
        {/* Recent alerts */}
        <div className="space-y-6">
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Alerts</h2>
            <div className="space-y-4">
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <Link
                    to={`/alerts/${alert.id}`}
                    key={alert.id}
                    className="block rounded-lg overflow-hidden shadow-sm border border-gray-200 hover:shadow-md transition-all duration-200"
                  >
                    <div className="flex flex-col">
                      {alert.screenshot_path ? (
                        <div className="relative">
                          <img
                            src={`${process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:8000'}${alert.screenshot_path.startsWith('/') ? '' : '/screenshots/'}${alert.screenshot_path}`}
                            alt="Alert screenshot"
                            className="w-full h-40 object-cover"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSIyMDAiIGZpbGw9IiNFNUU3RUIiLz48cGF0aCBkPSJNMTAwIDcwQzEwMCA4MS4wNDU3IDkxLjA0NTcgOTAgODAgOTBDNjguOTU0MyA5MCA2MCA4MS4wNDU3IDYwIDcwQzYwIDU4Ljk1NDMgNjguOTU0MyA1MCA4MCA1MEM5MS4wNDU3IDUwIDEwMCA1OC45NTQzIDEwMCA3MFoiIGZpbGw9IiNBMUExQUEiLz48cGF0aCBkPSJNMTQwIDEzMEMxNDAgMTUyLjA5MSAxMjIuMDkxIDE3MCAxMDAgMTcwQzc3LjkwODYgMTcwIDYwIDE1Mi4wOTEgNjAgMTMwQzYwIDEwNy45MDkgNzcuOTA4NiA5MCAxMDAgOTBDMTIyLjA5MSA5MCAxNDAgMTA3LjkwOSAxNDAgMTMwWiIgZmlsbD0iI0ExQTFBQSIvPjwvc3ZnPg==';
                            }}
                          />
                          <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/60 to-transparent text-white p-2">
                            <span className="px-2 py-0.5 bg-red-600/90 text-white text-xs font-semibold rounded uppercase">
                              {alert.violation_type.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent text-white text-xs p-2">
                            <div className="flex justify-between items-center">
                              <div>
                                {new Date(alert.created_at).toLocaleString([], {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </div>
                              <div className="px-2 py-0.5 bg-white/20 backdrop-blur-sm rounded-full">
                                {Math.round(alert.confidence * 100)}% confidence
                              </div>
                            </div>
                          </div>
                          {alert.bbox && (
                            <div className="absolute inset-0 pointer-events-none">
                              <div className="w-full h-full border-2 border-yellow-400 border-dashed opacity-60 animate-pulse"></div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="bg-gray-100 h-32 flex items-center justify-center">
                          <div className="text-center">
                            <FaBell className="mx-auto h-8 w-8 text-gray-300 mb-2" />
                            <p className="text-sm text-gray-500">No image available</p>
                          </div>
                        </div>
                      )}
                      <div className="p-3 bg-white">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center">
                            <FaBell className="h-4 w-4 text-danger-500 mr-2" />
                            <p className="text-sm font-medium text-gray-900">
                              {alert.violation_type.replace(/_/g, " ")}
                            </p>
                          </div>
                          <div className="flex items-center">
                            <span className={`px-2 py-0.5 text-xs rounded-full ${alert.resolved ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                              {alert.resolved ? 'Resolved' : 'Active'}
                            </span>
                          </div>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
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
                    </div>
                  </Link>
                ))
              ) : (
                <div className="text-center py-8 bg-gray-50 rounded-lg border border-gray-100">
                  <FaBell className="mx-auto h-10 w-10 text-gray-300 mb-2" />
                  <p className="text-gray-500">No alerts found for this camera</p>
                  <p className="text-xs text-gray-400 mt-1">Alerts will appear here when safety violations are detected</p>
                </div>
              )}
            </div>
            {alerts.length > 0 && (
              <div className="mt-4 text-center">
                <Link
                  to={`/alerts?camera=${camera.id}`}
                  className="text-sm text-primary-600 hover:text-primary-800"
                >
                  View all alerts for this camera
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Edit Camera Modal */}
      {showEditModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen px-4">
            {/* Backdrop */}
            <div
              className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
              onClick={() => setShowEditModal(false)}
            ></div>
            
            {/* Modal */}
            <div className="bg-white rounded-lg overflow-hidden shadow-xl transform transition-all sm:max-w-lg sm:w-full z-10">
              <div className="px-6 py-4 bg-primary-700 text-white">
                <h3 className="text-lg font-medium">Edit Camera</h3>
              </div>
              
              <form onSubmit={handleSubmit}>
                <div className="p-6 space-y-4">
                  <div>
                    <label htmlFor="name" className="form-label">
                      Camera Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="form-input"
                      required
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="url" className="form-label">
                      Stream URL
                    </label>
                    <input
                      type="text"
                      id="url"
                      name="url"
                      value={formData.url}
                      onChange={handleInputChange}
                      className="form-input"
                      placeholder="rtsp:// or http:// stream URL"
                      required
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      RTSP, HTTP or local video file path
                    </p>
                  </div>
                  
                  <div>
                    <label htmlFor="location" className="form-label">
                      Location (Optional)
                    </label>
                    <input
                      type="text"
                      id="location"
                      name="location"
                      value={formData.location}
                      onChange={handleInputChange}
                      className="form-input"
                      placeholder="e.g., Factory Floor, Assembly Line 1"
                    />
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="is_active"
                      name="is_active"
                      checked={formData.is_active}
                      onChange={handleInputChange}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <label htmlFor="is_active" className="ml-2 block text-sm text-gray-900">
                      Active
                    </label>
                  </div>
                </div>
                
                <div className="px-6 py-4 bg-gray-50 flex justify-end space-x-2">
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => setShowEditModal(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn-primary"
                  >
                    Save Changes
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraDetail;