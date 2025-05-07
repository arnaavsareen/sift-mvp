import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { camerasApi, dashboardApi, alertsApi } from '../api/api';
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
  
  const [camera, setCamera] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  
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
  
  // Frame refresh interval
  useEffect(() => {
    if (!isProcessing) return;
    
    const refreshFrame = () => {
      if (frameRef.current) {
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        frameRef.current.src = `http://localhost:8000/api/dashboard/cameras/${id}/latest-frame?format=jpeg&t=${timestamp}`;
      }
    };
    
    // Initial load
    refreshFrame();
    
    // Set up interval
    const interval = setInterval(refreshFrame, 1000); // Refresh every second
    
    return () => clearInterval(interval);
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
            <h2 className="text-lg font-medium text-gray-900 mb-4">Live Feed</h2>
            <div className="bg-gray-200 rounded-lg w-full aspect-video overflow-hidden flex items-center justify-center">
              {isProcessing ? (
                <img
                  ref={frameRef}
                  src={`http://localhost:8000/api/dashboard/cameras/${id}/latest-frame?format=jpeg`}
                  alt="Camera feed"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    e.target.src = '';
                    e.target.className = 'hidden';
                    e.target.nextSibling.className = 'block text-center p-4';
                  }}
                />
              ) : (
                <div className="text-center p-4">
                  <FaVideo className="mx-auto h-16 w-16 text-gray-400 mb-4" />
                  <p className="text-gray-600">
                    Camera is not currently monitoring. 
                    Click "Start Monitoring" to begin processing the video feed.
                  </p>
                </div>
              )}
              <div className="hidden text-center p-4">
                <FaExclamationTriangle className="mx-auto h-12 w-12 text-warning-500 mb-4" />
                <p className="text-gray-700">
                  Unable to load camera feed. 
                  The camera might be offline or the stream URL is invalid.
                </p>
              </div>
            </div>
            
            {isProcessing && (
              <div className="mt-4 text-sm text-gray-600">
                <p>Processing active. Refreshing feed every second.</p>
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
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-medium text-gray-900">Recent Alerts</h2>
              <Link 
                to={`/alerts?camera=${camera.id}`}
                className="text-sm text-primary-600 hover:text-primary-800"
              >
                View all
              </Link>
            </div>
            
            <div className="space-y-4">
              {alerts.length > 0 ? (
                alerts.map(alert => (
                  <Link
                    key={alert.id}
                    to={`/alerts/${alert.id}`}
                    className="block border border-gray-200 rounded-md p-3 hover:bg-gray-50"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start">
                        <div className="flex-shrink-0">
                          <div className="h-8 w-8 rounded-full bg-danger-100 flex items-center justify-center">
                            <FaBell className="h-4 w-4 text-danger-600" />
                          </div>
                        </div>
                        <div className="ml-3">
                          <p className="font-medium text-gray-900">
                            {alert.violation_type.replace('_', ' ')}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(alert.created_at).toLocaleString()}
                          </p>
                        </div>
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
                    {alert.screenshot_path && (
                      <div className="mt-2">
                        <img
                          src={`http://localhost:8000${alert.screenshot_path}`}
                          alt="Alert screenshot"
                          className="rounded-md w-full h-20 object-cover"
                        />
                      </div>
                    )}
                  </Link>
                ))
              ) : (
                <div className="text-center py-6">
                  <p className="text-gray-500">No alerts found for this camera</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Camera zones (placeholder for future feature) */}
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Monitoring Zones</h2>
            <div className="text-center py-6">
              <p className="text-gray-500">
                No zones configured. 
                Monitoring zones feature will be available in the next version.
              </p>
            </div>
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