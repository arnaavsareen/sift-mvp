import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { alertsApi, camerasApi } from '../api/api';
import { 
  FaBell, 
  FaArrowLeft, 
  FaExclamationTriangle, 
  FaCheck, 
  FaCamera,
  FaCalendarAlt,
  FaPercentage,
  FaMapMarkerAlt
} from 'react-icons/fa';

const AlertDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const [alert, setAlert] = useState(null);
  const [camera, setCamera] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch alert details
  useEffect(() => {
    const fetchAlertDetails = async () => {
      try {
        setLoading(true);
        
        const alertData = await alertsApi.getById(id);
        setAlert(alertData);
        
        // Fetch associated camera
        if (alertData.camera_id) {
          const cameraData = await camerasApi.getById(alertData.camera_id);
          setCamera(cameraData);
        }
        
        setLoading(false);
      } catch (err) {
        console.error(`Error fetching alert ${id}:`, err);
        setError('Failed to load alert details. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchAlertDetails();
  }, [id]);
  
  // Handle resolving alert
  const resolveAlert = async () => {
    try {
      const updatedAlert = await alertsApi.resolve(id, !alert.resolved);
      setAlert(updatedAlert);
    } catch (err) {
      console.error(`Error resolving alert ${id}:`, err);
      alert('Failed to update alert status. Please try again.');
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading alert details...</p>
        </div>
      </div>
    );
  }
  
  if (error || !alert) {
    return (
      <div className="text-center p-6 bg-danger-50 text-danger-700 rounded-lg">
        <FaExclamationTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p>{error || 'Alert not found'}</p>
        <div className="mt-6">
          <Link to="/alerts" className="btn-primary">
            <FaArrowLeft className="mr-2" />
            Back to Alerts
          </Link>
        </div>
      </div>
    );
  }
  
  return (
    <div>
      {/* Header with back button */}
      <div className="flex items-center mb-6">
        <Link to="/alerts" className="mr-4 text-gray-600 hover:text-gray-900">
          <FaArrowLeft className="text-xl" />
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 flex-grow">
          Alert Details
        </h1>
        <button
          onClick={resolveAlert}
          className={`btn ${
            alert.resolved
              ? 'btn-secondary'
              : 'btn-success'
          }`}
        >
          <FaCheck className="mr-2" />
          {alert.resolved ? 'Mark as Unresolved' : 'Mark as Resolved'}
        </button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Status card */}
          <div className="card p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">
                {alert.violation_type.replace('_', ' ')}
              </h2>
              <span
                className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                  alert.resolved
                    ? 'bg-success-100 text-success-800'
                    : 'bg-danger-100 text-danger-800'
                }`}
              >
                {alert.resolved ? 'Resolved' : 'Active'}
              </span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center">
                <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                  <FaCamera className="h-5 w-5 text-primary-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-500">Camera</p>
                  <p className="text-sm">
                    {camera ? camera.name : `Camera ${alert.camera_id}`}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                  <FaCalendarAlt className="h-5 w-5 text-primary-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-500">Detected</p>
                  <p className="text-sm">
                    {new Date(alert.created_at).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                  <FaPercentage className="h-5 w-5 text-primary-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-500">Confidence</p>
                  <p className="text-sm">
                    {Math.round(alert.confidence * 100)}%
                  </p>
                </div>
              </div>
              
              {camera && camera.location && (
                <div className="flex items-center">
                  <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                    <FaMapMarkerAlt className="h-5 w-5 text-primary-600" />
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-500">Location</p>
                    <p className="text-sm">{camera.location}</p>
                  </div>
                </div>
              )}
            </div>
            
            {alert.resolved && alert.resolved_at && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm font-medium text-gray-500">
                  Resolved at: {new Date(alert.resolved_at).toLocaleString()}
                </p>
              </div>
            )}
          </div>
          
          {/* Screenshot */}
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Screenshot</h2>
            {alert.screenshot_path ? (
              <div className="rounded-lg overflow-hidden">
                <img
                  src={`http://localhost:8000${alert.screenshot_path}`}
                  alt="Alert screenshot"
                  className="w-full object-contain max-h-96"
                />
              </div>
            ) : (
              <div className="bg-gray-200 rounded-lg flex items-center justify-center h-64">
                <div className="text-center p-4">
                  <FaBell className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-gray-500">
                    No screenshot available for this alert
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Bounding box information (if available) */}
          {alert.bbox && (
            <div className="card p-4">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Detection Details</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">X1</p>
                  <p className="text-sm">{Math.round(alert.bbox[0])}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Y1</p>
                  <p className="text-sm">{Math.round(alert.bbox[1])}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">X2</p>
                  <p className="text-sm">{Math.round(alert.bbox[2])}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Y2</p>
                  <p className="text-sm">{Math.round(alert.bbox[3])}</p>
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-500">
                  Bounding box coordinates representing the location of the detected violation
                  in the camera frame (x1, y1) to (x2, y2).
                </p>
              </div>
            </div>
          )}
        </div>
        
        {/* Sidebar */}
        <div className="space-y-6">
          {/* Camera information */}
          {camera && (
            <div className="card p-4">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Camera Information</h2>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Name</p>
                  <p className="mt-1">{camera.name}</p>
                </div>
                
                {camera.location && (
                  <div>
                    <p className="text-sm font-medium text-gray-500">Location</p>
                    <p className="mt-1">{camera.location}</p>
                  </div>
                )}
                
                <div>
                  <p className="text-sm font-medium text-gray-500">Status</p>
                  <p className="mt-1">
                    <span
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        camera.is_active
                          ? 'bg-success-100 text-success-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {camera.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </p>
                </div>
                
                <div className="pt-2">
                  <Link
                    to={`/cameras/${camera.id}`}
                    className="text-primary-600 hover:text-primary-800 text-sm font-medium"
                  >
                    View Camera
                  </Link>
                </div>
              </div>
            </div>
          )}
          
          {/* Related alerts */}
          <div className="card p-4">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Related Actions</h2>
            <div className="space-y-3">
              <Link
                to={`/alerts?camera=${alert.camera_id}`}
                className="flex items-center text-primary-600 hover:text-primary-800"
              >
                <FaBell className="mr-2" />
                View all alerts from this camera
              </Link>
              
              <Link
                to={`/alerts?type=${alert.violation_type}`}
                className="flex items-center text-primary-600 hover:text-primary-800"
              >
                <FaExclamationTriangle className="mr-2" />
                View all {alert.violation_type.replace('_', ' ')} alerts
              </Link>
              
              <button
                onClick={resolveAlert}
                className={`flex items-center ${
                  alert.resolved
                    ? 'text-warning-600 hover:text-warning-800'
                    : 'text-success-600 hover:text-success-800'
                }`}
              >
                <FaCheck className="mr-2" />
                {alert.resolved ? 'Mark as unresolved' : 'Mark as resolved'}
              </button>
            </div>
          </div>
          
          {/* Alert ID information */}
          <div className="card p-4 bg-gray-50">
            <div className="text-sm text-gray-500">
              <p>Alert ID: {alert.id}</p>
              <p>Created: {new Date(alert.created_at).toLocaleString()}</p>
              {alert.resolved && alert.resolved_at && (
                <p>Resolved: {new Date(alert.resolved_at).toLocaleString()}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertDetail;