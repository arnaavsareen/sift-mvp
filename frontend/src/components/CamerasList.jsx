import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { camerasApi } from '../api/api';
import { 
  FaVideo, 
  FaPlus, 
  FaPlay, 
  FaStop, 
  FaEdit, 
  FaTrash,
  FaExclamationTriangle,
  FaEye
} from 'react-icons/fa';

const CamerasList = () => {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [processingStatus, setProcessingStatus] = useState({});
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    url: '',
    location: '',
    is_active: true
  });
  
  // Fetch cameras
  const fetchCameras = async () => {
    try {
      setLoading(true);
      const data = await camerasApi.getAll();
      setCameras(data);
      
      // Initialize processing status
      const status = {};
      for (const camera of data) {
        try {
          const cameraStatus = await camerasApi.getStatus(camera.id);
          status[camera.id] = cameraStatus.is_processing;
        } catch (err) {
          console.error(`Error fetching status for camera ${camera.id}:`, err);
          status[camera.id] = false;
        }
      }
      
      setProcessingStatus(status);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError('Failed to load cameras. Please try again later.');
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchCameras();
    
    // Refresh status every 10 seconds
    const interval = setInterval(() => {
      // Only update status, not the full list
      cameras.forEach(async (camera) => {
        try {
          const cameraStatus = await camerasApi.getStatus(camera.id);
          setProcessingStatus(prev => ({
            ...prev,
            [camera.id]: cameraStatus.is_processing
          }));
        } catch (err) {
          console.error(`Error updating status for camera ${camera.id}:`, err);
        }
      });
    }, 10000);
    
    return () => clearInterval(interval);
  }, [cameras.length]); // Only re-run if the number of cameras changes
  
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
      await camerasApi.create(formData);
      setShowAddModal(false);
      setFormData({
        name: '',
        url: '',
        location: '',
        is_active: true
      });
      fetchCameras(); // Refresh the list
    } catch (err) {
      console.error('Error creating camera:', err);
      alert('Failed to create camera. Please try again.');
    }
  };
  
  // Handle camera actions
  const startCamera = async (id) => {
    try {
      await camerasApi.start(id);
      setProcessingStatus(prev => ({
        ...prev,
        [id]: true
      }));
    } catch (err) {
      console.error(`Error starting camera ${id}:`, err);
      alert('Failed to start camera processing. Please try again.');
    }
  };
  
  const stopCamera = async (id) => {
    try {
      await camerasApi.stop(id);
      setProcessingStatus(prev => ({
        ...prev,
        [id]: false
      }));
    } catch (err) {
      console.error(`Error stopping camera ${id}:`, err);
      alert('Failed to stop camera processing. Please try again.');
    }
  };
  
  const deleteCamera = async (id) => {
    if (!window.confirm('Are you sure you want to delete this camera?')) {
      return;
    }
    
    try {
      await camerasApi.delete(id);
      // Show success message
      alert('Camera deleted successfully');
      fetchCameras(); // Refresh the list
    } catch (err) {
      console.error(`Error deleting camera ${id}:`, err);
      // Show more specific error message based on the error
      const errorMessage = err.response?.status === 404
        ? 'Camera not found. It may have been already deleted.'
        : 'Failed to delete camera. Please try again.';
      alert(errorMessage);
    }
  };
  
  if (loading && cameras.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading cameras...</p>
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
          onClick={() => fetchCameras()}
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
        <h1 className="text-2xl font-bold text-gray-900">Cameras</h1>
        <button
          className="btn-primary inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          onClick={() => setShowAddModal(true)}
        >
          <FaPlus className="mr-2 -ml-1 h-4 w-4" />
          Add Camera
        </button>
      </div>
      
      {/* Camera list */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        {cameras.length === 0 ? (
          <div className="text-center p-8">
            <FaVideo className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h2 className="text-xl font-medium text-gray-900 mb-2">No cameras found</h2>
            <p className="text-gray-500 mb-4">Add a camera to start monitoring for safety compliance</p>
            <button
              className="btn-primary inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              onClick={() => setShowAddModal(true)}
            >
              <FaPlus className="mr-2 -ml-1 h-4 w-4" />
              Add Camera
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    URL
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Location
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
                {cameras.map((camera) => (
                  <tr key={camera.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-primary-100 rounded-full">
                          <FaVideo className="h-5 w-5 text-primary-600" />
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {camera.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            ID: {camera.id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-500 truncate max-w-xs">
                        {camera.url}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-500">
                        {camera.location || 'Not specified'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-3 py-1.5 inline-flex items-center justify-center text-xs font-medium rounded-full shadow-sm ${
                          !camera.is_active
                            ? 'bg-gray-100 text-gray-800'
                            : processingStatus[camera.id]
                            ? 'bg-green-100 text-green-800 ring-1 ring-green-600/20'
                            : 'bg-amber-100 text-amber-800 ring-1 ring-amber-600/20'
                        }`}
                      >
                        {!camera.is_active
                          ? 'Inactive'
                          : processingStatus[camera.id]
                          ? 'Monitoring'
                          : 'Idle'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2">
                        <Link
                          to={`/cameras/${camera.id}`}
                          className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 transition-colors"
                        >
                          <FaEye className="mr-1.5 h-3.5 w-3.5" />
                          View
                        </Link>
                        
                        {camera.is_active && (
                          processingStatus[camera.id] ? (
                            <button
                              onClick={() => stopCamera(camera.id)}
                              className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
                            >
                              <FaStop className="mr-1.5 h-3.5 w-3.5" />
                              Stop
                            </button>
                          ) : (
                            <button
                              onClick={() => startCamera(camera.id)}
                              className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-green-50 text-green-700 hover:bg-green-100 transition-colors"
                            >
                              <FaPlay className="mr-1.5 h-3.5 w-3.5" />
                              Start
                            </button>
                          )
                        )}
                        
                        <button
                          onClick={() => deleteCamera(camera.id)}
                          className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
                        >
                          <FaTrash className="mr-1.5 h-3.5 w-3.5" />
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      
      {/* Add Camera Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen px-4">
            {/* Backdrop */}
            <div
              className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
              onClick={() => setShowAddModal(false)}
            ></div>
            
            {/* Modal */}
            <div className="bg-white rounded-xl overflow-hidden shadow-2xl transform transition-all sm:max-w-lg sm:w-full z-10 border border-gray-200">
              <div className="px-6 py-4 bg-white border-b border-gray-100">
                <h3 className="text-xl font-semibold text-gray-900">Add Camera</h3>
              </div>
              
              <form onSubmit={handleSubmit}>
                <div className="p-6 space-y-4">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                      Camera Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-400 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
                      required
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-1">
                      Stream URL
                    </label>
                    <input
                      type="text"
                      id="url"
                      name="url"
                      value={formData.url}
                      onChange={handleInputChange}
                      className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-400 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
                      placeholder="rtsp:// or http:// stream URL"
                      required
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      RTSP, HTTP or local video file path
                    </p>
                  </div>
                  
                  <div>
                    <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-1">
                      Location (Optional)
                    </label>
                    <input
                      type="text"
                      id="location"
                      name="location"
                      value={formData.location}
                      onChange={handleInputChange}
                      className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-400 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
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
                
                <div className="px-6 py-4 bg-gray-50 flex justify-end space-x-2 border-t border-gray-100">
                  <button
                    type="button"
                    className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-100 font-medium text-sm transition-colors"
                    onClick={() => setShowAddModal(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="inline-flex items-center px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 font-medium text-sm transition-colors shadow-sm"
                  >
                    Add Camera
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

export default CamerasList;