import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { camerasApi, dashboardApi, alertsApi, websocketApi } from '../api/api';
import Hls from 'hls.js';
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
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const hlsRef = useRef(null);
  
  const [camera, setCamera] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [wsError, setWsError] = useState(null);
  const [isVideoMode, setIsVideoMode] = useState(true);
  const [streamType, setStreamType] = useState('websocket'); // Default to WebSocket
  
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
      
      // Check if this is a direct video URL (file, RTSP, HTTP video)
      const isVideoUrl = data.url.startsWith('file://') || 
                         data.url.startsWith('rtsp://') || 
                         data.url.startsWith('http://') ||
                         data.url.startsWith('https://') ||
                         data.url.endsWith('.mp4') ||
                         data.url.endsWith('.mov') ||
                         data.url.endsWith('.webm');
      
      setIsVideoMode(isVideoUrl);
      
      // Determine best streaming method based on URL type
      if (data.url.startsWith('rtsp://')) {
        // For RTSP streams, use WebSocket as first choice instead of HLS
        setStreamType('websocket');
      } else if (isVideoUrl) {
        // For other video URLs, try direct video playback
        setStreamType('video');
      } else {
        // Default to WebSocket for other URLs
        setStreamType('websocket');
      }
      
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
      setError('Failed to load camera details. Please try again.');
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
  
  // Clean up all streaming connections when component unmounts
  useEffect(() => {
    return () => {
      // Clean up video element
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.src = "";
        videoRef.current.load();
      }
      
      // Clean up HLS
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
      
      // Clean up WebSocket
      if (wsRef.current) {
        wsRef.current.closedManually = true;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);
  
  // Handle direct video playback
  useEffect(() => {
    if (!isProcessing || !isVideoMode || !camera || !videoRef.current) return;
    if (streamType !== 'video') return;
    
    let playAttempts = 0;
    const maxPlayAttempts = 3;
    
    const handlePlayError = (err) => {
      console.error("Error playing video:", err);
      
      playAttempts++;
      if (playAttempts >= maxPlayAttempts) {
        console.log(`Failed after ${playAttempts} attempts. Trying HLS mode...`);
        setWsError("Failed to play video. Switching to HLS streaming...");
        setStreamType('hls');
      } else {
        setWsError(`Failed to play video. Retry attempt ${playAttempts}/${maxPlayAttempts}...`);
        
        // Try again with a delay and a fresh URL
        setTimeout(loadVideo, 1000);
      }
    };
    
    const loadVideo = () => {
      // Always use our streaming endpoint for all URL types
      const streamingUrl = `${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/cameras/${id}/stream?format=video&t=${new Date().getTime()}&retry=${playAttempts}`;
      console.log("Setting video source to streaming URL:", streamingUrl);
      
      // Set a timeout to detect if video loading takes too long
      const timeoutId = setTimeout(() => {
        if (videoRef.current && videoRef.current.readyState === 0) {
          console.warn("Video loading timed out");
          handlePlayError(new Error("Video loading timed out"));
        }
      }, 10000); // 10 second timeout
      
      // Clean up the timeout if video starts loading
      videoRef.current.onloadedmetadata = () => {
        console.log("Video metadata loaded successfully");
        clearTimeout(timeoutId);
        setWsError(null);
      };
      
      videoRef.current.onplaying = () => {
        console.log("Video is playing successfully");
        setWsError(null);
        playAttempts = 0; // Reset attempts when playing successfully
      };
      
      videoRef.current.onerror = (e) => {
        console.error("Video element error:", e);
        clearTimeout(timeoutId);
        
        // Get detailed error information
        const errorCode = videoRef.current.error ? videoRef.current.error.code : 0;
        const errorMessage = videoRef.current.error ? videoRef.current.error.message : "Unknown error";
        console.error(`Video error (${errorCode}): ${errorMessage}`);
        
        handlePlayError(new Error(`Video error (${errorCode}): ${errorMessage}`));
      };
      
      // Set the video source and attempt to play
      videoRef.current.src = streamingUrl;
      videoRef.current.load();
      videoRef.current.play().catch(handlePlayError);
    };
    
    // Initial load attempt
    loadVideo();
    
    return () => {
      if (videoRef.current) {
        clearTimeout(videoRef.current.timeoutId);
        videoRef.current.onloadedmetadata = null;
        videoRef.current.onplaying = null;
        videoRef.current.onerror = null;
        videoRef.current.pause();
        videoRef.current.src = "";
        videoRef.current.load();
      }
    };
  }, [isProcessing, isVideoMode, camera, id, streamType]);
  
  // Handle HLS streaming
  useEffect(() => {
    if (!isProcessing || !isVideoMode || !camera || !videoRef.current) return;
    if (streamType !== 'hls') return;
    
    const setupHls = () => {
      // Clean up existing HLS instance if any
      if (hlsRef.current) {
        hlsRef.current.destroy();
      }
      
      // Check if HLS.js is supported in this browser
      if (!Hls.isSupported()) {
        console.error("HLS.js is not supported in this browser");
        setWsError("HLS streaming not supported in this browser. Falling back to WebSocket mode.");
        setStreamType('websocket');
        return;
      }
      
      const hlsUrl = `${process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}/cameras/${id}/stream?format=hls&t=${new Date().getTime()}`;
      console.log("Setting up HLS stream with URL:", hlsUrl);
      
      try {
        // Create a new HLS instance
        const hls = new Hls({
          debug: false,
          enableWorker: true,
          lowLatencyMode: true,
          backBufferLength: 30
        });
        
        // Handle HLS events
        hls.on(Hls.Events.MEDIA_ATTACHED, () => {
          console.log("HLS: Media attached");
          setWsError("Starting HLS stream...");
        });
        
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          console.log("HLS: Manifest parsed, playback starting");
          setWsError(null);
          videoRef.current.play().catch(err => {
            console.error("Failed to auto-play HLS stream:", err);
          });
        });
        
        hls.on(Hls.Events.ERROR, (event, data) => {
          console.error("HLS error:", data);
          if (data.fatal) {
            switch(data.type) {
              case Hls.ErrorTypes.NETWORK_ERROR:
                setWsError("HLS network error. Retrying...");
                hls.startLoad();
                break;
              case Hls.ErrorTypes.MEDIA_ERROR:
                setWsError("HLS media error. Recovering...");
                hls.recoverMediaError();
                break;
              default:
                setWsError("Fatal HLS error. Falling back to WebSocket mode.");
                setStreamType('websocket');
                break;
            }
          }
        });
        
        // Bind HLS to the video element
        hls.loadSource(hlsUrl);
        hls.attachMedia(videoRef.current);
        hlsRef.current = hls;
        
        // Request first few frames
        hls.startLoad();
      } catch (error) {
        console.error("Failed to setup HLS:", error);
        setWsError("Failed to setup HLS streaming. Falling back to WebSocket mode.");
        setStreamType('websocket');
      }
    };
    
    // Start HLS streaming
    setupHls();
    
    return () => {
      // Clean up HLS
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [isProcessing, isVideoMode, camera, id, streamType]);
  
  // Update WebSocket handling to be more robust
  useEffect(() => {
    if (!isProcessing || (isVideoMode && streamType !== 'websocket')) {
      // Clean up any existing connection
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }
    
    // Start WebSocket connection for streaming frames
    const frameImg = document.getElementById('frame-img');
    const errorMessage = document.getElementById('camera-error-message');
    
    if (frameImg) {
      console.log("Setting up WebSocket streaming mode");
      setWsError("Connecting via WebSocket for frame-by-frame streaming...");
      
      // Force loading indicator to show until first frame arrives
      frameImg.src = '';
      
      // Show loading indicator
      const loadingIndicator = document.getElementById('loading-indicator');
      if (loadingIndicator) {
        loadingIndicator.classList.remove('hidden');
      }
      
      if (errorMessage) {
        errorMessage.classList.add('hidden');
      }
      
      // Initialize WebSocket for real-time frame streaming with enhanced options
      wsRef.current = websocketApi.createCameraStream(
        id,
        // onFrame handler
        (frameData, timestamp) => {
          if (!frameImg || !frameData) return;
          
          // Hide the loading indicator when frames start arriving
          const loadingIndicator = document.getElementById('loading-indicator');
          if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
          }
          
          // Hide error message if it's visible
          if (errorMessage) {
            errorMessage.classList.add('hidden');
          }
          
          // Show the frame image if it's hidden
          if (frameImg.classList.contains('hidden')) {
            frameImg.classList.remove('hidden');
          }
          
          // Set the image source to the received frame
          frameImg.src = `data:image/jpeg;base64,${frameData}`;
          
          // Reset error state
          setWsError(null);
        },
        // onAlert handler (not used here, alerts handled separately)
        null,
        // onError handler
        (errorMsg) => {
          console.error(`WebSocket error for camera ${id}:`, errorMsg);
          setWsError(`WebSocket error: ${errorMsg}. Attempting to reconnect...`);
          
          // Show error message if error persists
          if (errorMsg.includes('Failed to reconnect') && errorMessage) {
            frameImg.classList.add('hidden');
            errorMessage.classList.remove('hidden');
          }
        },
        // onClose handler
        (event) => {
          console.log(`WebSocket closed for camera ${id}:`, event);
        },
        // Enhanced options
        {
          maxReconnectAttempts: 10,  // Increase from default of 5
          reconnectInterval: 2000,   // 2 seconds between reconnect attempts
          debug: true                // Enable detailed logging
        }
      );
    }
    
    // Cleanup function to close WebSocket when component unmounts or deps change
    return () => {
      if (wsRef.current) {
        // Set closedManually to prevent auto-reconnect
        wsRef.current.closedManually = true;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [id, isProcessing, isVideoMode, streamType]);
  
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
  
  // Retry streaming with different method
  const retryStream = () => {
    // Try each streaming method in sequence
    if (streamType === 'video') {
      setStreamType('hls');
    } else if (streamType === 'hls') {
      setStreamType('websocket');
    } else {
      // If we're already on websocket, restart with direct video again
      setStreamType('video');
    }
    
    setWsError(`Trying ${streamType} streaming mode...`);
  };
  
  // Render the camera view based on type
  const renderCameraView = () => {
    if (!isProcessing) {
      return (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 text-white">
          <FaVideo className="h-16 w-16 text-gray-400 mb-4" />
          <p className="text-xl font-medium text-center mb-2">No Active Video Feed</p>
          <p className="text-gray-400 max-w-md text-center mb-6">
            Camera is not currently monitoring. 
            Click "Start Monitoring" to begin processing the video feed.
          </p>
          <button
            onClick={startCamera}
            className="inline-flex items-center px-4 py-2.5 rounded-lg bg-green-600 text-white hover:bg-green-700 font-medium text-sm transition-colors shadow-sm"
          >
            <FaPlay className="mr-2 h-4 w-4" />
            <span>Start Monitoring</span>
          </button>
        </div>
      );
    }

    return (
      <>
        {isVideoMode && (streamType === 'video' || streamType === 'hls') ? (
          // For direct video or HLS feeds, use the HTML5 video element
          <video
            ref={videoRef}
            className="w-full h-full object-contain bg-black"
            autoPlay
            playsInline
            muted
            controls={false}
          />
        ) : (
          // For WebSocket streaming, use image-based approach
          <div className="relative w-full h-full">
            <img
              id="frame-img"
              alt="Camera feed"
              className="w-full h-full object-contain bg-black"
              onError={(e) => {
                console.log("Image error occurred");
                e.target.classList.add('hidden');
                document.getElementById('camera-error-message')?.classList.remove('hidden');
              }}
              style={{
                display: 'block', /* Always show unless explicitly hidden */
                maxHeight: '100%',
                maxWidth: '100%'
              }}
            />
            <div 
              id="loading-indicator" 
              className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50 z-10 hidden"
            >
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
            </div>
          </div>
        )}
        
        {/* Overlay with camera information */}
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
        
        {/* Stream type indicator */}
        <div className="absolute top-0 right-0 p-3">
          <div className="bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
            {streamType.toUpperCase()} MODE
          </div>
        </div>
        
        {/* Timestamp */}
        <div className="absolute bottom-3 right-3 bg-black bg-opacity-50 px-2 py-1 rounded text-xs text-white">
          {new Date().toLocaleTimeString()}
        </div>
        
        {/* Error message overlay (hidden by default) */}
        <div id="camera-error-message" className="hidden absolute inset-0 flex flex-col items-center justify-center bg-gray-800 text-white">
          <FaExclamationTriangle className="h-16 w-16 text-yellow-500 mb-4" />
          <p className="text-xl font-medium text-center mb-2">Connection Error</p>
          <p className="text-gray-400 max-w-md text-center mb-6">
            {wsError || "Unable to load camera feed. The camera might be offline or the stream URL is invalid."}
          </p>
          <div className="flex space-x-4">
            <button
              onClick={retryStream}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center space-x-2"
            >
              <span>Try Different Method</span>
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
      </>
    );
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
        <div className="flex space-x-3">
          {camera.is_active && (
            isProcessing ? (
              <button
                onClick={stopCamera}
                className="inline-flex items-center px-4 py-2.5 rounded-lg bg-red-600 text-white hover:bg-red-700 font-medium text-sm transition-colors shadow-sm"
              >
                <FaStop className="mr-2 h-4 w-4" />
                Stop Monitoring
              </button>
            ) : (
              <button
                onClick={startCamera}
                className="inline-flex items-center px-4 py-2.5 rounded-lg bg-green-600 text-white hover:bg-green-700 font-medium text-sm transition-colors shadow-sm"
              >
                <FaPlay className="mr-2 h-4 w-4" />
                Start Monitoring
              </button>
            )
          )}
          <button
            onClick={() => setShowEditModal(true)}
            className="inline-flex items-center px-4 py-2.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-100 font-medium text-sm transition-colors shadow-sm"
          >
            <FaEdit className="mr-2 h-4 w-4" />
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
                {renderCameraView()}
              </div>
            </div>
            
            {isProcessing && (
              <div className="mt-4 text-sm flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`w-3 h-3 mr-2 rounded-full ${wsError ? 'bg-danger-500' : 'bg-success-500'}`}></div>
                  <p className={wsError ? 'text-danger-700' : 'text-success-700'}>
                    {wsError 
                      ? `Connection issue: ${wsError}` 
                      : isVideoMode ? "Direct video stream" : "Connected via WebSocket"}
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
            <div className="bg-white rounded-xl overflow-hidden shadow-2xl transform transition-all sm:max-w-lg sm:w-full z-10 border border-gray-200">
              <div className="px-6 py-4 bg-white border-b border-gray-100">
                <h3 className="text-xl font-semibold text-gray-900">Edit Camera</h3>
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
                    onClick={() => setShowEditModal(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="inline-flex items-center px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 font-medium text-sm transition-colors shadow-sm"
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