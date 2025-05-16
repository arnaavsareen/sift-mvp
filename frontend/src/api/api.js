import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add axios interceptor to handle trailing slashes
api.interceptors.request.use(config => {
  // Add trailing slash to GET requests if not present and not a fully qualified URL
  if (config.method?.toLowerCase() === 'get' && config.url && !config.url.includes('://')) {
    if (!config.url.endsWith('/') && !config.url.includes('?')) {
      config.url = `${config.url}/`;
    }
  }
  return config;
});

// WebSocket connection utilities
export const websocketApi = {
  // Create a camera stream connection
  createCameraStream: (cameraId, onFrame, onAlert, onError, onClose, options = {}) => {
    const createWebSocket = () => {
      // Create a new WebSocket connection - Fix the URL construction to avoid the double '/ws/ws/' path
      // Extract the base URL without the '/ws' suffix
      const baseWsUrl = WS_URL.endsWith('/ws') ? WS_URL.substring(0, WS_URL.length - 3) : WS_URL;
      const wsUrl = `${baseWsUrl}/ws/cameras/${cameraId}/stream`;
      
      console.log(`Connecting to WebSocket at: ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      
      // Connection status
      let wasConnected = false;
      let reconnectAttempts = 0;
      const maxReconnectAttempts = options.maxReconnectAttempts || 10;
      const baseReconnectDelay = options.reconnectInterval || 1000; // Start with 1 second
      const debug = options.debug || false;
      
      ws.onopen = () => {
        console.log(`WebSocket connection opened for camera ${cameraId}`);
        wasConnected = true;
        reconnectAttempts = 0; // Reset reconnect counter on successful connection
        
        // Send a ping immediately to verify the connection is working
        try {
          ws.send(JSON.stringify({ type: 'ping' }));
          if (debug) console.log("Sent initial ping");
        } catch (e) {
          console.error("Error sending initial ping:", e);
        }
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'ping') {
            // Just a ping to keep connection alive, no action needed
            if (debug) console.log("Ping received");
            return;
          } else if (data.type === 'frame' && typeof onFrame === 'function') {
            // Debug log for frame data
            if (debug) console.log(`Frame received, size: ${data.frame ? data.frame.length : 0} chars`);
            onFrame(data.frame, data.timestamp);
          } else if (data.type === 'alert' && typeof onAlert === 'function') {
            console.log("Alert received:", data.data);
            onAlert(data.data);
          } else if (data.type === 'error') {
            console.error(`WebSocket error from server: ${data.message}`);
            if (typeof onError === 'function') {
              onError(data.message);
            }
          } else {
            console.warn("Unknown message type:", data.type);
          }
        } catch (err) {
          console.error('Error processing WebSocket message:', err, event.data);
        }
      };
      
      // Set up ping interval to keep connection alive
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify({ type: 'ping' }));
            if (debug) console.log("Sent ping");
          } catch (e) {
            console.error("Error sending ping:", e);
          }
        }
      }, 30000); // Send ping every 30 seconds
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (typeof onError === 'function') {
          onError('Connection error occurred. Will try to reconnect automatically.');
        }
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket connection closed for camera ${cameraId}:`, event.code, event.reason);
        
        // Clear ping interval
        clearInterval(pingInterval);
        
        if (typeof onClose === 'function') {
          onClose(event);
        }
        
        // Attempt to reconnect with exponential backoff
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const reconnectDelay = Math.min(30000, baseReconnectDelay * Math.pow(1.5, reconnectAttempts - 1));
          
          console.log(`Attempting to reconnect WebSocket for camera ${cameraId} in ${reconnectDelay/1000} seconds (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          
          if (typeof onError === 'function') {
            onError(`Connection closed. Reconnecting in ${Math.round(reconnectDelay/1000)} seconds (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);
          }
          
          setTimeout(() => {
            if (ws.closedManually !== true) {
              const newConnection = createWebSocket();
              // Replace the old socket reference with the new one
              Object.assign(ws, newConnection);
            }
          }, reconnectDelay);
        } else if (wasConnected) {
          // Max reconnect attempts reached
          if (typeof onError === 'function') {
            onError(`Failed to reconnect after ${maxReconnectAttempts} attempts. Please try refreshing the page.`);
          }
          console.error(`WebSocket max reconnect attempts (${maxReconnectAttempts}) reached for camera ${cameraId}`);
        }
      };
      
      return ws;
    };
    
    // Create the initial WebSocket connection
    const ws = createWebSocket();
    
    // Custom property to track manual close
    ws.closedManually = false;
    
    // Return the WebSocket instance and a close function
    return {
      socket: ws,
      close: () => {
        if (ws && ws.readyState !== WebSocket.CLOSED) {
          ws.closedManually = true; // Mark as manually closed to prevent auto-reconnect
          ws.close();
        }
      }
    };
  }
};

// API functions for cameras
export const camerasApi = {
  // Get all cameras
  getAll: async () => {
    const response = await api.get('/cameras');
    return response.data;
  },
  
  // Get camera by ID
  getById: async (id) => {
    const response = await api.get(`/cameras/${id}`);
    return response.data;
  },
  
  // Create new camera
  create: async (data) => {
    const response = await api.post('/cameras', data);
    return response.data;
  },
  
  // Update camera
  update: async (id, data) => {
    const response = await api.put(`/cameras/${id}`, data);
    return response.data;
  },
  
  // Delete camera
  delete: async (id) => {
    const response = await api.delete(`/cameras/${id}`);
    return response.data || { status: "success" }; // Handle empty response
  },
  
  // Start camera processing
  start: async (id) => {
    const response = await api.post(`/cameras/${id}/start`);
    return response.data;
  },
  
  // Stop camera processing
  stop: async (id) => {
    const response = await api.post(`/cameras/${id}/stop`);
    return response.data;
  },
  
  // Get camera status
  getStatus: async (id) => {
    const response = await api.get(`/cameras/${id}/status`);
    return response.data;
  },
};

// API functions for alerts
export const alertsApi = {
  // Get all alerts with optional filters
  getAll: async (params = {}) => {
    const response = await api.get('/alerts', { params });
    return response.data;
  },
  
  // Get alert by ID
  getById: async (id) => {
    const response = await api.get(`/alerts/${id}`);
    return response.data;
  },
  
  // Resolve alert
  resolve: async (id, resolved = true) => {
    const response = await api.put(`/alerts/${id}/resolve`, { resolved });
    return response.data;
  },
  
  // Get alert stats
  getStats: async (params = {}) => {
    const response = await api.get('/alerts/stats/summary', { params });
    return response.data;
  },
};

// API functions for dashboard
export const dashboardApi = {
  // Get dashboard overview
  getOverview: async (params = {}) => {
    const response = await api.get('/dashboard/overview', { params });
    return response.data;
  },
  
  // Get latest frame from camera
  getLatestFrame: async (cameraId, params = {}) => {
    const response = await api.get(`/dashboard/cameras/${cameraId}/latest-frame`, { 
      params,
      responseType: params.format === 'base64' ? 'json' : 'blob'
    });
    return response.data;
  },
  
  // Get alert timeline data
  getTimeline: async (params = {}) => {
    const response = await api.get('/dashboard/timeline', { params });
    return response.data;
  },
};