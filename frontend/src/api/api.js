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

// WebSocket connection utilities
export const websocketApi = {
  // Create a camera stream connection
  createCameraStream: (cameraId, onFrame, onAlert, onError, onClose) => {
    const ws = new WebSocket(`${WS_URL}/cameras/${cameraId}/stream`);
    
    ws.onopen = () => {
      console.log(`WebSocket connection opened for camera ${cameraId}`);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame' && typeof onFrame === 'function') {
          onFrame(data.frame, data.timestamp);
        } else if (data.type === 'alert' && typeof onAlert === 'function') {
          onAlert(data.data);
        } else if (data.type === 'error') {
          console.error(`WebSocket error from server: ${data.message}`);
          if (typeof onError === 'function') {
            onError(data.message);
          }
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (typeof onError === 'function') {
        onError(error);
      }
    };
    
    ws.onclose = (event) => {
      console.log(`WebSocket connection closed for camera ${cameraId}:`, event.code, event.reason);
      if (typeof onClose === 'function') {
        onClose(event);
      }
    };
    
    // Return the WebSocket instance and a close function
    return {
      socket: ws,
      close: () => {
        if (ws && ws.readyState !== WebSocket.CLOSED) {
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
    return response.data;
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