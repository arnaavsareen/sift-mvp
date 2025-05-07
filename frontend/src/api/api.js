import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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