import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Layout and components
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import CamerasList from './components/CamerasList';
import CameraDetail from './components/CamerasDetail';
import AlertsList from './components/AlertsList';
import AlertDetail from './components/AlertsDetail';
import NotFound from './components/NotFound';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        {/* Default route */}
        <Route index element={<Navigate to="/dashboard" replace />} />
        
        {/* Main routes */}
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="cameras" element={<CamerasList />} />
        <Route path="cameras/:id" element={<CameraDetail />} />
        <Route path="alerts" element={<AlertsList />} />
        <Route path="alerts/:id" element={<AlertDetail />} />
        
        {/* 404 route */}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

export default App;