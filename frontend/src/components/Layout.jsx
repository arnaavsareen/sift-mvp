import React, { useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { 
  FaTachometerAlt, 
  FaVideo, 
  FaBell, 
  FaCog,
  FaBars,
  FaTimes
} from 'react-icons/fa';

const Layout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black bg-opacity-50 transition-opacity lg:hidden"
          onClick={toggleSidebar}
        ></div>
      )}
      
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 transform bg-primary-800 transition duration-300 lg:translate-x-0 lg:static lg:inset-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Sidebar header */}
          <div className="flex h-16 items-center justify-between px-4 text-white">
            <span className="text-xl font-bold">SIFT</span>
            <button onClick={toggleSidebar} className="lg:hidden">
              <FaTimes size={24} />
            </button>
          </div>
          
          {/* Sidebar navigation */}
          <nav className="flex-1 space-y-1 px-2 py-4">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `flex items-center px-4 py-2 text-sm font-medium rounded-md ${
                  isActive
                    ? 'bg-primary-900 text-white'
                    : 'text-white hover:bg-primary-700'
                }`
              }
            >
              <FaTachometerAlt className="mr-3 h-5 w-5" />
              Dashboard
            </NavLink>
            
            <NavLink
              to="/cameras"
              className={({ isActive }) =>
                `flex items-center px-4 py-2 text-sm font-medium rounded-md ${
                  isActive
                    ? 'bg-primary-900 text-white'
                    : 'text-white hover:bg-primary-700'
                }`
              }
            >
              <FaVideo className="mr-3 h-5 w-5" />
              Cameras
            </NavLink>
            
            <NavLink
              to="/alerts"
              className={({ isActive }) =>
                `flex items-center px-4 py-2 text-sm font-medium rounded-md ${
                  isActive
                    ? 'bg-primary-900 text-white'
                    : 'text-white hover:bg-primary-700'
                }`
              }
            >
              <FaBell className="mr-3 h-5 w-5" />
              Alerts
            </NavLink>
          </nav>
          
          {/* Sidebar footer */}
          <div className="p-4 text-sm text-white">
            <p>SIFT v0.1.0</p>
            <p>© 2023 Safety Labs</p>
          </div>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow">
          <div className="flex h-16 items-center justify-between px-4">
            <button
              onClick={toggleSidebar}
              className="text-gray-500 focus:outline-none lg:hidden"
            >
              <FaBars size={24} />
            </button>
            
            <h1 className="text-2xl font-bold text-gray-800">
              Safety Compliance Monitoring
            </h1>
            
            <div className="flex items-center">
              <button className="ml-4 rounded-full bg-gray-200 p-1">
                <FaCog className="h-6 w-6 text-gray-600" />
              </button>
            </div>
          </div>
        </header>
        
        {/* Main content area */}
        <main className="flex-1 overflow-auto p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;