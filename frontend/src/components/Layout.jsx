// frontend/src/components/Layout.jsx
import React, { useState, useEffect } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { 
  FaTachometerAlt, 
  FaVideo, 
  FaBell, 
  FaCog,
  FaBars,
  FaTimes,
  FaShieldAlt,
  FaHardHat,
  FaInfoCircle
} from 'react-icons/fa';

const Layout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const [pageTitle, setPageTitle] = useState('Dashboard');
  
  // Update page title based on current route
  useEffect(() => {
    const path = location.pathname;
    if (path.includes('/dashboard')) setPageTitle('Dashboard');
    else if (path.includes('/cameras')) {
      if (path.includes('/cameras/') && path.split('/cameras/')[1]) {
        setPageTitle('Camera Details');
      } else {
        setPageTitle('Cameras');
      }
    }
    else if (path.includes('/alerts')) {
      if (path.includes('/alerts/') && path.split('/alerts/')[1]) {
        setPageTitle('Alert Details');
      } else {
        setPageTitle('Safety Alerts');
      }
    }
    else setPageTitle('SIFT Platform');
  }, [location]);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  
  // Close sidebar when clicking a link on mobile
  const closeSidebarOnMobile = () => {
    if (window.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  };
  
  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black/40 backdrop-blur-sm transition-opacity lg:hidden"
          aria-label="Sidebar overlay"
          tabIndex={0}
          role="button"
          onClick={toggleSidebar}
        ></div>
      )}
      
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 transition-all duration-300 ease-in-out
          ${sidebarOpen ? 'w-64' : 'w-20'} 
          bg-white border-r border-gray-200
          lg:translate-x-0 lg:static lg:inset-0 
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
        aria-label="Sidebar"
      >
        <div className="flex h-full flex-col">
          {/* Sidebar header */}
          <div className="flex h-16 items-center justify-between px-4 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              <FaShieldAlt className="h-8 w-8 text-primary-600" />
              {sidebarOpen && (
                <span className="text-xl font-bold text-gray-900 tracking-tight">SIFT</span>
              )}
            </div>
            <button 
              onClick={toggleSidebar} 
              className="text-gray-500 hover:text-gray-700 p-2 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? <FaTimes size={20} /> : <FaBars size={20} />}
            </button>
          </div>
            
          {/* Sidebar navigation */}
          <nav className="flex-1 space-y-1 px-3 py-6" aria-label="Main navigation">
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 mb-3">Main</div>
            <NavLink
              to="/dashboard"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-3 ${
                  isActive
                    ? 'bg-primary-50 text-primary-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
              aria-current={({ isActive }) => (isActive ? 'page' : undefined)}
              title="Dashboard"
            >
              <FaTachometerAlt className="h-5 w-5" />
              {sidebarOpen && 'Dashboard'}
            </NavLink>
            <NavLink
              to="/cameras"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-3 ${
                  isActive
                    ? 'bg-primary-50 text-primary-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
              aria-current={({ isActive }) => (isActive ? 'page' : undefined)}
              title="Cameras"
            >
              <FaVideo className="h-5 w-5" />
              {sidebarOpen && 'Cameras'}
            </NavLink>
            <NavLink
              to="/alerts"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-3 ${
                  isActive
                    ? 'bg-primary-50 text-primary-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
              aria-current={({ isActive }) => (isActive ? 'page' : undefined)}
              title="Alerts"
            >
              <FaBell className="h-5 w-5" />
              {sidebarOpen && 'Alerts'}
            </NavLink>
            
            {sidebarOpen && (
              <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 mt-6 mb-3">Resources</div>
            )}
            <button
              className={`flex items-center w-full ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors gap-3`}
              aria-label="Safety Rules"
              title="Safety Rules"
            >
              <FaHardHat className="h-5 w-5" />
              {sidebarOpen && 'Safety Rules'}
            </button>
            <button
              className={`flex items-center w-full ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors gap-3`}
              aria-label="Documentation"
              title="Documentation"
            >
              <FaInfoCircle className="h-5 w-5" />
              {sidebarOpen && 'Documentation'}
            </button>
          </nav>

          {/* Sidebar footer with user profile */}
          <div className="mt-auto p-4 border-t border-gray-100 bg-white overflow-hidden">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-full bg-primary-100 flex items-center justify-center text-primary-600 font-semibold text-lg border border-gray-200 flex-shrink-0">A</div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">Arnav Sareen</p>
                <p className="text-xs text-gray-500 truncate">Admin</p>
              </div>
              <button 
                className="text-gray-400 hover:text-primary-600 p-2 rounded-lg hover:bg-gray-100 transition-all duration-200 border border-transparent focus:border-primary-200 flex-shrink-0" 
                aria-label="Account settings"
              >
                <FaCog className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 shadow-sm">
          <div className="flex h-16 items-center justify-between px-4 md:px-6 gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={toggleSidebar}
                className="text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg p-2 mr-2 lg:hidden"
                aria-label="Open sidebar"
              >
                <FaBars size={20} />
              </button>
              {/* Breadcrumb navigation */}
              <nav className="flex items-center text-sm" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-2">
                  <li>
                    <NavLink to="/dashboard" className="text-gray-500 hover:text-gray-700 font-medium">Dashboard</NavLink>
                  </li>
                  {location.pathname !== '/dashboard' && (
                    <li className="flex items-center">
                      <span className="mx-2 text-gray-400">/</span>
                      <span className="text-gray-900 font-medium">{pageTitle}</span>
                    </li>
                  )}
                </ol>
              </nav>
            </div>
            <div className="flex items-center gap-2">
              <button 
                className="rounded-lg hover:bg-gray-50 p-2 text-gray-500 hover:text-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500" 
                aria-label="Notifications"
              >
                <FaBell className="h-5 w-5" />
              </button>
              <button 
                className="rounded-lg hover:bg-gray-50 p-2 text-gray-500 hover:text-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500" 
                aria-label="Settings"
              >
                <FaCog className="h-5 w-5" />
              </button>
              <div className="hidden md:flex items-center ml-3 px-3 py-1.5 rounded-lg bg-primary-50 text-primary-600 text-sm font-medium">
                <FaShieldAlt className="mr-2 h-4 w-4" />
                Safety Mode
              </div>
            </div>
          </div>
        </header>
        
        {/* Main content area */}
        <main className="flex-1 overflow-auto p-4 md:p-6 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;