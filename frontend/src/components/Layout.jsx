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
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black bg-opacity-40 backdrop-blur-sm transition-opacity lg:hidden"
          aria-label="Sidebar overlay"
          tabIndex={0}
          role="button"
          onClick={toggleSidebar}
        ></div>
      )}
      
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 transition-all duration-300 ease-in-out shadow-2xl border-r border-green-main
          ${sidebarOpen ? 'w-56 bg-green-main' : 'w-16 bg-green-main'}
          lg:translate-x-0 lg:static lg:inset-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
        aria-label="Sidebar"
      >
        <div className="flex h-full flex-col">
          {/* Sidebar header */}
          <div className="flex h-16 items-center justify-between px-2 border-b border-primary-800">
  <div className="flex items-center space-x-2">
    <FaShieldAlt className="h-7 w-7 text-white" />
    {sidebarOpen && <span className="text-lg font-extrabold text-white tracking-wide transition-all duration-200">SIFT</span>}
  </div>
  <button 
    onClick={toggleSidebar} 
    className="text-white bg-green-main hover:bg-green-main/90 p-2 rounded-full transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-green-main"
    aria-label="Toggle sidebar"
  >
    {sidebarOpen ? <FaTimes size={20} /> : <FaBars size={20} />}
  </button>
</div>
            
          {/* Sidebar navigation */}
          <nav className="flex-1 space-y-1 px-3 py-6" aria-label="Main navigation">
            <div className="text-xs font-semibold text-secondary uppercase tracking-wider px-3 mb-3">Main</div>
            <NavLink
              to="/dashboard"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-2 ${
                  isActive
                    ? 'bg-white text-green-main shadow-sm'
                    : 'text-secondary hover:bg-green-main hover:text-white'
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
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-2 ${
                  isActive
                    ? 'bg-white text-green-main shadow-sm'
                    : 'text-secondary hover:bg-green-main hover:text-white'
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
                `flex items-center ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg transition-colors gap-2 ${
                  isActive
                    ? 'bg-white text-green-main shadow-sm'
                    : 'text-secondary hover:bg-green-main hover:text-white'
                }`
              }
              aria-current={({ isActive }) => (isActive ? 'page' : undefined)}
              title="Alerts"
            >
              <FaBell className="h-5 w-5" />
              {sidebarOpen && 'Alerts'}
            </NavLink>
            {sidebarOpen && (
              <div className="text-xs font-semibold text-secondary uppercase tracking-wider px-3 mt-6 mb-3">Resources</div>
            )}
            <button
              className={`flex items-center w-full ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg text-secondary hover:bg-green-main hover:text-white transition-colors gap-2`}
              aria-label="Safety Rules"
              title="Safety Rules"
            >
              <FaHardHat className="h-5 w-5" />
              {sidebarOpen && 'Safety Rules'}
            </button>
            <button
              className={`flex items-center w-full ${sidebarOpen ? 'px-4' : 'justify-center'} py-3 text-sm font-medium rounded-lg text-secondary hover:bg-green-main hover:text-white transition-colors gap-2`}
              aria-label="Documentation"
              title="Documentation"
            >
              <FaInfoCircle className="h-5 w-5" />
              {sidebarOpen && 'Documentation'}
            </button>
          </nav>
          {/* Sidebar footer with user profile */}
          <div className="mt-auto p-4 border-t border-green-main bg-white shadow-inner">
  <div className="flex items-center gap-3">
    <div className="h-10 w-10 rounded-full bg-green-main flex items-center justify-center text-white font-extrabold text-lg shadow">A</div>
    <div className="flex-1">
      <p className="text-sm font-semibold text-green-main">Arnav Sareen</p>
      <p className="text-xs text-gray-400">Admin</p>
    </div>
    <button className="ml-auto text-green-main hover:bg-green-main/10 p-2 rounded-full transition-all duration-200" aria-label="Account settings">
      <FaCog className="h-5 w-5" />
    </button>
  </div>
</div>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-lg z-10 border-b border-green-main">
          <div className="flex h-16 items-center justify-between px-4 md:px-6 gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={toggleSidebar}
                className="text-primary-600 hover:text-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-600 rounded p-1 mr-2 lg:hidden"
                aria-label="Open sidebar"
              >
                <FaBars size={20} />
              </button>
              {/* Breadcrumb navigation */}
              <nav className="flex items-center text-sm text-gray-500" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-2">
                  <li>
                    <NavLink to="/dashboard" className="hover:text-primary-600 font-medium">Dashboard</NavLink>
                  </li>
                  {location.pathname !== '/dashboard' && (
                    <li className="flex items-center">
                      <span className="mx-2">/</span>
                      <span className="capitalize text-gray-700 font-semibold">{pageTitle}</span>
                    </li>
                  )}
                </ol>
              </nav>
              <div className="ml-4 text-xs text-primary-700 rounded-md px-2 py-1 bg-secondary hidden md:block">
                
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="rounded-full hover:bg-primary-100 p-2 text-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-600" aria-label="Notifications">
                <FaBell className="h-5 w-5" />
              </button>
              <button className="rounded-full hover:bg-primary-100 p-2 text-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-600" aria-label="Settings">
                <FaCog className="h-5 w-5" />
              </button>
              <div className="hidden md:flex items-center ml-3 px-3 py-1 rounded-full bg-primary-600 text-white text-sm font-medium">
                <FaShieldAlt className="mr-2 h-4 w-4" />
                
              </div>
            </div>
          </div>
        </header>
        {/* Main content area */}
        <main className="flex-1 overflow-auto p-4 md:p-6 bg-white">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;