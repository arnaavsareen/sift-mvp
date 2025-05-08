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
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black bg-opacity-70 backdrop-blur-sm transition-opacity lg:hidden"
          onClick={toggleSidebar}
        ></div>
      )}
      
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 transform bg-primary transition-all duration-300 ease-in-out shadow-xl lg:translate-x-0 lg:static lg:inset-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Sidebar header */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-primary-700">
            <div className="flex items-center space-x-2">
              <FaShieldAlt className="h-6 w-6 text-light" />
              <span className="text-xl font-bold text-light tracking-wide">SIFT</span>
            </div>
            <button 
              onClick={toggleSidebar} 
              className="lg:hidden text-light hover:text-white focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-20 rounded"
            >
              <FaTimes size={20} />
            </button>
          </div>
          
          {/* Sidebar navigation */}
          <nav className="flex-1 space-y-1 px-3 py-6">
            <div className="text-xs font-semibold text-secondary uppercase tracking-wider px-3 mb-3">Main</div>
            <NavLink
              to="/dashboard"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                  isActive
                    ? 'bg-background text-light shadow-sm'
                    : 'text-light hover:bg-primary-700'
                }`
              }
            >
              <FaTachometerAlt className="mr-3 h-5 w-5" />
              Dashboard
            </NavLink>
            
            <NavLink
              to="/cameras"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                  isActive
                    ? 'bg-background text-light shadow-sm'
                    : 'text-light hover:bg-primary-700'
                }`
              }
            >
              <FaVideo className="mr-3 h-5 w-5" />
              Cameras
            </NavLink>
            
            <NavLink
              to="/alerts"
              onClick={closeSidebarOnMobile}
              className={({ isActive }) =>
                `flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                  isActive
                    ? 'bg-background text-light shadow-sm'
                    : 'text-light hover:bg-primary-700'
                }`
              }
            >
              <FaBell className="mr-3 h-5 w-5" />
              <div className="flex justify-between items-center w-full">
                <span>Alerts</span>
                <span className="inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none bg-danger text-white rounded-full">4</span>
              </div>
            </NavLink>

            <div className="text-xs font-semibold text-secondary uppercase tracking-wider px-3 mt-6 mb-3">Resources</div>
            <button className="flex items-center w-full text-left px-4 py-3 text-sm font-medium rounded-lg text-light hover:bg-primary-700 transition-colors">
              <FaHardHat className="mr-3 h-5 w-5" />
              Safety Rules
            </button>
            <button className="flex items-center w-full text-left px-4 py-3 text-sm font-medium rounded-lg text-light hover:bg-primary-700 transition-colors">
              <FaInfoCircle className="mr-3 h-5 w-5" />
              Documentation
            </button>
          </nav>
          
          {/* Sidebar footer */}
          <div className="p-4 border-t border-primary-700 text-xs text-secondary">
            <div className="flex items-center justify-between">
              <p>SIFT v1.0.0</p>
              <p>OSHA Compliance</p>
            </div>
          </div>
        </div>
      </aside>
      
      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-surface shadow-md z-10">
          <div className="flex h-16 items-center justify-between px-6">
            <div className="flex items-center">
              <button
                onClick={toggleSidebar}
                className="text-light hover:text-white focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-20 rounded p-1 mr-4 lg:hidden"
              >
                <FaBars size={20} />
              </button>
            
              <h1 className="text-xl font-bold text-light hidden sm:block">
                {pageTitle}
              </h1>
              
              <div className="ml-4 text-xs text-secondary rounded-md px-2 py-1 bg-surface-light hidden md:block">
                PPE Detection Active
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button className="rounded-full hover:bg-surface-light p-2 text-light transition-colors focus:outline-none focus:ring-2 focus:ring-primary">
                <FaBell className="h-5 w-5" />
              </button>
              <button className="rounded-full hover:bg-surface-light p-2 text-light transition-colors focus:outline-none focus:ring-2 focus:ring-primary">
                <FaCog className="h-5 w-5" />
              </button>
              <div className="hidden md:flex items-center ml-3 px-3 py-1 rounded-full bg-primary text-light text-sm font-medium">
                <FaShieldAlt className="mr-2 h-4 w-4" />
                SIFT Monitor
              </div>
            </div>
          </div>
        </header>
        
        {/* Main content area */}
        <main className="flex-1 overflow-auto p-6 bg-background">
          <div className="container mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;