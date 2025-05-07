import React from 'react';
import { Link } from 'react-router-dom';
import { FaExclamationTriangle, FaHome } from 'react-icons/fa';

const NotFound = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-6">
      <FaExclamationTriangle className="text-warning-500 h-16 w-16 mb-6" />
      <h1 className="text-4xl font-bold text-gray-900 mb-4">404 - Page Not Found</h1>
      <p className="text-lg text-gray-600 mb-8 max-w-md">
        The page you are looking for does not exist or has been moved to another location.
      </p>
      <Link to="/" className="btn-primary inline-flex items-center">
        <FaHome className="mr-2" />
        Return to Dashboard
      </Link>
    </div>
  );
};

export default NotFound;