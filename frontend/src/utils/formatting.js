/**
 * Formats a violation type string into consistent Title Case
 * Handles comma-separated lists and underscore or dash separated words
 * 
 * @param {string} type - The violation type string to format (e.g. "no_hardhat,no_vest")
 * @returns {string} The formatted string in Title Case (e.g. "No Hardhat, No Vest")
 */
export const formatViolationType = (type) => {
  if (!type) return '';
  
  return type
    .split(',')
    .map(part => 
      part
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ')
    )
    .join(', ');
};

/**
 * Formats a date string to a localized date and time format
 * 
 * @param {string} dateString - ISO date string
 * @param {Object} options - Formatting options for toLocaleString
 * @returns {string} Formatted date string
 */
export const formatDateTime = (dateString, options = {}) => {
  if (!dateString) return '';
  
  const dateObj = new Date(dateString);
  if (isNaN(dateObj.getTime())) return '';
  
  const defaultOptions = { 
    month: 'short', 
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit', 
    minute: '2-digit'
  };
  
  return dateObj.toLocaleString([], {...defaultOptions, ...options});
};

/**
 * Formats a confidence value (0-1) as a percentage
 * 
 * @param {number} confidence - Confidence value between 0 and 1
 * @returns {string} Formatted percentage string
 */
export const formatConfidence = (confidence) => {
  if (confidence === undefined || confidence === null) return '';
  
  return `${Math.round(confidence * 100)}%`;
}; 