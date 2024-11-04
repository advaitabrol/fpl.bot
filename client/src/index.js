//React entry point
import React from 'react';
import ReactDOM from 'react-dom/client';  // Notice the '/client' for React 18+
import './index.css';  // Optional: Ensure this file exists or remove this line
import App from './App';

// Create a root and render your App
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
