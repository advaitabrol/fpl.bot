//Main app entry

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/homePage';
// Import other components for different pages
// import GenerateTeam from './components/GenerateTeam';
// import FetchTeam from './components/FetchTeam';
// import SuggestTransfers from './components/SuggestTransfers';

function App() {
  return (
    <div>
      <h1>Welcome to the Home Page</h1>
      <p>This is a simple React app.</p>
    </div>
  );
}

export default App;

