

import React from 'react';
import { Link } from 'react-router-dom';
import './homePage.css';

function HomePage() {
    return (
        <div className="homepage">
            <h1>Welcome to the Fantasy Team Optimizer</h1>
            <p>
                Use this tool to generate an optimal fantasy team, or fetch your existing team 
                by entering your team name to get personalized transfer suggestions.
            </p>
            <div className="navigation-links">
                <Link to="/generate-team">Generate Optimal Team</Link>
                <Link to="/fetch-team">Fetch My Team by Name</Link>
                <Link to="/suggest-transfers">Suggest Transfers</Link>
            </div>
        </div>
    );
}

export default HomePage;
