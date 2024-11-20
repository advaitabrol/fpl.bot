import React from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
} from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';
import OptimalTeam from './pages/OptimalTeam.tsx';
import PlayersPage from './pages/PlayerPage.tsx';
import YourTeam from './pages/YourTeam.tsx';

// Define a global theme style to match EPL aesthetics
const GlobalStyle = createGlobalStyle`
   html, body, #root {
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
  }
  
  body {
    margin: 0;
    font-family: 'Times New Roman', sans-serif;
    background-color: #fff; /* EPL-themed dark background */
    color: #000;
  }
`;

const Navbar = styled.nav`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1rem;
  background-color: #70ff72; /* EPL blue color */
  width: 100%; /* Full width */
  position: fixed; /* Fixes navbar at the top */
  top: 0; /* Aligns it to the top of the screen */
  left: 0; /* Aligns it to the left of the screen */
  z-index: 1000; /* Ensures it stays above other content */
`;

const NavItem = styled(NavLink)`
  color: #ffffff;
  text-decoration: none;
  font-size: 1.2rem;
  margin: 0 1.5rem;
  font-weight: bold;

  &.active {
    color: #37003c; /* EPL yellow accent for active link */
    border-bottom: 2px solid #37003c;
  }

  &:hover {
    color: #37003c;
  }
`;

const Main = styled.main`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

const ContentContainer = styled.div`
  display: flex;
  flex: 1;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  text-align: center;
`;

const App: React.FC = () => {
  return (
    <>
      <GlobalStyle />
      <Router>
        <Navbar>
          <NavItem to="/" end>
            Optimal Team(s)
          </NavItem>
          <NavItem to="/your-team">Your Team</NavItem>
          <NavItem to="/players">Players</NavItem>
        </Navbar>
        <Main>
          <ContentContainer>
            <Routes>
              <Route path="/" element={<OptimalTeam />} />
              <Route path="/your-team" element={<YourTeam />} />
              <Route path="/players" element={<PlayersPage />} />
            </Routes>
          </ContentContainer>
        </Main>
      </Router>
    </>
  );
};

export default App;
