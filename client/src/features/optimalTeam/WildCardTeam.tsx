import React from 'react';
import styled from 'styled-components';
import PlayerRow from '../team/PlayerRow';
import Bench from '../team/Bench';
import teamData from '../../data/testTeam.json';

const PageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  min-height: 100vh;
  padding: 1rem;
`;

const TeamWrapper = styled.div`
  text-align: center;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  margin-top: 0.5rem;
`;

const TeamStatsBox = styled.div`
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  padding: 0.8rem; /* Reduced padding for a more compact look */
  border-radius: 8px;
  margin-bottom: 0.5rem;
  text-align: center;
  width: 250px; /* Slightly reduced width */
  min-height: 60px; /* Slightly reduced minimum height */
`;

const StatsHeader = styled.h3`
  font-size: 1.1rem;
  font-weight: bold;
  margin: 0; /* Removed extra margin */
  text-align: center;
`;

const StatsRow = styled.div`
  display: flex;
  justify-content: space-around;
  width: 100%;
  gap: 1rem; /* Space between stats */
  margin-top: 0.5rem; /* Slight space below the header */
`;

const Stat = styled.div`
  font-size: 0.9rem;
  font-weight: bold;
  text-align: center;
`;

const WildCardTeam: React.FC = () => {
  const currentWeek = 0; // Use index 0 for the first week in expected_points

  // Filter players by position and bench status for the current week
  const startingPlayers = {
    GK: teamData.team.filter(
      (player) => player.position === 'GK' && !player.isBench[currentWeek]
    ),
    DEF: teamData.team.filter(
      (player) => player.position === 'DEF' && !player.isBench[currentWeek]
    ),
    MID: teamData.team.filter(
      (player) => player.position === 'MID' && !player.isBench[currentWeek]
    ),
    ATT: teamData.team.filter(
      (player) => player.position === 'ATT' && !player.isBench[currentWeek]
    ),
  };

  const benchPlayers = teamData.team.filter(
    (player) => player.isBench[currentWeek]
  );

  // Calculate projected points for Weeks 1, 2, and 3
  const projectedPoints = [0, 1, 2].map(
    (weekIndex) =>
      teamData.team
        .filter((player) => !player.isBench[weekIndex]) // Include only starting players
        .reduce((sum, player) => sum + player.expected_points[weekIndex], 0)
        .toFixed(2) // Round to two decimal places
  );

  return (
    <PageWrapper>
      <TeamStatsBox>
        <StatsHeader>Projected Points</StatsHeader>
        <StatsRow>
          <Stat>Week 1: {projectedPoints[0]}</Stat>
          <Stat>Week 2: {projectedPoints[1]}</Stat>
          <Stat>Week 3: {projectedPoints[2]}</Stat>
        </StatsRow>
      </TeamStatsBox>

      <TeamWrapper>
        <SectionTitle>Starting XI</SectionTitle>
        <PlayerRow players={startingPlayers.GK} />
        <PlayerRow players={startingPlayers.DEF} />
        <PlayerRow players={startingPlayers.MID} />
        <PlayerRow players={startingPlayers.ATT} />

        <SectionTitle>Bench</SectionTitle>
        <Bench players={benchPlayers} />
      </TeamWrapper>
    </PageWrapper>
  );
};

export default WildCardTeam;
