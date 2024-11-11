import React from 'react';
import styled from 'styled-components';
import PlayerRow from '../aiTeamPlayer/PlayerRow';
import Bench from '../aiTeamPlayer/Bench';
import teamData from '../../data/testTeam.json';

const PageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  padding: 1rem;
`;

const TeamWrapper = styled.div`
  text-align: center;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  margin-top: 1.5rem;
`;

const TeamStatsBox = styled.div`
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  padding: 0.8rem; /* Reduced padding for a more compact look */
  border-radius: 8px;
  margin-bottom: 1.5rem;
  text-align: center;
  width: 250px; /* Slightly reduced width */
  min-height: 60px; /* Slightly reduced minimum height */
`;

const Stat = styled.div`
  font-size: 1rem;
  font-weight: bold;
`;

const FreeHitTeam: React.FC = () => {
  const currentWeek = 0;

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

  const projectedPoints = teamData.team
    .filter((player) => !player.isBench[currentWeek])
    .reduce((sum, player) => sum + player.expected_points[currentWeek], 0)
    .toFixed(2);

  return (
    <PageWrapper>
      <TeamStatsBox>
        <Stat>Projected Points: {projectedPoints} pts</Stat>
      </TeamStatsBox>

      <TeamWrapper>
        <SectionTitle>Starting XI</SectionTitle>
        <PlayerRow players={startingPlayers.GK} weekIndex={0} />
        <PlayerRow players={startingPlayers.DEF} weekIndex={0} />
        <PlayerRow players={startingPlayers.MID} weekIndex={0} />
        <PlayerRow players={startingPlayers.ATT} weekIndex={0} />
        <SectionTitle>Bench</SectionTitle>
        <Bench players={benchPlayers} weekIndex={0} />{' '}
        {/* Pass weekIndex to Bench */}
      </TeamWrapper>
    </PageWrapper>
  );
};

export default FreeHitTeam;
