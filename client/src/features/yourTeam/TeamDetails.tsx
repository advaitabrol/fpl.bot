import React, { useState } from 'react';
import styled from 'styled-components';
import PlayerRow from '../team/PlayerRow';
import Bench from '../team/Bench';
import TeamActions from './TeamActions';

import { Player, TeamDetailsResponse } from '../../services/interfaces';

const TeamStatsContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 1rem;
`;

const TeamStatsBox = styled.div`
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  padding: 0.8rem;
  border-radius: 8px;
  text-align: center;
  width: 300px;
`;

const StatsRow = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const Stat = styled.div`
  font-size: 0.9rem;
  font-weight: bold;
  text-align: center;
`;

const SectionTitle = styled.h2`
  margin: 1rem 0 0.5rem;
  font-size: 1.2rem;
  color: #333;
`;

const WeekColumn = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  border-radius: 8px;
  padding: 0.5rem;
  width: 80px; /* Adjust width as needed */
  text-align: center;
`;

const WeekLabel = styled.div`
  font-weight: bold;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  color: #555;
`;

const WeekPoints = styled.div`
  font-size: 1rem;
  font-weight: bold;
  color: #333;
`;

interface TeamDetailsProps {
  teamData: TeamDetailsResponse;
}

const TeamDetails: React.FC<TeamDetailsProps> = ({ teamData }) => {
  const [team, setTeam] = useState(teamData.team);
  const [bank, setBank] = useState(teamData.bank);

  const calculateProjectedPoints = (team: Player[]) => {
    return [0, 1, 2].map((weekIndex) =>
      team
        .filter((player) => !player.isBench[weekIndex])
        .reduce((sum, player) => sum + player.expected_points[weekIndex], 0)
        .toFixed(2)
    );
  };

  const projectedPoints = calculateProjectedPoints(team);

  return (
    <div>
      {/* Display team name, projected points, and bank amount */}
      <TeamStatsContainer>
        <TeamStatsBox>
          <h3>{teamData.team_name}</h3>
          <StatsRow>
            {projectedPoints.map((points, index) => (
              <WeekColumn key={index}>
                <WeekLabel>Week {index + 1}</WeekLabel>
                <WeekPoints>{points}</WeekPoints>
              </WeekColumn>
            ))}
          </StatsRow>
          <Stat>Bank: £{bank.toFixed(1)}M</Stat>
        </TeamStatsBox>
      </TeamStatsContainer>

      {/* Optimize and Suggest Transfers Buttons */}
      <TeamActions
        team={team}
        setTeam={setTeam}
        bank={bank}
        setBank={setBank}
      />

      {/* Display starting XI */}
      <SectionTitle>Starting XI</SectionTitle>
      {['GK', 'DEF', 'MID', 'FWD'].map((position) => (
        <PlayerRow
          key={position}
          players={team.filter(
            (player) => player.position === position && !player.isBench[0]
          )}
        />
      ))}

      {/* Display bench players */}
      <SectionTitle>Bench</SectionTitle>
      <Bench players={team.filter((player) => player.isBench[0])} />
    </div>
  );
};

export default TeamDetails;
