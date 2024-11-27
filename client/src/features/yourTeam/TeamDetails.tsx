import React, { useState } from 'react';
import styled from 'styled-components';
import PlayerRow from '../team/PlayerRow';
import Bench from '../team/Bench';
import TeamActions from './TeamActions';

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
  justify-content: space-around;
  width: 100%;
  gap: 1rem;
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

interface TeamPlayer {
  name: string;
  team: string;
  position: 'GK' | 'DEF' | 'MID' | 'FWD';
  price: number;
  expected_points: number[];
  isBench: boolean[];
  isCaptain: boolean[];
}

interface TeamDetailsProps {
  teamData: {
    team_name: string;
    team: TeamPlayer[];
  };
}

const TeamDetails: React.FC<TeamDetailsProps> = ({ teamData }) => {
  const [team, setTeam] = useState(teamData.team);

  const calculateProjectedPoints = (team: TeamPlayer[]) => {
    return [0, 1, 2].map((weekIndex) =>
      team
        .filter((player) => !player.isBench[weekIndex])
        .reduce((sum, player) => sum + player.expected_points[weekIndex], 0)
        .toFixed(2)
    );
  };

  return (
    <div>
      {/* Display team name and projected points */}
      <TeamStatsContainer>
        <TeamStatsBox>
          <h3>{teamData.team_name}</h3>
          <StatsRow>
            {calculateProjectedPoints(team).map((points, index) => (
              <Stat key={index}>
                Week {index + 1}: {points}
              </Stat>
            ))}
          </StatsRow>
        </TeamStatsBox>
      </TeamStatsContainer>

      {/* Optimize and Suggest Transfers Buttons */}
      <TeamActions team={team} setTeam={setTeam} />

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
