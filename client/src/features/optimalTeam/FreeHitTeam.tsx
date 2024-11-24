import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import PlayerRow from '../team/PlayerRow';
import Bench from '../team/Bench';
import { TeamData, Player } from './types'; // Ensure correct import

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
  padding: 0.8rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
  text-align: center;
  width: 250px;
  min-height: 30px;
`;

const Stat = styled.div`
  font-size: 1rem;
  font-weight: bold;
`;

const FreeHitTeam: React.FC = () => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const currentWeek = 0; // Assuming current week is 0-based index

  useEffect(() => {
    const fetchTeamData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${import.meta.env.VITE_API_URL}/teams/freehit-team`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch team data.');
        }
        const data: TeamData = await response.json(); // Specify expected type
        console.log(data);
        setTeamData(data);
      } catch (err: any) {
        setError(err.message || 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchTeamData();
  }, []);

  if (loading) {
    return <PageWrapper>Loading...</PageWrapper>;
  }

  if (error) {
    return <PageWrapper>Error: {error}</PageWrapper>;
  }

  if (!teamData || !teamData.team) {
    return <PageWrapper>No team data available.</PageWrapper>;
  }

  // Safeguard: Ensure teamData and its properties are defined
  const startingPlayers = {
    GK: teamData.team.filter(
      (player: Player) => player.position === 'GK' && !player.isBench[0]
    ),
    DEF: teamData.team.filter(
      (player: Player) => player.position === 'DEF' && !player.isBench[0]
    ),
    MID: teamData.team.filter(
      (player: Player) => player.position === 'MID' && !player.isBench[0]
    ),
    ATT: teamData.team.filter(
      (player: Player) => player.position === 'ATT' && !player.isBench[0]
    ),
  };

  const benchPlayers = teamData.team.filter(
    (player: Player) => player.isBench[0]
  );

  const projectedPoints = teamData.team
    .filter((player: Player) => !player.isBench[0])
    .reduce((sum: number, player: Player) => sum + player.expected_points[0], 0)
    .toFixed(2);

  return (
    <PageWrapper>
      <TeamStatsBox>
        <Stat>Projected Points: {projectedPoints} pts</Stat>
      </TeamStatsBox>

      <TeamWrapper>
        <SectionTitle>Starting XI</SectionTitle>
        <PlayerRow players={startingPlayers.GK} weekIndex={currentWeek} />
        <PlayerRow players={startingPlayers.DEF} weekIndex={currentWeek} />
        <PlayerRow players={startingPlayers.MID} weekIndex={currentWeek} />
        <PlayerRow players={startingPlayers.ATT} weekIndex={currentWeek} />
        <SectionTitle>Bench</SectionTitle>
        <Bench players={benchPlayers} weekIndex={currentWeek} />
      </TeamWrapper>
    </PageWrapper>
  );
};

export default FreeHitTeam;
