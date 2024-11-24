import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import PlayerRow from '../team/PlayerRow';
import Bench from '../team/Bench';
import { Player, TeamData } from './types';

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
  min-height: 60px;
`;

const StatsHeader = styled.h3`
  font-size: 1.1rem;
  font-weight: bold;
  margin: 0;
  text-align: center;
`;

const StatsRow = styled.div`
  display: flex;
  justify-content: space-around;
  width: 100%;
  gap: 1rem;
  margin-top: 0.5rem;
`;

const Stat = styled.div`
  font-size: 0.9rem;
  font-weight: bold;
  text-align: center;
`;

const WildCardTeam: React.FC = () => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const currentWeek = 0; // Assuming index 0 for the first week in `expected_points`

  useEffect(() => {
    const fetchTeamData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${import.meta.env.VITE_API_URL}/teams/wildcard-team`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch team data.');
        }
        const data: TeamData = await response.json();
        console.log('Fetched team data:', data); // Debug fetched data
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

  // Filter players by position and bench status for the current week
  const startingPlayers: { [key in Player['position']]: Player[] } = {
    GK: teamData.team.filter(
      (player) => player.position === 'GK' && !player.isBench?.[currentWeek]
    ),
    DEF: teamData.team.filter(
      (player) => player.position === 'DEF' && !player.isBench?.[currentWeek]
    ),
    MID: teamData.team.filter(
      (player) => player.position === 'MID' && !player.isBench?.[currentWeek]
    ),
    ATT: teamData.team.filter(
      (player) => player.position === 'ATT' && !player.isBench?.[currentWeek]
    ),
  };

  const benchPlayers: Player[] = teamData.team.filter(
    (player) => player.isBench?.[currentWeek]
  );

  // Calculate projected points for Weeks 1, 2, and 3
  const projectedPoints: string[] = [0, 1, 2].map((weekIndex) =>
    teamData.team
      .filter((player) => !player.isBench?.[weekIndex])
      .reduce(
        (sum, player) => sum + (player.expected_points?.[weekIndex] || 0),
        0
      )
      .toFixed(2)
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
