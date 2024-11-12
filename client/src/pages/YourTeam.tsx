import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import PlayerRow from '../features/team/PlayerRow';
import Bench from '../features/team/Bench';
import testTeamNames from '../data/testTeamNames.json';
import testTeamData from '../data/testTeamData.json';
import testOptimizedTeam from '../data/testOptimizedTeam.json';

const PageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start; /* Aligns content to start of the container */
  height: 100vh; /* Set to 100vh for full viewport height */
  width: 100%;
  padding: 2rem 1rem; /* Padding from top to avoid sticking */
  box-sizing: border-box; /* Ensures padding doesn’t add to width */
  overflow-y: auto; /* Allows scroll if content overflows */
`;

const SearchWrapper = styled.div`
  width: 100%;
  max-width: 400px;
  margin-bottom: 1rem;
  margin-top: 4rem; /* Adjust based on navbar height */
  position: relative;
  display: flex;
  gap: 0.5rem;
`;

const SearchInput = styled.input`
  width: 100%;
  padding: 0.8rem;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

const SearchButton = styled.button`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: #007bff;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: #0056b3;
  }
`;

const SelectionContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 1rem;
`;

const TeamButton = styled.button<{ selected: boolean }>`
  padding: 0.8rem;
  border: 2px solid ${({ selected }) => (selected ? '#007bff' : '#ccc')};
  background-color: ${({ selected }) => (selected ? '#e6f7ff' : '#fff')};
  color: #333;
  margin-bottom: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  width: 100%;
  max-width: 300px;
  &:hover {
    border-color: #007bff;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
`;

const ConfirmButton = styled.button`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: #28a745;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  flex: 1;
  &:hover {
    background-color: #218838;
  }
`;

const CancelButton = styled.button`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: #dc3545;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  flex: 1;
  &:hover {
    background-color: #c82333;
  }
`;

const LoadingMessage = styled.div`
  padding: 1rem;
  font-size: 1.2rem;
  color: #333;
`;

const TeamStatsBox = styled.div`
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  padding: 0.8rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
  text-align: center;
  width: 250px;
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

const ActionButtons = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
`;

const OptimizeButton = styled.button<{ isOptimized: boolean }>`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: ${({ isOptimized }) =>
    isOptimized ? '#ffc107' : '#007bff'};
  color: white;
  border-radius: 4px;
  cursor: pointer;
  flex: 1;
  &:hover {
    background-color: ${({ isOptimized }) =>
      isOptimized ? '#e0a800' : '#0056b3'};
  }
`;

const TransferButton = styled.button`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: #17a2b8;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  flex: 1;
  &:hover {
    background-color: #117a8b;
  }
`;

const YourTeam: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filteredTeams, setFilteredTeams] = useState<TeamName[]>([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [teamData, setTeamData] = useState<Team | null>(null);
  const [previousTeamData, setPreviousTeamData] = useState<Team | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isOptimized, setIsOptimized] = useState<boolean>(false);

  interface TeamName {
    teamName: string;
    id: string;
  }

  interface Player {
    name: string;
    team: string;
    position: 'GK' | 'DEF' | 'MID' | 'ATT';
    price: number;
    expected_points: number[];
    isBench: boolean[];
    isCaptain: boolean[];
  }

  interface Team {
    team: Player[];
  }

  // Search function with delay and loading indication
  const handleSearch = () => {
    setLoading(true);
    setTeamData(null); // Hide the current team display while loading
    setSearchQuery(''); // Clear the search bar
    setTimeout(() => {
      const matches = (testTeamNames as TeamName[]).filter((team) =>
        team.id.includes(searchQuery)
      );
      setFilteredTeams(matches);
      setLoading(false);
    }, 1000); // Simulated delay for loading
  };

  // Handle Enter key to activate search
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  // Handle selecting a team option
  const handleTeamSelect = (teamId: string) => {
    setSelectedTeamId(teamId);
  };

  // Confirm selection and display team
  const handleConfirmSelection = () => {
    if (selectedTeamId) {
      const team = (testTeamData as { teams: Record<string, Team> }).teams[
        selectedTeamId
      ];
      setPreviousTeamData(team); // Store the current team data for potential reversion
      setTeamData(team);
      setFilteredTeams([]); // Hide the selection container
    }
  };

  // Cancel selection and revert to the previous team
  const handleCancelSelection = () => {
    setFilteredTeams([]); // Hide the selection container
    setTeamData(previousTeamData); // Revert to previous team data if available
  };

  // Handle Optimize functionality
  const handleOptimize = () => {
    console.log('Selected Team ID:', selectedTeamId); // Debugging: check selectedTeamId

    if (isOptimized) {
      setTeamData(previousTeamData);
      setIsOptimized(false);
    } else if (selectedTeamId) {
      // Retrieve the optimized team for the selected ID
      const optimizedTeam = (
        testOptimizedTeam as { teams: Record<string, Team> }
      ).teams[selectedTeamId.trim()];

      if (optimizedTeam) {
        console.log('Optimizing team:', optimizedTeam);
        setPreviousTeamData(teamData);
        setTeamData(optimizedTeam);
        setIsOptimized(true);
      } else {
        console.error(`No optimized team found for ID: ${selectedTeamId}`);
      }
    }
  };
  // Calculate projected points for Weeks 1, 2, and 3
  const calculateProjectedPoints = (team: Team) => {
    return [0, 1, 2].map((weekIndex) =>
      team.team
        .filter((player) => !player.isBench[weekIndex])
        .reduce((sum, player) => sum + player.expected_points[weekIndex], 0)
        .toFixed(2)
    );
  };

  return (
    <PageWrapper>
      <SearchWrapper>
        <SearchInput
          type="text"
          placeholder="Enter team ID to search..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <SearchButton onClick={handleSearch}>Search</SearchButton>
      </SearchWrapper>

      {loading && <LoadingMessage>Loading...</LoadingMessage>}

      {filteredTeams.length > 0 && !loading && (
        <SelectionContainer>
          {filteredTeams.map((team) => (
            <TeamButton
              key={team.id}
              selected={team.id === selectedTeamId}
              onClick={() => handleTeamSelect(team.id)}
            >
              {team.teamName}
            </TeamButton>
          ))}
          <ButtonGroup>
            <ConfirmButton
              onClick={handleConfirmSelection}
              disabled={!selectedTeamId}
            >
              Confirm
            </ConfirmButton>
            <CancelButton onClick={handleCancelSelection}>Cancel</CancelButton>
          </ButtonGroup>
        </SelectionContainer>
      )}

      {teamData && !loading && (
        <>
          <TeamStatsBox>
            <h3>Projected Points</h3>
            <StatsRow>
              {calculateProjectedPoints(teamData).map((points, index) => (
                <Stat key={index}>
                  Week {index + 1}: {points}
                </Stat>
              ))}
            </StatsRow>
          </TeamStatsBox>

          <ActionButtons>
            <OptimizeButton onClick={handleOptimize} isOptimized={isOptimized}>
              {isOptimized ? 'Revert' : 'Optimize'}
            </OptimizeButton>
            <TransferButton>Transfer</TransferButton>
          </ActionButtons>

          <div>
            <h2>Starting XI</h2>
            {['GK', 'DEF', 'MID', 'ATT'].map((position) => (
              <PlayerRow
                key={position}
                players={teamData.team.filter(
                  (player) => player.position === position && !player.isBench[0]
                )}
              />
            ))}

            <h2>Bench</h2>
            <Bench
              players={teamData.team.filter((player) => player.isBench[0])}
            />
          </div>
        </>
      )}
    </PageWrapper>
  );
};

export default YourTeam;
