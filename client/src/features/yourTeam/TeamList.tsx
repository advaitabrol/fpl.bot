import React from 'react';
import styled from 'styled-components';
import ManualSearch from './ManualSearch';

const SelectionContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 1rem;
`;

const TeamButton = styled.button<{ selected: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  padding: 0.8rem 1rem;
  border: 2px solid ${({ selected }) => (selected ? '#007bff' : '#ccc')};
  background-color: ${({ selected }) => (selected ? '#e6f7ff' : '#fff')};
  color: #333;
  margin-bottom: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  width: 100%;
  max-width: 400px;
  text-align: left;
  outline: none;
  &:hover {
    border-color: #007bff;
  }

  .team-name {
    font-weight: bold;
  }

  .manager-name {
    font-size: 0.9rem;
    color: #888;
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

interface TeamListProps {
  teams: { team_name: string; manager_name: string; team_id: string }[];
  selectedTeamId: string | null;
  onSelectTeam: (id: string) => void;
  manualTeamId: string;
  setManualTeamId: (id: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
}

const TeamList: React.FC<TeamListProps> = ({
  teams,
  selectedTeamId,
  onSelectTeam,
  manualTeamId,
  setManualTeamId,
  onConfirm,
  onCancel,
}) => {
  return (
    <>
      <SelectionContainer>
        {teams.map((team) => (
          <TeamButton
            key={team.team_id}
            selected={team.team_id === selectedTeamId}
            onClick={() => onSelectTeam(team.team_id)}
          >
            <span className="team-name">{team.team_name}</span>
            <span className="manager-name">{team.manager_name}</span>
          </TeamButton>
        ))}
      </SelectionContainer>

      <ManualSearch
        manualTeamId={manualTeamId}
        setManualTeamId={setManualTeamId}
      />

      <ButtonGroup>
        <ConfirmButton
          onClick={onConfirm}
          disabled={!selectedTeamId && !manualTeamId}
        >
          Confirm
        </ConfirmButton>
        <CancelButton onClick={onCancel}>Cancel</CancelButton>
      </ButtonGroup>
    </>
  );
};

export default TeamList;
