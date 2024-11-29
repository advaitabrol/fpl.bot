import React, { useState } from 'react';
import styled from 'styled-components';
import ConfigureTransfers from './ConfigureTransfers';
import useGetTransfers from '../../services/useGetTransfers';
import useOptimizeTeam from '../../services/useOptimizeTeam';
import {
  Player,
  TransferSuggestion,
  TransferConfiguration,
  Transfer,
} from '../../services/interfaces';

// Styled-components
const ButtonContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
  justify-content: center;
`;

const ActionButton = styled.button<{ primary: boolean }>`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: ${({ primary }) => (primary ? '#007bff' : '#ffc107')};
  color: white;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: ${({ primary }) => (primary ? '#0056b3' : '#e0a800')};
  }
`;

const TeamActions: React.FC<{
  team: Player[];
  setTeam: (team: Player[]) => void;
  bank: number;
  setBank: (value: number) => void;
}> = ({ team, setTeam, bank, setBank }) => {
  const [showModal, setShowModal] = useState(false);
  const [view, setView] = useState<'configure' | 'suggestions'>('configure');
  const [suggestedTransfers, setSuggestedTransfers] = useState<Transfer[]>([]);
  const [totalNetPoints, setTotalNetPoints] = useState(0);
  const [totalNetCost, setTotalNetCost] = useState(0);
  const [updatedTeam, setUpdatedTeam] = useState<Player[]>(team);
  const [updatedBank, setUpdatedBank] = useState(0);

  const { getTransfers } = useGetTransfers();
  const { optimizeTeam } = useOptimizeTeam();

  const handleOptimize = async () => {
    try {
      const optimizedTeam = await optimizeTeam(team);
      setTeam(optimizedTeam.team); // Update the team state with the optimized team
    } catch (error) {
      console.error('Failed to optimize team:', error);
    }
  };

  const handleSuggestTransfersSubmit = async (
    transferConfig: TransferConfiguration
  ) => {
    try {
      const response = await getTransfers({
        team: transferConfig.team,
        max_transfers: transferConfig.max_transfers,
        keep_players: transferConfig.keep_players,
        avoid_players: transferConfig.avoid_players,
        keep_teams: transferConfig.keep_teams,
        avoid_teams: transferConfig.avoid_teams,
        desired_selected: transferConfig.desired_selected,
        captain_scale: transferConfig.captain_scale,
        bank: bank,
      });

      setUpdatedTeam(response.optimized_team);
      setUpdatedBank(response.bank);

      const transfers = response.transfers_suggestion.map(
        (transfer: TransferSuggestion) => ({
          out: transfer.out,
          in_player: transfer.in_player,
          net_points:
            transfer.in_player.expected_points.reduce(
              (a: number, b: number) => a + b,
              0.0
            ) -
            transfer.out.expected_points.reduce(
              (a: number, b: number) => a + b,
              0.0
            ),
        })
      );

      console.log(transfers);

      const totalPoints = transfers.reduce(
        (sum: number, t: Transfer) => sum + t.net_points,
        0
      );
      const totalCost = transfers.reduce(
        (sum: number, t: Transfer) => sum + t.in_player.price - t.out.price,
        0
      );

      setSuggestedTransfers(transfers);
      setTotalNetPoints(totalPoints);
      setTotalNetCost(totalCost);
      setView('suggestions');
    } catch (error) {
      console.error('Error suggesting transfers:', error);
    }
  };

  const handleApprove = () => {
    setTeam(updatedTeam);
    setBank(updatedBank);
    setShowModal(false);
    setView('configure');
  };

  const handleReject = () => {
    setView('configure');
  };

  return (
    <>
      <ButtonContainer>
        <ActionButton primary onClick={handleOptimize}>
          Optimize
        </ActionButton>
        <ActionButton primary={false} onClick={() => setShowModal(true)}>
          Suggest Transfers
        </ActionButton>
      </ButtonContainer>

      {showModal && (
        <ConfigureTransfers
          team={team}
          onClose={() => {
            setShowModal(false);
            setView('configure');
          }}
          onSubmit={handleSuggestTransfersSubmit}
          view={view}
          suggestedTransfers={suggestedTransfers}
          totalNetPoints={totalNetPoints}
          totalNetCost={totalNetCost}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      )}
    </>
  );
};

export default TeamActions;
