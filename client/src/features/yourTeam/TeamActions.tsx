import React, { useState } from 'react';
import styled from 'styled-components';
import ConfigureTransfers from './ConfigureTransfers';
import useGetTransfers from '../../services/useGetTransfers';
import useOptimizeTeam, { Player } from '../../services/useOptimizeTeam';

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
}> = ({ team, setTeam }) => {
  const [showModal, setShowModal] = useState(false);
  const [view, setView] = useState<'configure' | 'suggestions'>('configure');
  const [suggestedTransfers, setSuggestedTransfers] = useState<any[]>([]);
  const [totalNetPoints, setTotalNetPoints] = useState(0);
  const [totalNetCost, setTotalNetCost] = useState(0);
  const [updatedTeam, setUpdatedTeam] = useState<Player[]>(team);

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

  const handleSuggestTransfersSubmit = async (transferConfig: any) => {
    try {
      const response = await getTransfers({
        team,
        maxTransfers: transferConfig.maxTransfers,
        keep: transferConfig.keepPlayers,
        blacklist: transferConfig.avoidPlayers,
      });

      setUpdatedTeam(response.optimized_team);

      const transfers = response.transfers_suggestion.map((transfer: any) => ({
        out: transfer.out,
        in: transfer.in,
        netPoints:
          transfer.in.expected_points.reduce(
            (a: number, b: number) => a + b,
            0
          ) -
          transfer.out.expected_points.reduce(
            (a: number, b: number) => a + b,
            0
          ),
      }));

      const totalPoints = transfers.reduce(
        (sum: number, t: any) => sum + t.netPoints,
        0
      );
      const totalCost = transfers.reduce(
        (sum: number, t: any) => sum + t.in.price - t.out.price,
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
