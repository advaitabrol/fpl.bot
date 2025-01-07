import React from 'react';
import styled from 'styled-components';
import PlayerIcon from '../../ui/PlayerIcon';
import { useTeamColors } from '../../hooks/useTeamColors';

import { Transfer } from '../../services/interfaces';

const TransferRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  margin-bottom: 1rem;
  background: #f9f9f9;
`;

const PlayerInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  text-align: left;
  flex: 1; /* Ensure both players take equal space */
`;

const PointsPriceWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
`;

const ArrowWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 5rem; /* Fixed width for consistent spacing */
  flex-shrink: 0; /* Prevent shrinking when resizing */
`;

const ArrowIcon = styled.div`
  font-size: 2rem;
  color: #007bff;
`;

const PointsChange = styled.div<{ netPoints: number }>`
  font-weight: bold;
  color: ${({ netPoints }) => (netPoints > 0 ? 'green' : 'red')};
  text-align: right;
  flex: 0.5; /* Consistent width for net points */
`;

const PlayerName = styled.p`
  font-weight: bold;
  margin: 0;
  text-transform: capitalize;
`;

const PlayerPoints = styled.p`
  margin: 0;
  font-size: 0.9rem;
  color: #555;
`;

const SuggestedTransfer: React.FC<Transfer> = ({
  out,
  in_player,
  net_points,
}) => {
  const { primary: outPrimary, secondary: outSecondary } = useTeamColors(
    out.team
  );
  const { primary: inPrimary, secondary: inSecondary } = useTeamColors(
    in_player.team
  );

  // Helper to format the player's display name (first and last name only)
  const formatName = (name: string) => {
    const nameParts = name.split(' ');
    return `${nameParts[0]} ${nameParts[nameParts.length - 1]}`;
  };

  return (
    <TransferRow>
      {/* Out Player Info */}
      <PlayerInfo>
        <PlayerIcon
          primaryColor={outPrimary}
          secondaryColor={outSecondary}
          size="icon"
        />
        <PointsPriceWrapper>
          <PlayerName>
            {formatName(out.name)} (£{out.price.toFixed(2)})
          </PlayerName>
          <PlayerPoints>
            Points:{' '}
            {out.expected_points.map((p: number) => p.toFixed(2)).join(', ')}
          </PlayerPoints>
        </PointsPriceWrapper>
      </PlayerInfo>

      {/* Arrow */}
      <ArrowWrapper>
        <ArrowIcon>➔</ArrowIcon>
      </ArrowWrapper>

      {/* In Player Info */}
      <PlayerInfo>
        <PlayerIcon
          primaryColor={inPrimary}
          secondaryColor={inSecondary}
          size="icon"
        />
        <PointsPriceWrapper>
          <PlayerName>
            {formatName(in_player.name)} (£{in_player.price.toFixed(2)})
          </PlayerName>
          <PlayerPoints>
            Points:{' '}
            {in_player.expected_points
              .map((p: number) => p.toFixed(2))
              .join(', ')}
          </PlayerPoints>
        </PointsPriceWrapper>
      </PlayerInfo>

      {/* Net Points Change */}
      <PointsChange netPoints={net_points}>
        {net_points > 0
          ? `+${net_points.toFixed(2)}`
          : `-${net_points.toFixed(2)}`}{' '}
        points
      </PointsChange>
    </TransferRow>
  );
};

export default SuggestedTransfer;
