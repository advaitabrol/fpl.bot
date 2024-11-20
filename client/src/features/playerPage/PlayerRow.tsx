import React from 'react';
import styled from 'styled-components';
import PlayerIcon from '../../ui/PlayerIcon';
import { useTeamColors } from '../../hooks/useTeamColors';

interface Player {
  name: string;
  team: string;
  price: number;
  position: string;
  week1: number;
  week2: number;
  week3: number;
  totalPoints: number;
}

interface PlayerRowProps {
  player: Player;
}

const Td = styled.td`
  padding: 10px;
  text-align: center;
  border-bottom: 1px solid #ddd;
`;

const PlayerNameCell = styled(Td)`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px; /* Space between icon and text */
`;

const PlayerRow: React.FC<PlayerRowProps> = ({ player }) => {
  const { primary, secondary } = useTeamColors(player.team);

  // Extract first and last name only
  const nameParts = player.name.split(' ');
  const displayName = `${nameParts[0]} ${nameParts[nameParts.length - 1]}`;

  return (
    <tr>
      <PlayerNameCell>
        <PlayerIcon
          primaryColor={primary}
          secondaryColor={secondary}
          size="icon"
        />
        {displayName}
      </PlayerNameCell>
      <Td>{player.team}</Td>
      <Td>{player.price}</Td>
      <Td>{player.position}</Td>
      <Td>{player.week1}</Td>
      <Td>{player.week2}</Td>
      <Td>{player.week3}</Td>
      <Td>{player.totalPoints}</Td>
    </tr>
  );
};

export default PlayerRow;
