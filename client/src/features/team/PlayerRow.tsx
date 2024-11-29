import React from 'react';
import styled from 'styled-components';
import Player from './Player';
import { Player as PlayerInterface } from '../../services/interfaces';

interface PlayerRowProps {
  players: PlayerInterface[];
  weekIndex?: number; // Optional weekIndex prop to show specific week's points
}

const RowWrapper = styled.div`
  margin-bottom: 0rem;
  text-align: center;
`;

const PlayersContainer = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
`;

const PlayerRow: React.FC<PlayerRowProps> = ({ players, weekIndex }) => {
  return (
    <RowWrapper>
      <PlayersContainer>
        {players.map((player, index) => (
          <Player key={index} player={player} weekIndex={weekIndex} />
        ))}
      </PlayersContainer>
    </RowWrapper>
  );
};

export default PlayerRow;
