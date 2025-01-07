import React from 'react';
import styled from 'styled-components';
import Player from './Player';
import { Player as PlayerType } from '../../services/interfaces';

interface BenchProps {
  players: PlayerType[];
  weekIndex?: number; // Optional weekIndex prop to show specific week's points
}

const BenchWrapper = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1rem;
`;

const Bench: React.FC<BenchProps> = ({ players, weekIndex }) => {
  // Sort players: GK always first, others ordered by highest expected points
  const sortedPlayers = [
    ...players.filter((player) => player.position === 'GK'), // GK always first
    ...players
      .filter((player) => player.position !== 'GK') // Other players
      .sort((a, b) => {
        const pointsA =
          weekIndex !== undefined
            ? a.expected_points[weekIndex]
            : Math.max(...a.expected_points);
        const pointsB =
          weekIndex !== undefined
            ? b.expected_points[weekIndex]
            : Math.max(...b.expected_points);
        return pointsB - pointsA; // Descending order by expected points
      }),
  ];

  return (
    <BenchWrapper>
      {sortedPlayers.map((player, index) => (
        <Player key={index} player={player} weekIndex={weekIndex} />
      ))}
    </BenchWrapper>
  );
};

export default Bench;
