import React from 'react';
import styled from 'styled-components';
import Player from './Player';

interface BenchProps {
  players: Array<{
    name: string;
    team: string;
    price: number;
    expected_points: number[];
    isCaptain: boolean[];
  }>;
  weekIndex?: number; // Optional weekIndex prop to show specific week's points
}

const BenchWrapper = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1rem;
`;

const Bench: React.FC<BenchProps> = ({ players, weekIndex }) => {
  return (
    <BenchWrapper>
      {players.map((player, index) => (
        <Player key={index} player={player} weekIndex={weekIndex} />
      ))}
    </BenchWrapper>
  );
};

export default Bench;
