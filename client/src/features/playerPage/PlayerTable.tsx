import React from 'react';
import styled from 'styled-components';
import PlayerRow from './PlayerRow';

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

interface SortConfig {
  key: keyof Player;
  direction: 'asc' | 'desc';
}

interface PlayerTableProps {
  players: Player[];
  sortConfig: SortConfig | null;
  onSortChange: (key: keyof Player) => void;
}

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  background-color: white;
  border: 1px solid #eaeaea;
`;

const Th = styled.th`
  padding: 10px;
  background-color: #37003c; /* Premier League purple */
  color: white;
  border-bottom: 2px solid #eaeaea;
  cursor: pointer;
  text-align: center;
  position: sticky;
  top: 0;
  z-index: 1;

  span {
    margin-left: 5px;
    font-size: 0.8em;
  }
`;

const Td = styled.td`
  padding: 10px;
  text-align: center;
  border-bottom: 1px solid #ddd;
  &:nth-child(odd) {
    background-color: #f9f9f9;
  }
`;

const PlayerTable: React.FC<PlayerTableProps> = ({
  players,
  sortConfig,
  onSortChange,
}) => {
  const getSortIcon = (key: keyof Player) => {
    if (sortConfig && sortConfig.key === key) {
      return sortConfig.direction === 'asc' ? '▲' : '▼';
    }
    return ''; // No icon if column is not sorted
  };

  return (
    <Table>
      <thead>
        <tr>
          <Th>Name</Th>
          <Th>Team</Th>
          <Th onClick={() => onSortChange('price')}>
            Price <span>{getSortIcon('price')}</span>
          </Th>
          <Th>Position</Th>
          <Th onClick={() => onSortChange('week1')}>
            Week 1 <span>{getSortIcon('week1')}</span>
          </Th>
          <Th onClick={() => onSortChange('week2')}>
            Week 2 <span>{getSortIcon('week2')}</span>
          </Th>
          <Th onClick={() => onSortChange('week3')}>
            Week 3 <span>{getSortIcon('week3')}</span>
          </Th>
          <Th onClick={() => onSortChange('totalPoints')}>
            Total Points <span>{getSortIcon('totalPoints')}</span>
          </Th>
        </tr>
      </thead>
      <tbody>
        {players.map((player, index) => (
          <PlayerRow key={index} player={player} />
        ))}
      </tbody>
    </Table>
  );
};

export default PlayerTable;
