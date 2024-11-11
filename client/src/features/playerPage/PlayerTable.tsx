import React from 'react';
import styled from 'styled-components';

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
`;

const Th = styled.th`
  padding: 10px;
  background-color: #f4f4f4;
  border-bottom: 1px solid #ddd;
  cursor: pointer;
  text-align: center;

  span {
    margin-left: 5px;
    font-size: 0.8em;
  }
`;

const Td = styled.td`
  padding: 10px;
  text-align: center;
  border-bottom: 1px solid #ddd;
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
          <tr key={index}>
            <Td>{player.name}</Td>
            <Td>{player.team}</Td>
            <Td>{player.price}</Td>
            <Td>{player.position}</Td>
            <Td>{player.week1}</Td>
            <Td>{player.week2}</Td>
            <Td>{player.week3}</Td>
            <Td>{player.totalPoints}</Td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
};

export default PlayerTable;
