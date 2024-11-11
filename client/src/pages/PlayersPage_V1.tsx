import React, { useEffect, useState } from 'react';
import Papa from 'papaparse';
import styled from 'styled-components';

const TableWrapper = styled.div`
  max-width: 100%;
  overflow-x: auto;
`;

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
    display: inline-block;
    margin-left: 5px;
  }
`;

const Td = styled.td`
  padding: 10px;
  text-align: center;
  border-bottom: 1px solid #ddd;
`;

const PaginationControls = styled.div`
  display: flex;
  justify-content: center;
  margin: 20px 0;
`;

const FilterContainer = styled.div`
  display: flex;
  gap: 10px;
  margin: 10px 0;
  align-items: center;
`;

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

const PlayersPage: React.FC = () => {
  const [players, setPlayers] = useState<Player[]>([]);
  const [filteredPlayers, setFilteredPlayers] = useState<Player[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchName, setSearchName] = useState('');
  const [sortConfig, setSortConfig] = useState<{
    key: keyof Player;
    direction: 'asc' | 'desc';
  } | null>({
    key: 'totalPoints',
    direction: 'desc',
  });
  const [selectedPositions, setSelectedPositions] = useState<string[]>([]);

  const playersPerPage = 20;

  useEffect(() => {
    Papa.parse('/all.csv', {
      download: true,
      header: true,
      complete: (result) => {
        const parsedData = result.data as Player[];
        const playersWithTotal = parsedData.map((player) => ({
          ...player,
          price: Number(player['price']) || 0,
          week1: Number(player['week1']) || 0,
          week2: Number(player['week2']) || 0,
          week3: Number(player['week3']) || 0,
          totalPoints: parseFloat(
            (
              Number(player['week1']) +
              Number(player['week2']) +
              Number(player['week3'])
            ).toFixed(2)
          ),
        }));

        // Sort by totalPoints by default in descending order
        const sortedPlayers = [...playersWithTotal].sort(
          (a, b) => b.totalPoints - a.totalPoints
        );
        setPlayers(sortedPlayers);
        setFilteredPlayers(sortedPlayers);
      },
    });
  }, []);

  useEffect(() => {
    const filtered = players
      .filter((player) =>
        player.name.toLowerCase().includes(searchName.toLowerCase())
      )
      .filter(
        (player) =>
          selectedPositions.length === 0 ||
          selectedPositions.includes(player.position)
      );

    setFilteredPlayers(filtered);
  }, [searchName, players, selectedPositions]);

  const handleSort = (key: keyof Player) => {
    const direction =
      sortConfig?.key === key && sortConfig.direction === 'asc'
        ? 'desc'
        : 'asc';
    setSortConfig({ key, direction });

    const sortedPlayers = [...filteredPlayers].sort((a, b) => {
      const aValue =
        typeof a[key] === 'number'
          ? (a[key] as number)
          : parseFloat(String(a[key]));
      const bValue =
        typeof b[key] === 'number'
          ? (b[key] as number)
          : parseFloat(String(b[key]));

      if (aValue < bValue) return direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return direction === 'asc' ? 1 : -1;
      return 0;
    });

    setFilteredPlayers(sortedPlayers);
  };

  const getSortIcon = (key: keyof Player) => {
    if (sortConfig && sortConfig.key === key) {
      return sortConfig.direction === 'asc' ? '▲' : '▼';
    }
    return '';
  };

  const handlePositionChange = (position: string) => {
    setSelectedPositions((prev) =>
      prev.includes(position)
        ? prev.filter((p) => p !== position)
        : [...prev, position]
    );
  };

  const resetFilters = () => {
    setSelectedPositions([]);
    setSearchName('');
    setSortConfig(null); // Reset sortConfig to null to remove all sorting icons
  };

  const indexOfLastPlayer = currentPage * playersPerPage;
  const indexOfFirstPlayer = indexOfLastPlayer - playersPerPage;
  const currentPlayers = filteredPlayers.slice(
    indexOfFirstPlayer,
    indexOfLastPlayer
  );
  const totalPages = Math.ceil(filteredPlayers.length / playersPerPage);

  const handlePageChange = (newPage: number) => {
    if (newPage > 0 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  return (
    <div>
      <h2>Player List</h2>

      {/* Filter Section */}
      <FilterContainer>
        <input
          name="name"
          placeholder="Search by Name"
          value={searchName}
          onChange={(e) => setSearchName(e.target.value)}
        />
        <label>
          <input
            type="checkbox"
            checked={selectedPositions.includes('GK')}
            onChange={() => handlePositionChange('GK')}
          />
          GK
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedPositions.includes('DEF')}
            onChange={() => handlePositionChange('DEF')}
          />
          DEF
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedPositions.includes('MID')}
            onChange={() => handlePositionChange('MID')}
          />
          MID
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedPositions.includes('FWD')}
            onChange={() => handlePositionChange('FWD')}
          />
          FWD
        </label>
        <button onClick={resetFilters}>Reset Filters</button>
      </FilterContainer>

      <TableWrapper>
        <Table>
          <thead>
            <tr>
              <Th>Name</Th>
              <Th>Team</Th>
              <Th onClick={() => handleSort('price')}>
                Price <span>{getSortIcon('price')}</span>
              </Th>
              <Th>Position</Th>
              <Th onClick={() => handleSort('week1')}>
                Week 1 <span>{getSortIcon('week1')}</span>
              </Th>
              <Th onClick={() => handleSort('week2')}>
                Week 2 <span>{getSortIcon('week2')}</span>
              </Th>
              <Th onClick={() => handleSort('week3')}>
                Week 3 <span>{getSortIcon('week3')}</span>
              </Th>
              <Th onClick={() => handleSort('totalPoints')}>
                Total Points <span>{getSortIcon('totalPoints')}</span>
              </Th>
            </tr>
          </thead>
          <tbody>
            {currentPlayers.map((player, index) => (
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
      </TableWrapper>

      {/* Pagination Controls */}
      <PaginationControls>
        <button
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </PaginationControls>
    </div>
  );
};

export default PlayersPage;
