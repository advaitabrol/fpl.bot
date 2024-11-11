import React, { useEffect, useState } from 'react';
import Papa from 'papaparse';
import styled from 'styled-components';
import FilterControls from '../features/playerPage/FilterControls';
import PlayerTable from '../features/playerPage/PlayerTable';
import PaginationControls from '../features/playerPage/PaginationControls';

const TableWrapper = styled.div`
  max-width: 100%;
  overflow-x: auto;
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
  }>({ key: 'totalPoints', direction: 'desc' });
  const [selectedPositions, setSelectedPositions] = useState<string[]>([]);
  const playersPerPage = 20;

  // Load and parse CSV data
  useEffect(() => {
    Papa.parse('/all.csv', {
      download: true,
      header: true,
      complete: (result) => {
        const parsedData = result.data as Player[];
        const playersWithTotal = parsedData
          .filter(
            (player) =>
              typeof player.name === 'string' && player.name.trim() !== ''
          )
          .map((player) => ({
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
        setPlayers(playersWithTotal);
      },
    });
  }, []);

  // Combined effect for filtering and sorting with forced synchronous updates
  useEffect(() => {
    console.log('useEffect triggered with dependencies:', {
      players,
      searchName,
      selectedPositions,
      sortConfig,
    });

    const applyFilterAndSort = () => {
      // Step 1: Apply filtering
      let updatedPlayers = players;
      if (searchName.trim()) {
        updatedPlayers = updatedPlayers.filter((player) =>
          player.name.toLowerCase().includes(searchName.toLowerCase())
        );
      }
      if (selectedPositions.length > 0) {
        updatedPlayers = updatedPlayers.filter((player) =>
          selectedPositions.includes(player.position)
        );
      }

      // Step 2: Apply sorting
      if (sortConfig) {
        updatedPlayers = [...updatedPlayers].sort((a, b) => {
          const aValue =
            typeof a[sortConfig.key] === 'number'
              ? a[sortConfig.key]
              : parseFloat(String(a[sortConfig.key]));
          const bValue =
            typeof b[sortConfig.key] === 'number'
              ? b[sortConfig.key]
              : parseFloat(String(b[sortConfig.key]));

          if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
          if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
          return 0;
        });
      }

      setFilteredPlayers(updatedPlayers);
    };

    applyFilterAndSort();
  }, [players, searchName, selectedPositions, sortConfig]);

  // Update current page players every time filteredPlayers or page changes
  const currentPlayers = filteredPlayers.slice(
    (currentPage - 1) * playersPerPage,
    currentPage * playersPerPage
  );
  const totalPages = Math.ceil(filteredPlayers.length / playersPerPage);

  const handlePageChange = (newPage: number) => setCurrentPage(newPage);

  const handleSortChange = (key: keyof Player) => {
    setSortConfig((prevSortConfig) => {
      const direction =
        prevSortConfig?.key === key && prevSortConfig.direction === 'asc'
          ? 'desc'
          : 'asc';
      return { key, direction };
    });
  };

  const handlePositionChange = (position: string) => {
    setSelectedPositions((prev) =>
      prev.includes(position)
        ? prev.filter((p) => p !== position)
        : [...prev, position]
    );
  };

  const handleResetFilters = () => {
    setSearchName('');
    setSelectedPositions([]);
    setSortConfig({ key: 'totalPoints', direction: 'desc' });
  };

  return (
    <div>
      <h2>Player List</h2>
      <FilterControls
        searchName={searchName}
        onSearchNameChange={setSearchName}
        selectedPositions={selectedPositions}
        onPositionChange={handlePositionChange}
        resetFilters={handleResetFilters}
      />
      <TableWrapper>
        <PlayerTable
          players={currentPlayers}
          sortConfig={sortConfig}
          onSortChange={handleSortChange}
        />
      </TableWrapper>
      <PaginationControls
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={handlePageChange}
      />
    </div>
  );
};

export default PlayersPage;
