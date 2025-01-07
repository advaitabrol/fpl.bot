import React from 'react';
import styled from 'styled-components';

const SearchWrapper = styled.div`
  width: 100%;
  max-width: 400px;
  margin-bottom: 1rem;
  margin-top: 4rem;
  position: relative;
  display: flex;
  gap: 0.5rem;
`;

const SearchInput = styled.input`
  width: 100%;
  padding: 0.8rem;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

const SearchButton = styled.button`
  padding: 0.8rem 1.2rem;
  border: none;
  background-color: #007bff;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  &:hover {
    background-color: #0056b3;
  }
`;

interface SearchBarProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  onSearch: () => void;
}

const TeamSearchBar: React.FC<SearchBarProps> = ({
  searchQuery,
  setSearchQuery,
  onSearch,
}) => {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      onSearch();
    }
  };

  return (
    <SearchWrapper>
      <SearchInput
        type="text"
        placeholder="Enter team name..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <SearchButton onClick={onSearch}>Search</SearchButton>
    </SearchWrapper>
  );
};

export default TeamSearchBar;
