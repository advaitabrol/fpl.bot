import React from 'react';
import styled from 'styled-components';

const FilterContainer = styled.div`
  display: flex;
  gap: 10px;
  margin: 10px 0;
  align-items: center;
`;

interface FilterControlsProps {
  searchName: string;
  onSearchNameChange: (name: string) => void;
  selectedPositions: string[];
  onPositionChange: (position: string) => void; // Changed to single string input for simplicity
  resetFilters: () => void;
}

const FilterControls: React.FC<FilterControlsProps> = ({
  searchName,
  onSearchNameChange,
  selectedPositions,
  onPositionChange,
  resetFilters,
}) => {
  return (
    <FilterContainer>
      <input
        name="name"
        placeholder="Search by Name"
        value={searchName}
        onChange={(e) => onSearchNameChange(e.target.value)}
      />
      <label>
        <input
          type="checkbox"
          checked={selectedPositions.includes('GK')}
          onChange={() => onPositionChange('GK')}
        />
        GK
      </label>
      <label>
        <input
          type="checkbox"
          checked={selectedPositions.includes('DEF')}
          onChange={() => onPositionChange('DEF')}
        />
        DEF
      </label>
      <label>
        <input
          type="checkbox"
          checked={selectedPositions.includes('MID')}
          onChange={() => onPositionChange('MID')}
        />
        MID
      </label>
      <label>
        <input
          type="checkbox"
          checked={selectedPositions.includes('FWD')}
          onChange={() => onPositionChange('FWD')}
        />
        FWD
      </label>
      <button onClick={resetFilters}>Reset Filters</button>
    </FilterContainer>
  );
};

export default FilterControls;
