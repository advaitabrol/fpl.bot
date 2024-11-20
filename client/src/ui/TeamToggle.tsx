import React from 'react';
import styled from 'styled-components';

interface TeamToggleProps<T> {
  option1: T; // Generic type for the first option
  option2: T; // Generic type for the second option
  selectedOption: T; // The currently selected option
  onToggle: (option: T) => void; // Callback function to handle toggle
}

const ToggleContainer = styled.div`
  display: flex;
  justify-content: center;
  margin: 1rem 0;
`;

const ToggleButton = styled.button<{ isActive: boolean }>`
  padding: 0.5rem 1.5rem;
  margin: 0 0.5rem;
  background-color: ${({ isActive }) =>
    isActive ? '#37003c' : 'rgba(55, 0, 60, 0.4)'};
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.3s ease;

  &:hover {
    background-color: #37003c;
  }
`;

function TeamToggle<T>({
  option1,
  option2,
  selectedOption,
  onToggle,
}: TeamToggleProps<T>) {
  return (
    <ToggleContainer>
      <ToggleButton
        isActive={selectedOption === option1}
        onClick={() => onToggle(option1)}
      >
        {String(option1)}
      </ToggleButton>
      <ToggleButton
        isActive={selectedOption === option2}
        onClick={() => onToggle(option2)}
      >
        {String(option2)}
      </ToggleButton>
    </ToggleContainer>
  );
}

export default TeamToggle;
