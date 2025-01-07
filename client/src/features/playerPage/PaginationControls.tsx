import React from 'react';
import styled from 'styled-components';

const PaginationContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center; /* Align items vertically */
  gap: 20px; /* Add space between elements */
  margin: 20px 0;
`;

const PaginationButton = styled.button`
  padding: 10px 20px; /* Add padding for better button sizing */
  border: none;
  background-color: #007bff; /* Use a primary color */
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;

  &:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }

  &:hover:not(:disabled) {
    background-color: #0056b3;
  }
`;

const PaginationText = styled.span`
  font-size: 1.1rem; /* Slightly larger text */
  font-weight: 500;
`;

interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (newPage: number) => void;
}

const PaginationControls: React.FC<PaginationControlsProps> = ({
  currentPage,
  totalPages,
  onPageChange,
}) => (
  <PaginationContainer>
    <PaginationButton
      onClick={() => onPageChange(currentPage - 1)}
      disabled={currentPage === 1}
    >
      Previous
    </PaginationButton>
    <PaginationText>
      Page {currentPage} of {totalPages}
    </PaginationText>
    <PaginationButton
      onClick={() => onPageChange(currentPage + 1)}
      disabled={currentPage === totalPages}
    >
      Next
    </PaginationButton>
  </PaginationContainer>
);

export default PaginationControls;
