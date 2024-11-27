import React from 'react';
import styled from 'styled-components';

const RecapContainer = styled.div`
  margin-top: 1rem;
  padding: 1rem;
  border-top: 1px solid #ddd;
  text-align: center;
`;

const RecapItem = styled.div`
  margin-bottom: 0.5rem;
  font-size: 1rem;
`;

const RecapActions = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1rem;
`;

const ActionButton = styled.button<{ primary: boolean }>`
  padding: 0.8rem 1.2rem;
  background-color: ${({ primary }) => (primary ? '#007bff' : '#dc3545')};
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;

  &:hover {
    background-color: ${({ primary }) => (primary ? '#0056b3' : '#c82333')};
  }
`;

interface NetTransferRecapProps {
  totalNetPoints: number;
  totalNetCost: number;
  onApprove: () => void;
  onReject: () => void;
}

const NetTransferRecap: React.FC<NetTransferRecapProps> = ({
  totalNetPoints,
  totalNetCost,
  onApprove,
  onReject,
}) => {
  return (
    <RecapContainer>
      <RecapItem>
        <strong>Net Points Gained:</strong> {totalNetPoints.toFixed(2)}
      </RecapItem>
      <RecapItem>
        <strong>Net Cost:</strong> {totalNetCost > 0 ? '+' : ''}
        {totalNetCost.toFixed(2)}
      </RecapItem>

      <RecapActions>
        <ActionButton primary onClick={onApprove}>
          Approve
        </ActionButton>
        <ActionButton primary={false} onClick={onReject}>
          Reject
        </ActionButton>
      </RecapActions>
    </RecapContainer>
  );
};

export default NetTransferRecap;
