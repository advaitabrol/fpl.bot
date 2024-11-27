import React from 'react';
import styled from 'styled-components';

const ManualSearchWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-top: 1rem;
  width: 100%;
  max-width: 400px;
`;

const ManualSearchLabel = styled.label`
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 0.5rem;
`;

const ManualIdInput = styled.input`
  width: 100%;
  padding: 0.8rem;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

interface ManualSearchProps {
  manualTeamId: string;
  setManualTeamId: (id: string) => void;
}

const ManualSearch: React.FC<ManualSearchProps> = ({
  manualTeamId,
  setManualTeamId,
}) => (
  <ManualSearchWrapper>
    <ManualSearchLabel>
      Don’t see your team? Search manually here! Enter team ID:
    </ManualSearchLabel>
    <ManualIdInput
      type="text"
      placeholder="Enter team ID..."
      value={manualTeamId}
      onChange={(e) => setManualTeamId(e.target.value)}
    />
  </ManualSearchWrapper>
);

export default ManualSearch;
