import React, { useState } from 'react';
import styled from 'styled-components';
import TeamToggle from '../ui/TeamToggle';
import FreeHitTeam from '../features/optimalTeam/FreeHitTeam';
import WildCardTeam from '../features/optimalTeam/WildCardTeam';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start; /* Start near the top */
  min-height: 100vh;
  min-width: 100%;
  padding-top: 2rem;
`;

const ToggleWrapper = styled.div`
  display: flex;
  justify-content: center;
  width: 100%; /* Full width to center the toggle within */
  max-width: 600px; /* Optional: limit width for a cleaner look */
`;

const OptimalTeam: React.FC = () => {
  const [selectedTeam, setSelectedTeam] = useState<'Free Hit' | 'Wildcard'>(
    'Free Hit'
  );

  const handleToggle = (option: 'Free Hit' | 'Wildcard') => {
    setSelectedTeam(option);
  };

  return (
    <Container>
      <ToggleWrapper>
        <TeamToggle
          option1="Free Hit"
          option2="Wildcard"
          selectedOption={selectedTeam}
          onToggle={handleToggle}
        />
      </ToggleWrapper>
      {selectedTeam === 'Free Hit' ? <FreeHitTeam /> : <WildCardTeam />}
    </Container>
  );
};

export default OptimalTeam;
