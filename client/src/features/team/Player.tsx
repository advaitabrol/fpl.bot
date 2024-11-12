import React from 'react';
import styled from 'styled-components';
import { useTeamColors } from '../../hooks/useTeamColors';
import PlayerIcon from '../../ui/PlayerIcon';

interface PlayerProps {
  player: {
    name: string;
    team: string;
    price: number;
    expected_points: number[];
    isCaptain: boolean[];
  };
  weekIndex?: number; // Optional prop to specify a specific week
}

const PlayerWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 0.5rem;
  width: 120px;
  position: relative;
`;

const CaptainIcon = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  transform: translate(30%, -20%);
  width: 15px;
  height: 15px;
  background-color: black;
  color: white;
  font-weight: bold;
  font-size: 0.8rem;
  border-radius: 50%;
  border: 1px solid #444;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
`;

const PlayerName = styled.div`
  font-weight: bold;
  text-align: center;
  margin-top: 0.5rem;
  height: 1.2rem;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
`;

const PointsBox = styled.div`
  display: flex;
  justify-content: center; /* Center the points horizontally */
  align-items: center;
  width: 100%;
  max-width: 90px;
  margin-top: 0.5rem;
  padding: 0.2rem 0;
  font-size: 0.8rem;
  text-align: center;
  gap: 12px; /* Increased gap for better spacing */
`;

const WeekPoints = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 30px; /* Ensures each point box has enough width */
  .week-label {
    font-size: 0.6rem;
    color: #666;
  }
  .points {
    font-weight: bold;
    padding-top: 0.1rem;
  }
`;
const Player: React.FC<PlayerProps> = ({ player, weekIndex }) => {
  const { name, team, price, expected_points, isCaptain } = player;
  const { primary, secondary } = useTeamColors(team);

  const isCaptainForCurrentWeek = isCaptain[0];
  const nameParts = name.split(' ');
  const displayName = `${nameParts[0]} ${nameParts[nameParts.length - 1]}`;

  return (
    <PlayerWrapper>
      {isCaptainForCurrentWeek && <CaptainIcon>C</CaptainIcon>}
      <PlayerIcon
        primaryColor={primary}
        secondaryColor={secondary}
        size="roster"
      />
      <PlayerName>{displayName}</PlayerName>
      <PointsBox>
        {weekIndex !== undefined ? (
          <div className="points">{expected_points[weekIndex]} pts</div>
        ) : (
          expected_points.slice(0, 3).map((points, index) => (
            <WeekPoints key={index}>
              <div className="week-label">Wk {index + 1}</div>
              <div className="points">{points}</div>
            </WeekPoints>
          ))
        )}
      </PointsBox>
      <div>£{price}m</div>
    </PlayerWrapper>
  );
};

export default Player;
