import React from 'react';
import styled from 'styled-components';

interface PlayerIconProps {
  primaryColor: string;
  secondaryColor: string;
  size?: 'roster' | 'icon'; // Define the size options
}

const IconWrapper = styled.svg<{
  size: 'roster' | 'icon';
  primaryColor: string;
  secondaryColor: string;
}>`
  width: ${({ size }) => (size === 'roster' ? '45px' : '30px')};
  height: ${({ size }) => (size === 'roster' ? '60px' : '40px')};
`;

const PlayerIcon: React.FC<PlayerIconProps> = ({
  primaryColor,
  secondaryColor,
  size = 'roster',
}) => (
  <IconWrapper
    primaryColor={primaryColor}
    secondaryColor={secondaryColor}
    size={size}
    viewBox="0 0 100 140"
  >
    <circle cx="50" cy="30" r="20" fill={secondaryColor} />
    <rect
      x="30"
      y="60"
      width="40"
      height="50"
      rx="10"
      ry="10"
      fill={primaryColor}
      stroke={secondaryColor}
      strokeWidth="2"
    />
  </IconWrapper>
);

export default PlayerIcon;
