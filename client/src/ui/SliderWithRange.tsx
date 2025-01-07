import React from 'react';
import styled from 'styled-components';

const SliderContainer = styled.div`
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const Slider = styled.input`
  flex: 1;
`;

const SliderRange = styled.div`
  font-size: 0.9rem;
  text-align: center;
`;

interface SliderWithRangeProps {
  range: [number, number];
  onChange: (value: number, index: number) => void;
}

const SliderWithRange: React.FC<SliderWithRangeProps> = ({
  range,
  onChange,
}) => (
  <>
    <SliderContainer>
      <Slider
        type="range"
        min="1"
        max="100"
        value={range[0]}
        onChange={(e) => onChange(Number(e.target.value), 0)}
      />
      <Slider
        type="range"
        min="1"
        max="100"
        value={range[1]}
        onChange={(e) => onChange(Number(e.target.value), 1)}
      />
    </SliderContainer>
    <SliderRange>{`${range[0]}% - ${range[1]}%`}</SliderRange>
  </>
);

export default SliderWithRange;
