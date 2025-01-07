import React from 'react';
import styled from 'styled-components';

const Input = styled.input`
  padding: 0.5rem;
  width: 100%;
  max-width: 100%;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin-top: 0.5rem;
  box-sizing: border-box;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Label = styled.label`
  font-weight: bold;
  display: block;
  margin-bottom: 0.5rem;
`;

interface LabeledInputProps {
  label: string;
  placeholder: string;
  onEnter: (value: string) => void;
}

const LabeledInput: React.FC<LabeledInputProps> = ({
  label,
  placeholder,
  onEnter,
}) => (
  <div>
    <Label>{label}</Label>
    <Input
      placeholder={placeholder}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          onEnter((e.target as HTMLInputElement).value);
          (e.target as HTMLInputElement).value = '';
        }
      }}
    />
  </div>
);

export default LabeledInput;
