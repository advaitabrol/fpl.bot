import React from 'react';
import styled from 'styled-components';

const Dropdown = styled.select`
  padding: 0.5rem;
  width: 100%;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

const Label = styled.label`
  font-weight: bold;
  display: block;
  margin-bottom: 0.5rem;
`;

interface LabeledDropdownProps {
  label: string;
  options: string[] | number[];
  onChange: (value: string | number) => void;
  value?: string | number;
}

const LabeledDropdown: React.FC<LabeledDropdownProps> = ({
  label,
  options,
  onChange,
  value,
}) => (
  <div>
    <Label>{label}</Label>
    <Dropdown
      value={value}
      onChange={(e) => {
        const newValue =
          typeof options[0] === 'number'
            ? Number(e.target.value)
            : e.target.value;
        onChange(newValue);
      }}
    >
      <option value="">Select an option</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </Dropdown>
  </div>
);

export default LabeledDropdown;
