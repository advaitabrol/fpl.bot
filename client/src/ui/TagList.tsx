import React from 'react';
import styled from 'styled-components';

const TagContainer = styled.div`
  margin-top: 0.5rem;
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
`;

const Tag = styled.div`
  background: #007bff;
  color: white;
  padding: 0.5rem 0.75rem;
  border-radius: 20px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
`;

const RemoveButton = styled.button`
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 1rem;
  line-height: 0;
`;

interface TagListProps {
  tags: string[];
  onRemove: (tag: string) => void;
}

const TagList: React.FC<TagListProps> = ({ tags, onRemove }) => (
  <TagContainer>
    {tags.map((tag) => (
      <Tag key={tag}>
        {tag} <RemoveButton onClick={() => onRemove(tag)}>×</RemoveButton>
      </Tag>
    ))}
  </TagContainer>
);

export default TagList;
