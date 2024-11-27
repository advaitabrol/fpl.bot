import React, { useState } from 'react';
import styled from 'styled-components';
import { useCurrentPLTeams } from '../../hooks/useCurrentPLTeams';
import TagList from '../../ui/TagList';
import SliderWithRange from '../../ui/SliderWithRange';
import LabeledDropdown from '../../ui/LabeledDropdown';
import LabeledInput from '../../ui/LabeledInput';
import SuggestedTransfer from './SuggestedTransfer';
import NetTransferRecap from './NetTransferRecap';

// Styled Components
const ModalBackground = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
`;

const ModalContainer = styled.div`
  position: relative;
  background: #fff;
  border-radius: 8px;
  padding: 1.5rem;
  width: 90%;
  max-width: 800px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  z-index: 10000;
`;

const CloseButton = styled.button`
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: none;
  border: none;
  font-size: 1.5rem;
  font-weight: bold;
  cursor: pointer;
  color: #333;

  &:hover {
    color: #ff0000;
  }
`;

const Header = styled.h2`
  text-align: center;
  margin-bottom: 1.5rem;
`;

const HalvesContainer = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 2rem;
`;

const HalfSection = styled.div`
  flex: 1;
  padding: 1rem;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 8px;
`;

const SubmitButton = styled.button`
  margin-top: 2rem;
  width: 100%;
  padding: 0.8rem 1.2rem;
  background: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;

  &:hover {
    background: #218838;
  }
`;

// Spinner Components
const SpinnerOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10001;
  border-radius: 8px;
`;

const SpinnerContainer = styled.div`
  text-align: center;

  .spinner {
    margin: 0 auto;
    width: 50px;
    height: 50px;
    border: 4px solid rgba(0, 0, 0, 0.2);
    border-top: 4px solid black;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  p {
    margin-top: 1rem;
    font-size: 1.2rem;
    color: black;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

const ModalContent = styled.div<{ loading: boolean }>`
  opacity: ${({ loading }) => (loading ? 0.5 : 1)};
  pointer-events: ${({ loading }) => (loading ? 'none' : 'auto')};
  position: relative;
`;

// Props
interface ConfigureTransfersProps {
  team: any[]; // Replace with your specific Player type
  onClose: () => void;
  onSubmit: (data: {
    maxTransfers: number;
    keepTeams: string[];
    avoidTeams: string[];
    keepPlayers: string[];
    avoidPlayers: string[];
    selectedRange: [number, number];
  }) => void;
  view: 'configure' | 'suggestions';
  suggestedTransfers?: any[];
  totalNetPoints?: number;
  totalNetCost?: number;
  onApprove: () => void;
  onReject: () => void;
}

// Component
const ConfigureTransfers: React.FC<ConfigureTransfersProps> = ({
  team,
  onClose,
  onSubmit,
  view,
  suggestedTransfers = [],
  totalNetPoints = 0,
  totalNetCost = 0,
  onApprove,
  onReject,
}) => {
  const teams = useCurrentPLTeams();

  const [maxTransfers, setMaxTransfers] = useState(1);
  const [keepTeams, setKeepTeams] = useState<string[]>([]);
  const [avoidTeams, setAvoidTeams] = useState<string[]>([]);
  const [keepPlayers, setKeepPlayers] = useState<string[]>([]);
  const [avoidPlayers, setAvoidPlayers] = useState<string[]>([]);
  const [selectedRange, setSelectedRange] = useState<[number, number]>([
    1, 100,
  ]);

  const [loading, setLoading] = useState(false);

  const handleSliderChange = (value: number, index: number) => {
    setSelectedRange((prevRange) => {
      const newRange = [...prevRange] as [number, number];
      newRange[index] = value;

      if (newRange[0] > newRange[1]) {
        if (index === 0) newRange[1] = newRange[0];
        else newRange[0] = newRange[1];
      }

      return newRange;
    });
  };

  const handleAddUnique = (
    item: string,
    list: string[],
    setList: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    if (!list.includes(item)) setList([...list, item]);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await onSubmit({
        maxTransfers,
        keepTeams,
        avoidTeams,
        keepPlayers,
        avoidPlayers,
        selectedRange,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <ModalBackground onClick={onClose}>
      <ModalContainer onClick={(e) => e.stopPropagation()}>
        <CloseButton onClick={onClose}>&times;</CloseButton>

        {loading && (
          <SpinnerOverlay>
            <SpinnerContainer>
              <div className="spinner"></div>
              <p>Hang tight, this can take up to a few minutes.</p>
            </SpinnerContainer>
          </SpinnerOverlay>
        )}

        <ModalContent loading={loading}>
          <Header>
            {view === 'configure'
              ? 'Configure Transfers'
              : 'Transfer Suggestions'}
          </Header>

          {view === 'configure' ? (
            <>
              <LabeledDropdown
                label="Maximum Transfers"
                options={[1, 2, 3, 4, 5]}
                value={maxTransfers}
                onChange={(value) => setMaxTransfers(Number(value))}
              />

              <HalvesContainer>
                <HalfSection>
                  <LabeledInput
                    label="Keep Players"
                    placeholder="Type player name and press Enter"
                    onEnter={(player) =>
                      handleAddUnique(player, keepPlayers, setKeepPlayers)
                    }
                  />
                  <TagList
                    tags={keepPlayers}
                    onRemove={(player) =>
                      setKeepPlayers(keepPlayers.filter((p) => p !== player))
                    }
                  />

                  <LabeledDropdown
                    label="Keep Teams"
                    options={teams}
                    onChange={(team) =>
                      handleAddUnique(team as string, keepTeams, setKeepTeams)
                    }
                  />
                  <TagList
                    tags={keepTeams}
                    onRemove={(team) =>
                      setKeepTeams(keepTeams.filter((t) => t !== team))
                    }
                  />
                </HalfSection>

                <HalfSection>
                  <LabeledInput
                    label="Avoid Players"
                    placeholder="Type player name and press Enter"
                    onEnter={(player) =>
                      handleAddUnique(player, avoidPlayers, setAvoidPlayers)
                    }
                  />
                  <TagList
                    tags={avoidPlayers}
                    onRemove={(player) =>
                      setAvoidPlayers(avoidPlayers.filter((p) => p !== player))
                    }
                  />

                  <LabeledDropdown
                    label="Avoid Teams"
                    options={teams}
                    onChange={(team) =>
                      handleAddUnique(team as string, avoidTeams, setAvoidTeams)
                    }
                  />
                  <TagList
                    tags={avoidTeams}
                    onRemove={(team) =>
                      setAvoidTeams(avoidTeams.filter((t) => t !== team))
                    }
                  />

                  <SliderWithRange
                    range={selectedRange}
                    onChange={handleSliderChange}
                  />
                </HalfSection>
              </HalvesContainer>

              <SubmitButton onClick={handleSubmit}>Submit</SubmitButton>
            </>
          ) : (
            <>
              {suggestedTransfers.map((transfer, index) => (
                <SuggestedTransfer
                  key={index}
                  outPlayer={transfer.out}
                  inPlayer={transfer.in}
                  netPoints={transfer.netPoints}
                />
              ))}
              <NetTransferRecap
                totalNetPoints={totalNetPoints}
                totalNetCost={totalNetCost}
                onApprove={onApprove}
                onReject={onReject}
              />
            </>
          )}
        </ModalContent>
      </ModalContainer>
    </ModalBackground>
  );
};

export default ConfigureTransfers;
