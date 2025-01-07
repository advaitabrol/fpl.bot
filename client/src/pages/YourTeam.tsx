import React from 'react';
import styled from 'styled-components';
import useTeamSearch from '../hooks/useTeamSearch';
import SearchBar from '../features/yourTeam/TeamSearchBar';
import TeamList from '../features/yourTeam/TeamList';
import TeamDetails from '../features/yourTeam/TeamDetails';

const PageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  height: 100vh;
  padding: 2rem;
  box-sizing: border-box;
`;

const ContentWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  width: 100%;
`;

const YourTeam: React.FC = () => {
  const {
    searchQuery,
    setSearchQuery,
    manualTeamId,
    setManualTeamId,
    filteredTeams,
    selectedTeamId,
    setSelectedTeamId,
    teamData,
    loading,
    isSearching,
    handleSearch,
    handleTeamFetch,
    resetState,
  } = useTeamSearch();

  const handleConfirmSelection = () => {
    const confirmedTeamId = manualTeamId || selectedTeamId;
    if (confirmedTeamId) {
      handleTeamFetch(confirmedTeamId);
      setSearchQuery(''); // Clear search query
      resetState(); // Exit search mode
    }
  };

  const handleCancelSelection = () => {
    setSearchQuery(''); // Clear the search bar text
    resetState(); // Reset the search state but retain teamData
  };

  return (
    <PageWrapper>
      <ContentWrapper>
        {!loading && (
          <SearchBar
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            onSearch={handleSearch}
          />
        )}

        {loading && <div>Loading...</div>}

        {isSearching && !loading && filteredTeams.length > 0 && (
          <TeamList
            teams={filteredTeams}
            selectedTeamId={selectedTeamId}
            onSelectTeam={setSelectedTeamId}
            manualTeamId={manualTeamId}
            setManualTeamId={setManualTeamId}
            onConfirm={handleConfirmSelection}
            onCancel={handleCancelSelection}
          />
        )}

        {!isSearching && teamData && <TeamDetails teamData={teamData} />}
      </ContentWrapper>
    </PageWrapper>
  );
};

export default YourTeam;
