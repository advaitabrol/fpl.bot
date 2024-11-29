import { useState } from 'react';
import axios from 'axios';

import {
  TeamSearchResponse,
  TeamDetailsResponse,
} from '../services/interfaces';

const useTeamSearch = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [manualTeamId, setManualTeamId] = useState<string>('');
  const [filteredTeams, setFilteredTeams] = useState<TeamSearchResponse[]>([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [teamData, setTeamData] = useState<TeamDetailsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false); // Tracks whether a search is in progress

  const handleSearch = async () => {
    setLoading(true);
    setIsSearching(true); // Begin a new search
    setFilteredTeams([]);
    setSelectedTeamId(null);
    try {
      const response = await axios.get(
        'http://127.0.0.1:8000/teams/search-team-name',
        { params: { team_name: searchQuery } }
      );
      setFilteredTeams(response.data.teams || []);
    } catch (error) {
      console.error('Error fetching teams:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTeamFetch = async (teamId: string) => {
    setLoading(true);
    try {
      const response = await axios.get(
        'http://127.0.0.1:8000/teams/your-team',
        { params: { team_id: teamId } }
      );
      console.log(response.data);
      setTeamData(response.data);
      setIsSearching(false); // Stop searching once a team is fetched
    } catch (error) {
      console.error('Error fetching team details:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFilteredTeams([]);
    setSelectedTeamId(null);
    setManualTeamId('');
    setIsSearching(false); // Exit search mode
  };

  return {
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
  };
};

export default useTeamSearch;
