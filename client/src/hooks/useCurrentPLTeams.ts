import { useState, useEffect } from 'react';
import { PREMIER_LEAGUE_TEAMS } from '../data/teams';

export const useCurrentPLTeams = () => {
  const [teams, setTeams] = useState<string[]>([]);

  useEffect(() => {
    // Simulate an API call or dynamic logic for fetching teams
    setTeams(PREMIER_LEAGUE_TEAMS);
  }, []);

  return teams;
};
