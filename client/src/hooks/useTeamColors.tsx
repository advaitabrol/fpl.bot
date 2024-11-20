import { useMemo } from 'react';
import { getClosestTeamColor } from '../utils/getClosestTeamColor';

export const useTeamColors = (team: string) => {
  const colors = useMemo(() => getClosestTeamColor(team), [team]);
  return colors;
};
