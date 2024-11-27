import axios from 'axios';
import { TransferConfiguration } from '../services/interfaces';

const useGetTransfers = () => {
  const getTransfers = async ({
    team,
    max_transfers,
    keep_players = [],
    avoid_players = [],
    keep_teams = [],
    avoid_teams = [],
    desired_selected = [0, 100],
    captain_scale = 2.0,
  }: TransferConfiguration) => {
    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/teams/suggest-transfers',
        {
          team,
          max_transfers,
          keep_players,
          avoid_players,
          keep_teams,
          avoid_teams,
          desired_selected,
          captain_scale,
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching transfers:', error);
      throw error;
    }
  };

  return { getTransfers };
};

export default useGetTransfers;
