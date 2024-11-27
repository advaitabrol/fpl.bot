import axios from 'axios';

// Define the structure of the Player type
export interface Player {
  name: string;
  team: string;
  position: string;
  price: number;
  expected_points: number[];
  [key: string]: any; // Add additional fields if necessary
}

const useOptimizeTeam = () => {
  const optimizeTeam = async (team: Player[]) => {
    try {
      // API call to optimize the team
      const response = await axios.post(
        'http://127.0.0.1:8000/teams/optimize-team',
        {
          team,
        }
      );

      // Return the optimized team
      return response.data.optimal;
    } catch (error) {
      console.error('Error optimizing team:', error);
      throw error;
    }
  };

  return { optimizeTeam };
};

export default useOptimizeTeam;
