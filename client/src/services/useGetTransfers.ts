import axios from 'axios';

interface GetTransfersParams {
  team: any[]; // Replace `any` with your specific Player type
  maxTransfers: number;
  keep?: string[];
  blacklist?: string[];
}

const useGetTransfers = () => {
  const getTransfers = async ({
    team,
    maxTransfers,
    keep = [],
    blacklist = [],
  }: GetTransfersParams) => {
    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/teams/suggest-transfers',
        {
          team,
          max_transfers: maxTransfers,
          keep,
          blacklist,
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
