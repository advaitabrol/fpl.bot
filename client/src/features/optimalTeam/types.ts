export interface Player {
  name: string;
  team: string;
  position: 'GK' | 'DEF' | 'MID' | 'ATT'; // Specific positions
  price: number;
  expected_points: number[]; // Array for weekly projected points
  isBench: boolean[]; // Array indicating bench status for each week
  isCaptain: boolean[]; // Array indicating captaincy status for each week
}

// Define the structure of the team data
export interface TeamData {
  team: Player[]; // Array of players
}
