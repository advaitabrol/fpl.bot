export interface Player {
  id: string; // Add all relevant properties of a player
  name: string;
  team: string;
  selected: number;
  position: 'GK' | 'DEF' | 'MID' | 'FWD';
  isCaptain: boolean[];
  isBench: boolean[];
  price: number;
  expected_points: number[]; // Array of points for future matches
}

export interface Transfer {
  out: Player;
  in_player: Player;
  net_points: number; // Net gain in points after the transfer
}

export interface GetTransfersResponse {
  optimized_team: Player[];
  transfers_suggestion: TransferSuggestion[];
}

export interface TransferSuggestion {
  out: Player;
  in_player: Player;
}
export interface TransferConfiguration {
  team: Player[]; // Replace `any` with your specific Player type
  max_transfers: number;
  keep_players?: string[];
  avoid_players?: string[];
  keep_teams?: string[]; // Added for players to avoid
  avoid_teams?: string[]; // Added for teams to avoid
  desired_selected?: [number, number]; // Added for selection percentage range
  captain_scale?: number;
}

export interface PlayerWeek {
  player: Player;
  weekIndex?: number;
}
