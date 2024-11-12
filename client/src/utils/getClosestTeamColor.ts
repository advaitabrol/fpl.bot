import Fuse from 'fuse.js';
import teamColors from '../data/teamColors';

// Preprocess team names into an array for Fuse
const teamNames = Object.keys(teamColors);
const fuse = new Fuse(teamNames, {
  includeScore: true,
  threshold: 0.4, // Adjust as needed; lower means stricter match
});

/**
 * Get the closest matching team color for a given team name.
 * @param {string} team - The team name to match.
 * @returns {object} - The primary and secondary colors for the closest match.
 */
export function getClosestTeamColor(team: string) {
  const result = fuse.search(team);
  if (result.length > 0) {
    const closestMatch = result[0].item;
    return teamColors[closestMatch];
  } else {
    // Return a default color if no match is found
    return { primary: '#CCCCCC', secondary: '#555555' };
  }
}
