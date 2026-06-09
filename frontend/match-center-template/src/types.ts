/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Team {
  id: string;
  name: string;
  shortName: string;
  logo: string; // Key or name of logo icon (e.g. 'G2', 'Vitality', 'Karmine', etc.)
  color?: string; // Hex color or Tailwind color class
}

export interface Player {
  id: string;
  name: string;
  rating?: number;
  avatarColor?: string;
  role?: string;
}

export interface TeamRoster {
  teamId: string;
  active: Player[];
  substitutes: Player[];
}

export interface Match {
  id: string;
  team1: Team;
  team2: Team;
  score1?: number;
  score2?: number;
  status: 'upcoming' | 'finished' | 'completed' | 'live';
  time?: string;
  date?: string;
  winProbability?: number; // probability of team1 winning, from 0 to 100
}

export interface BracketMatch {
  id: string;
  matchIndex: number; // 0-based index for layout positioning
  team1?: Team;
  team2?: Team;
  score1?: number;
  score2?: number;
  winnerId?: string;
  status: 'scheduled' | 'live' | 'completed';
  roundIndex: number; // 0: Quarterfinals, 1: Semifinals, 2: Finals
  section?: string;   // e.g. "Playoffs", "Group A", "Group B"
  round?: string;     // e.g. "Upper Bracket Quarter-Finals", "Round 1"
  bestOf?: number;    // e.g. 5 or 7
}

export interface PlayerRating {
  name: string;
  rating: number;
  isCustom?: boolean;
}

export interface TeamForm {
  teamName: string;
  logo: string;
  form: ('W' | 'L')[];
}

export interface TournamentData {
  name: string;
  url: string;
  game: string;
  bracketMatches: BracketMatch[];
  upcomingMatches: Match[];
  finishedMatches: Match[];
  completedMatches: Match[];
  winProbability: {
    team1: Team;
    team2: Team;
    prob1: number;
    prob2: number;
  };
  playerRatings: PlayerRating[];
  teamForms: TeamForm[];
  rosters: Record<string, TeamRoster>; // Map from teamId to roster
}
