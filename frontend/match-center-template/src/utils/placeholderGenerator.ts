/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { TournamentData, BracketMatch, Match, PlayerRating, TeamForm, TeamRoster } from '../types';

export function generatePlaceholderData(url: string, sections?: string[]): TournamentData {
  const lowerUrl = url.toLowerCase();

  // Strict Rocket League validation
  if (!lowerUrl.startsWith('https://liquipedia.net/rocketleague/') && !lowerUrl.startsWith('http://liquipedia.net/rocketleague/')) {
    throw new Error('URL Error: RLPredictor exclusively analyzes Rocket League on Liquipedia. Please provide a URL starting with https://liquipedia.net/rocketleague/');
  }

  // Default Rocket League mapping
  const game = 'Rocket League';
  let tournamentName = 'RLCS Major 1 - Copenhagen Playoffs';
  const teamPresets = [
    { id: 'g2', name: 'G2 Esports', shortName: 'G2', logo: 'G2', color: '#111827' },
    { id: 'vit', name: 'Team Vitality', shortName: 'Vitality', logo: 'Vitality', color: '#eab308' },
    { id: 'kc', name: 'Karmine Corp', shortName: 'Karmine', logo: 'Karmine', color: '#3b82f6' },
    { id: 'swn', name: 'Swnder Esports', shortName: 'Swnder', logo: 'Swnder', color: '#f97316' },
    { id: 'nop', name: 'Noppes Esports', shortName: 'Noppes', logo: 'Noppes', color: '#a855f7' },
    { id: 'sin', name: 'Sinzline Gaming', shortName: 'Sinzline', logo: 'Sinzline', color: '#ef4444' },
    { id: 't5w', name: 'Team 5WS', shortName: 'Team 5WS', logo: 'Team 5WS', color: '#22c55e' },
    { id: 'aud', name: 'Audacity Team', shortName: 'Audacity', logo: 'Audacity', color: '#f59e0b' }
  ];

  // Extract tournament slug name from link if possible
  try {
    const parsedUrl = new URL(url);
    const pathParts = parsedUrl.pathname.split('/').filter(Boolean);
    if (pathParts.length > 0) {
      const slug = pathParts[pathParts.length - 1]
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      const category = pathParts.length > 1 ? pathParts[pathParts.length - 2]
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ') : '';

      if (slug.length > 3) {
        tournamentName = `Scraped: ${category ? category + ' ' : ''}${slug}`;
      }
    }
  } catch (e) {
    // Keep default
  }

  // Define Bracket matchups (roundIndex: 0)
  const bracketMatches: BracketMatch[] = [
    // Quarterfinals
    { id: 'q1', matchIndex: 0, team1: teamPresets[0], team2: teamPresets[1], score1: 3, score2: 2, winnerId: teamPresets[0].id, status: 'completed', roundIndex: 0 },
    { id: 'q2', matchIndex: 1, team1: teamPresets[2], team2: teamPresets[3], score1: 3, score2: 1, winnerId: teamPresets[2].id, status: 'completed', roundIndex: 0 },
    { id: 'q3', matchIndex: 2, team1: teamPresets[4], team2: teamPresets[5], score1: 1, score2: 3, winnerId: teamPresets[5].id, status: 'completed', roundIndex: 0 },
    { id: 'q4', matchIndex: 3, team1: teamPresets[6], team2: teamPresets[7], score1: 2, score2: 3, winnerId: teamPresets[7].id, status: 'completed', roundIndex: 0 },
    
    // Semifinals
    { id: 's1', matchIndex: 0, team1: teamPresets[0], team2: teamPresets[2], score1: 4, score2: 3, winnerId: teamPresets[0].id, status: 'completed', roundIndex: 1 },
    { id: 's2', matchIndex: 1, team1: teamPresets[5], team2: teamPresets[7], score1: 1, score2: 4, winnerId: teamPresets[7].id, status: 'completed', roundIndex: 1 },
    
    // Finals
    { id: 'f1', matchIndex: 0, team1: teamPresets[0], team2: teamPresets[7], status: 'live', roundIndex: 2 }
  ];

  // Define sidebar upcoming layout matches list
  const upcomingMatches: Match[] = [
    { id: 'u1', team1: teamPresets[0], team2: teamPresets[3], status: 'upcoming', time: '19:00', date: 'Daily 12:00' },
    { id: 'u2', team1: teamPresets[1], team2: teamPresets[7], status: 'completed', score1: 1, score2: 3, time: '19:30', date: 'Daily 12:00' }
  ];

  const finishedMatches: Match[] = [
    { id: 'fi1', team1: teamPresets[0], team2: teamPresets[2], status: 'finished', time: '13:00', date: 'Last 17:00' }
  ];

  const completedMatches: Match[] = [
    { id: 'c1', team1: teamPresets[4], team2: teamPresets[7], status: 'completed', score1: 1, score2: 3, time: '13:00', date: 'Completed 13:00' },
    { id: 'c2', team1: teamPresets[2], team2: teamPresets[3], status: 'completed', score1: 3, score2: 1, time: '13:00', date: 'Completed 10:00' },
    { id: 'c3', team1: teamPresets[1], team2: teamPresets[2], status: 'completed', score1: 2, score2: 2, time: '13:00', date: 'Completed 9:30' }
  ];

  // Win probability
  const winProbability = {
    team1: teamPresets[0],
    team2: teamPresets[7],
    prob1: 76,
    prob2: 64
  };

  // Standout Player performance ratings based on rosters
  const playerRatings: PlayerRating[] = [
    { name: 'Alorin', rating: 7.94 },
    { name: 'Kamerian', rating: 6.85 }
  ];

  // Standout Recent Team streaks
  const teamForms: TeamForm[] = [
    { teamName: teamPresets[0].shortName, logo: teamPresets[0].logo, form: ['W', 'W', 'W', 'L', 'W', 'L'] },
    { teamName: teamPresets[7].shortName, logo: teamPresets[7].logo, form: ['L', 'W', 'W', 'W', 'W', 'W'] }
  ];

  // Construct Custom dynamic rosters mapping for hovering
  const rosters: Record<string, TeamRoster> = {};
  
  // Game rosters presets
  const rlPlayers = [
    ['Alorin', 'Binder', 'Data', 'Sugan', 'Rauksl', 'Markeen'],
    ['Porth', 'Evinik', 'Nersoc', 'Dexter', 'Aero', 'Sizz'],
    ['Zen', 'Alpha54', 'Radosin', 'FairyPeak', 'Kaydop', 'Ferro'],
    ['Vatira', 'Atow.', 'Rise.', 'Kamet0', 'Ferra', 'Eversax'],
    ['Swndr_1', 'Spider', 'Cobweb', 'Worm', 'Phlox', 'Viper'],
    ['Noppe_1', 'Shield', 'Guard', 'Bastion', 'Rampart', 'Vanguard'],
    ['Sinz_Red', 'Line', 'Track', 'Cross', 'Dot', 'Grid'],
    ['Five', 'Wave', 'Storm', 'Tide', 'Gale', 'Breeze']
  ];

  const genericRosterNames = rlPlayers;

  teamPresets.forEach((team, idx) => {
    const list = genericRosterNames[idx] || genericRosterNames[0];
    rosters[team.id] = {
      teamId: team.id,
      active: [
        { id: `${team.id}-p1`, name: list[0], role: 'Core Starter' },
        { id: `${team.id}-p2`, name: list[1], role: 'Core Starter' },
        { id: `${team.id}-p3`, name: list[2], role: 'Core Starter' },
        { id: `${team.id}-p4`, name: list[3], role: 'Core Starter' }
      ],
      substitutes: [
        { id: `${team.id}-sub1`, name: list[4], role: 'Active Substitute' },
        { id: `${team.id}-coach`, name: list[5], role: 'Strategic Coach' }
      ]
    };
  });

  return {
    name: tournamentName,
    url,
    game,
    bracketMatches,
    upcomingMatches,
    finishedMatches,
    completedMatches,
    winProbability,
    playerRatings,
    teamForms,
    rosters
  };
}
