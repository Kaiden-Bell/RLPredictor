/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { TournamentData, BracketMatch, Match, PlayerRating, TeamForm, TeamRoster } from '../types';

export function generatePlaceholderData(url: string): TournamentData {
  const lowerUrl = url.toLowerCase();

  // 1. Identify context
  let game = 'Rocket League';
  let tournamentName = 'RLCS Major Championship';
  
  let teamPresets: { id: string; name: string; shortName: string; logo: string; color: string }[] = [];

  if (lowerUrl.includes('valorant') || lowerUrl.includes('vct') || lowerUrl.includes('champions')) {
    game = 'Valorant';
    tournamentName = 'VCT Champions - Berlin Stage';
    teamPresets = [
      { id: 'sen', name: 'Sentinels', shortName: 'SEN', logo: 'G2', color: '#ef4444' }, // map G2 logo style
      { id: 'fnc', name: 'Fnatic', shortName: 'FNC', logo: 'Vitality', color: '#ff5500' }, // map Vitality logo style
      { id: 'prx', name: 'Paper Rex', shortName: 'PRX', logo: 'Swnder', color: '#ec4899' },
      { id: 'geng', name: 'Gen.G Esports', shortName: 'GEN', logo: 'Karmine', color: '#eab308' },
      { id: 'th', name: 'Team Heretics', shortName: 'TH', logo: 'Noppes', color: '#15803d' },
      { id: 'edg', name: 'EDward Gaming', shortName: 'EDG', logo: 'Sinzline', color: '#111827' },
      { id: 'loud', name: 'LOUD Esports', shortName: 'LOUD', logo: 'Team 5WS', color: '#22c55e' },
      { id: 'drx', name: 'DRX Vision', shortName: 'DRX', logo: 'Audacity', color: '#2563eb' }
    ];
  } else if (lowerUrl.includes('cs') || lowerUrl.includes('counterstrike') || lowerUrl.includes('pgl') || lowerUrl.includes('iem')) {
    game = 'Counter-Strike 2';
    tournamentName = 'IEM Katowice - Championship Bracket';
    teamPresets = [
      { id: 'faze', name: 'FaZe Clan', shortName: 'FaZe', logo: 'G2', color: '#ef4444' },
      { id: 'navic', name: 'Natus Vincere', shortName: 'NaVi', logo: 'Vitality', color: '#eab308' },
      { id: 'g2cs', name: 'G2 Esports', shortName: 'G2', logo: 'G2', color: '#111827' },
      { id: 'vitcs', name: 'Team Vitality', shortName: 'Vitality', logo: 'Vitality', color: '#f59e0b' },
      { id: 'spirit', name: 'Team Spirit', shortName: 'Spirit', logo: 'Swnder', color: '#3b82f6' },
      { id: 'mouz', name: 'MOUZ Esports', shortName: 'MOUZ', logo: 'Noppes', color: '#ef4444' },
      { id: 'ast', name: 'Astralis', shortName: 'Astralis', logo: 'Team 5WS', color: '#ef4444' },
      { id: 'vp', name: 'Virtus.pro', shortName: 'VP', logo: 'Audacity', color: '#f97316' }
    ];
  } else if (lowerUrl.includes('lol') || lowerUrl.includes('league') || lowerUrl.includes('lck') || lowerUrl.includes('lcs') || lowerUrl.includes('lec') || lowerUrl.includes('worlds')) {
    game = 'League of Legends';
    tournamentName = 'League of Legends Worlds Cup';
    teamPresets = [
      { id: 't1', name: 'T1 Esports', shortName: 'T1', logo: 'G2', color: '#e11d48' },
      { id: 'geng', name: 'Gen.G LoL', shortName: 'GEN', logo: 'Karmine', color: '#eab308' },
      { id: 'wbg', name: 'Weibo Gaming', shortName: 'WBG', logo: 'Swnder', color: '#f97316' },
      { id: 'blg', name: 'Bilibili Gaming', shortName: 'BLG', logo: 'Audacity', color: '#06b6d4' },
      { id: 'fncl', name: 'Fnatic LoL', shortName: 'FNC', logo: 'Vitality', color: '#ff5500' },
      { id: 'g2l', name: 'G2 League', shortName: 'G2', logo: 'G2', color: '#111827' },
      { id: 'hle', name: 'Hanwha Life', shortName: 'HLE', logo: 'Noppes', color: '#f97316' },
      { id: 'fly', name: 'FlyQuest', shortName: 'FLY', logo: 'Team 5WS', color: '#15803d' }
    ];
  } else {
    // Default Rocket League mapping
    game = 'Rocket League';
    tournamentName = 'RLCS Major 1 - Copenhagen Playoffs';
    teamPresets = [
      { id: 'g2', name: 'G2 Esports', shortName: 'G2', logo: 'G2', color: '#111827' },
      { id: 'vit', name: 'Team Vitality', shortName: 'Vitality', logo: 'Vitality', color: '#eab308' },
      { id: 'kc', name: 'Karmine Corp', shortName: 'Karmine', logo: 'Karmine', color: '#3b82f6' },
      { id: 'swn', name: 'Swnder Esports', shortName: 'Swnder', logo: 'Swnder', color: '#f97316' },
      { id: 'nop', name: 'Noppes Esports', shortName: 'Noppes', logo: 'Noppes', color: '#a855f7' },
      { id: 'sin', name: 'Sinzline Gaming', shortName: 'Sinzline', logo: 'Sinzline', color: '#ef4444' },
      { id: 't5w', name: 'Team 5WS', shortName: 'Team 5WS', logo: 'Team 5WS', color: '#22c55e' },
      { id: 'aud', name: 'Audacity Team', shortName: 'Audacity', logo: 'Audacity', color: '#f59e0b' }
    ];
  }

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
    { name: game === 'Valorant' ? 'TenZ' : game === 'Counter-Strike 2' ? 'm0NESY' : game === 'League of Legends' ? 'Faker' : 'Alorin', rating: 7.94 },
    { name: game === 'Valorant' ? 'Boaster' : game === 'Counter-Strike 2' ? 'ZyWOo' : game === 'League of Legends' ? 'Chovy' : 'Kamerian', rating: 6.85 }
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

  const genericRosterNames = game === 'Valorant' ? [
    ['TenZ', 'zekken', 'johnqt', 'Sacy', 'Zellsis', 'Kaplan'],
    ['Boaster', 'Derke', 'Alfajer', 'Chronicle', 'Leo', 'Elmapuddy'],
    ['something', 'f0rsakeN', 'mindfreak', 'd4v41', 'Jinggg', 'alecks'],
    ['Chovy', 'Kiin', 'Canyon', 'Peyz', 'Lehends', 'Kim'],
    ['Boo', 'benjyfishy', 'MiniBoo', 'RieNs', 'Wo0t', 'neilzinho'],
    ['Nobody', 'Smoggy', 'Haodong', 'CHICHOO', 'ZmjKK', 'Muggle'],
    ['saadhak', 'Less', 'tuyz', 'cauanzin', 'Quick', 'pe固定'],
    ['stax', 'BuZz', 'MaKo', 'Foxy9', 'BeYN', 'terry']
  ] : game === 'Counter-Strike 2' ? [
    ['karrigan', 'rain', 'Broky', 'ropz', 'frozen', 'NEO'],
    ['Aleksib', 'iM', 'b1t', 'jL', 'w0nd3rful', 'B1ad3'],
    ['Snax', 'Hunter', 'Niko', 'm0NESY', 'malbsMd', 'TaZ'],
    ['apEX', 'ZyWOo', 'spinx', 'flameZ', 'mezii', 'XTQZZZ'],
    ['donk', 'sh1ro', 'chopper', 'magixx', 'zoner', 'hally'],
    ['siuhy', 'torzsi', 'Jimpphat', 'xertioN', 'Brollan', 'sycrone'],
    ['dev1ce', 'Staehr', 'jabbi', 'stavn', 'br0', 'ruggi'],
    ['Jame', 'FL1T', 'electroNic', 'fame', 'n0rb3r7', 'dastan']
  ] : game === 'League of Legends' ? [
    ['Zeus', 'Oner', 'Faker', 'Gumayusi', 'Keria', 'kkOma'],
    ['Kiin', 'Canyon', 'Chovy', 'Peyz', 'Lehends', 'Mata'],
    ['TheShy', 'Weiwei', 'Xiaohu', 'Light', 'Crisp', 'Daeny'],
    ['Bin', 'Xun', 'knight', 'Elk', 'ON', 'Easyhoon'],
    ['Oscarinin', 'Razork', 'Humanoid', 'Noah', 'Jun', 'Nightshare'],
    ['BrokenBlade', 'Yike', 'Caps', 'Hans Sama', 'Mikyx', 'Dylan'],
    ['Doran', 'Peanut', 'Zeka', 'Viper', 'Delight', 'DanDy'],
    ['Bwipo', 'Inspired', 'Jensen', 'Massu', 'Busio', 'Nukeduck']
  ] : rlPlayers;

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
