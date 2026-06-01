/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { CalendarRange, ChevronDown } from 'lucide-react';
import { Match, Team } from '../types';
import TeamLogo from './TeamLogo';

interface MatchesListProps {
  upcomingMatches: Match[];
  finishedMatches: Match[];
  completedMatches: Match[];
  onHoverTeam?: (team: Team | null) => void;
  activeHoverTeam: Team | null;
  onSelectMatch?: (match: Match) => void;
  selectedMatchId?: string | null;
}

export default function MatchesList({
  upcomingMatches,
  finishedMatches,
  completedMatches,
  onHoverTeam,
  activeHoverTeam,
  onSelectMatch,
  selectedMatchId
}: MatchesListProps) {
  return (
    <div className="bg-app-surface border border-app-border rounded-2xl p-4 flex flex-col h-full overflow-y-auto select-none gap-4">
      {/* SECTION 1: UPCOMING MATCHES */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display font-bold text-gray-200 text-xs tracking-wider uppercase flex items-center gap-1.5 pt-1">
            Upcoming Matches
          </h3>
          <ChevronDown size={14} className="text-gray-500" />
        </div>

        <div className="flex flex-col gap-2.5">
          {upcomingMatches.map((match) => (
            <MatchCard
              key={match.id}
              match={match}
              onHover={(t) => onHoverTeam?.(t)}
              activeHoverTeam={activeHoverTeam}
              onSelect={onSelectMatch}
              isSelected={selectedMatchId === match.id}
            />
          ))}
        </div>
      </div>

      {/* SECTION 2: FINISHED DIVISION */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-display font-medium text-gray-500 text-xs tracking-wider uppercase pt-1">
            Finished
          </h3>
        </div>

        <div className="flex flex-col gap-2">
          {finishedMatches.map((match) => (
            <MatchCard
              key={match.id}
              match={match}
              onHover={(t) => onHoverTeam?.(t)}
              activeHoverTeam={activeHoverTeam}
              onSelect={onSelectMatch}
              isSelected={selectedMatchId === match.id}
            />
          ))}
        </div>
      </div>

      {/* SECTION 3: COMPLETED DIVISION */}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-display font-medium text-gray-500 text-xs tracking-wider uppercase pt-1">
            Completed
          </h3>
        </div>

        <div className="flex flex-col gap-2">
          {completedMatches.map((match) => (
            <MatchCard
              key={match.id}
              match={match}
              onHover={(t) => onHoverTeam?.(t)}
              activeHoverTeam={activeHoverTeam}
              onSelect={onSelectMatch}
              isSelected={selectedMatchId === match.id}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Internal reusable card representation for games/matches
interface MatchCardProps {
  key?: string | number;
  match: Match;
  onHover?: (team: Team | null) => void;
  activeHoverTeam: Team | null;
  onSelect?: (match: Match) => void;
  isSelected?: boolean;
}

function MatchCard({ match, onHover, activeHoverTeam, onSelect, isSelected }: MatchCardProps) {
  const isTeam1Hovered = activeHoverTeam?.id === match.team1?.id;
  const isTeam2Hovered = activeHoverTeam?.id === match.team2?.id;

  const hasScores = match.score1 !== undefined && match.score2 !== undefined;
  
  // Completed status check for highlighted teal overlays
  const isCompletedFinal = match.status === 'completed' && hasScores;

  return (
    <div 
      onClick={() => onSelect?.(match)}
      className={`rounded-xl border p-3 flex items-center justify-between cursor-pointer bg-[#110e1a]/70 hover:bg-[#151120] transition-colors select-none ${
        isSelected 
          ? 'border-brand-pink shadow-[0_0_12px_rgba(236,72,153,0.25)] bg-[#100b1a]'
          : isCompletedFinal 
            ? 'border-[#292243] hover:border-brand-pink/30' 
            : 'border-[#211a33] hover:border-brand-pink/20'
      }`}
    >
      {/* Team 1 Section */}
      <div 
        onMouseEnter={() => onHover?.(match.team1)}
        onMouseLeave={() => onHover?.(null)}
        className={`flex flex-col items-center justify-center w-[30%] gap-1.5 cursor-pointer rounded-lg p-1 transition-all ${
          isTeam1Hovered ? 'bg-purple-950/20 text-white font-bold scale-102' : 'text-gray-400'
        }`}
      >
        <TeamLogo name={match.team1.logo} size={26} />
        <span className="text-[10px] font-mono tracking-wide font-medium text-center truncate w-full select-none">
          {match.team1.shortName}
        </span>
        {hasScores && (
          <span className={`text-sm font-bold font-mono mt-0.5 ${match.score1! > match.score2! ? 'text-brand-orange' : 'text-gray-400'}`}>
            {match.score1}
          </span>
        )}
      </div>

      {/* Middle Status Timeline info */}
      <div className="flex flex-col items-center justify-center flex-1 max-w-[40%] text-center px-1">
        <span className="text-[10px] text-gray-500 font-mono font-medium tracking-normal select-none">
          {match.time || '13:00'}
        </span>
        <span className="text-[10px] text-gray-300 font-bold font-mono tracking-normal leading-tight select-none">
          {match.date || '10:00'}
        </span>
        
        {isCompletedFinal && (
          <span className="text-[9px] text-brand-orange border border-brand-orange/30 bg-orange-950/20 px-1.5 rounded uppercase font-bold tracking-widest font-mono mt-1 scale-90 select-none">
            Final
          </span>
        )}
      </div>

      {/* Team 2 Section */}
      <div 
        onMouseEnter={() => onHover?.(match.team2)}
        onMouseLeave={() => onHover?.(null)}
        className={`flex flex-col items-center justify-center w-[30%] gap-1.5 cursor-pointer rounded-lg p-1 transition-all ${
          isTeam2Hovered ? 'bg-purple-950/20 text-white font-bold scale-102' : 'text-gray-400'
        }`}
      >
        <TeamLogo name={match.team2.logo} size={26} />
        <span className="text-[10px] font-mono tracking-wide font-medium text-center truncate w-full select-none">
          {match.team2.shortName}
        </span>
        {hasScores && (
          <span className={`text-sm font-bold font-mono mt-0.5 ${match.score2! > match.score1! ? 'text-brand-orange' : 'text-gray-400'}`}>
            {match.score2}
          </span>
        )}
      </div>
    </div>
  );
}
