/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { User, Layers, Share2, HelpCircle } from 'lucide-react';
import { BracketMatch, Team, TeamRoster } from '../types';
import TeamLogo from './TeamLogo';

interface BracketProps {
  bracketMatches: BracketMatch[];
  rosters: Record<string, TeamRoster>;
  onHoverTeam?: (team: Team | null) => void;
  activeHoverTeam: Team | null;
  selectedMatchId?: string | null;
  onSelectMatch?: (match: BracketMatch) => void;
}

export default function Bracket({ 
  bracketMatches, 
  rosters, 
  onHoverTeam, 
  activeHoverTeam,
  selectedMatchId,
  onSelectMatch 
}: BracketProps) {
  const [selectedRosterTeam, setSelectedRosterTeam] = useState<Team | null>(null);
  const [hoveredMatchId, setHoveredMatchId] = useState<string | null>(null);

  // Group matches by roundIndex
  const round0 = bracketMatches.filter(m => m.roundIndex === 0).sort((a, b) => a.matchIndex - b.matchIndex);
  const round1 = bracketMatches.filter(m => m.roundIndex === 1).sort((a, b) => a.matchIndex - b.matchIndex);
  const round2 = bracketMatches.filter(m => m.roundIndex === 2).sort((a, b) => a.matchIndex - b.matchIndex);

  // Automatically fetch comparative team for footer inside the popover
  const getComparativeOpponent = (team: Team): Team | null => {
    // Find a match containing this team
    const match = bracketMatches.find(m => m.team1?.id === team.id || m.team2?.id === team.id);
    if (!match) return null;
    if (match.team1?.id === team.id) return match.team2 || null;
    return match.team1 || null;
  };

  const currentRoster = selectedRosterTeam ? rosters[selectedRosterTeam.id] : null;
  const opponentTeam = selectedRosterTeam ? getComparativeOpponent(selectedRosterTeam) : null;

  return (
    <div className="bg-app-surface border border-app-border rounded-2xl h-full flex flex-col p-6 overflow-hidden relative select-none">
      {/* Bracket Header row */}
      <div className="flex items-center justify-between mb-8 flex-shrink-0 z-10">
        <div className="flex items-center gap-1.5">
          <Layers size={18} className="text-brand-pink" />
          <h2 className="font-display font-semibold text-gray-200 text-sm md:text-base tracking-wide uppercase">
            Active interactive Tournament Bracket
          </h2>
        </div>

        <div className="flex items-center gap-3">
          {selectedMatchId ? (
            <div className="hidden sm:flex items-center gap-1.5 bg-brand-pink/10 border border-brand-pink/30 px-2.5 py-1 rounded-full text-[9px] uppercase font-bold tracking-wider text-brand-pink animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-pink"></span>
              <span>Match Selected</span>
            </div>
          ) : (
            <div className="hidden sm:flex items-center gap-1.5 bg-cyan-950/20 border border-brand-glow/30 px-2.5 py-1 rounded-full text-[9px] uppercase font-bold tracking-wider text-brand-glow">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-glow animate-ping"></span>
              <span>Click Match to Analyze</span>
            </div>
          )}
          
          {/* Legend and help tools */}
          <button className="flex items-center gap-1.5 bg-purple-950/20 hover:bg-purple-900/30 text-gray-400 hover:text-white border border-app-border px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors">
            <Share2 size={13} />
            <span>Bracket</span>
            <ChevronDownIcon />
          </button>
        </div>
      </div>

      {/* Bracket Tree with Columns */}
      <div className="flex-1 flex justify-between items-center gap-4 relative min-h-[460px] py-4">
        {/* Quarter-Finals (4 matches, 8 teams) */}
        <div className="flex flex-col justify-between h-full w-[24%] relative z-10">
          <div className="text-[10px] text-gray-500 font-mono tracking-widest uppercase mb-1.5 text-center">Quarter-Finals</div>
          <div className="flex-1 flex flex-col justify-around gap-4">
            {round0.map((match) => (
              <div 
                key={match.id}
                onClick={() => onSelectMatch?.(match)}
                onMouseEnter={() => setHoveredMatchId(match.id)}
                onMouseLeave={() => setHoveredMatchId(null)}
                className={`flex flex-col gap-1.5 p-2 rounded-xl border transition-all duration-200 cursor-pointer bg-[#120e1d]/90 ${
                  selectedMatchId === match.id 
                    ? 'border-brand-pink shadow-[0_0_15px_rgba(236,72,153,0.35)] bg-pink-950/15'
                    : hoveredMatchId === match.id 
                      ? 'border-brand-pink/50 shadow-[0_0_12px_rgba(236,72,153,0.15)] bg-purple-950/10' 
                      : 'border-[#27213d] hover:border-brand-pink/30'
                }`}
              >
                <TeamSlot 
                  team={match.team1} 
                  score={match.score1} 
                  isWinner={match.winnerId === match.team1?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
                <TeamSlot 
                  team={match.team2} 
                  score={match.score2} 
                  isWinner={match.winnerId === match.team2?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Connecting SVGs (Quarter to Semi) */}
        <div className="absolute left-[24%] right-[52%] top-14 bottom-4 pointer-events-none hidden md:block">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            {/* Draw orthogonal connects */}
            {/* Connection 1 (matches 0, 1 to Semis 0) */}
            <path d="M 0 45 L 25 45 C 35 45, 35 90, 45 90" stroke="#2e2645" strokeWidth="1.5" fill="none" />
            <path d="M 0 135 L 25 135 C 35 135, 35 90, 45 90" stroke="#2e2645" strokeWidth="1.5" fill="none" />
            
            {/* Connection 2 (matches 2, 3 to Semis 1) */}
            <path d="M 0 225 L 25 225 C 35 225, 35 270, 45 270" stroke="#2e2645" strokeWidth="1.5" fill="none" />
            <path d="M 0 315 L 25 315 C 35 315, 35 270, 45 270" stroke="#2e2645" strokeWidth="1.5" fill="none" />
          </svg>
        </div>

        {/* Semi-Finals (2 matches, 4 teams) */}
        <div className="flex flex-col justify-between h-full w-[24%] relative z-10 mx-auto">
          <div className="text-[10px] text-gray-500 font-mono tracking-widest uppercase mb-1.5 text-center">Semi-Finals</div>
          <div className="flex-1 flex flex-col justify-around gap-12">
            {round1.map((match) => (
              <div 
                key={match.id}
                onClick={() => onSelectMatch?.(match)}
                onMouseEnter={() => setHoveredMatchId(match.id)}
                onMouseLeave={() => setHoveredMatchId(null)}
                className={`flex flex-col gap-1.5 p-2 rounded-xl border transition-all duration-200 cursor-pointer bg-[#120e1d]/90 ${
                  selectedMatchId === match.id 
                    ? 'border-brand-pink shadow-[0_0_15px_rgba(236,72,153,0.35)] bg-pink-950/15'
                    : hoveredMatchId === match.id 
                      ? 'border-brand-pink/50 shadow-[0_0_12px_rgba(236,72,153,0.15)] bg-purple-950/10' 
                      : 'border-[#27213d] hover:border-brand-pink/30'
                }`}
              >
                <TeamSlot 
                  team={match.team1} 
                  score={match.score1} 
                  isWinner={match.winnerId === match.team1?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
                <TeamSlot 
                  team={match.team2} 
                  score={match.score2} 
                  isWinner={match.winnerId === match.team2?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Connecting SVGs (Semi to Final) */}
        <div className="absolute left-[52%] right-[24%] top-14 bottom-4 pointer-events-none hidden md:block">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <path d="M 0 90 L 30 90 C 40 90, 40 180, 50 180" stroke="#2e2645" strokeWidth="1.5" fill="none" />
            <path d="M 0 270 L 30 270 C 40 270, 40 180, 50 180" stroke="#2e2645" strokeWidth="1.5" fill="none" />
          </svg>
        </div>

        {/* Grand Finale (1 match, 2 teams) */}
        <div className="flex flex-col justify-between h-full w-[24%] relative z-10">
          <div className="text-[10px] text-gray-500 font-mono tracking-widest uppercase mb-1.5 text-center">Grand Final</div>
          <div className="flex-1 flex flex-col justify-center">
            {round2.map((match) => (
              <div 
                key={match.id}
                onClick={() => onSelectMatch?.(match)}
                onMouseEnter={() => setHoveredMatchId(match.id)}
                onMouseLeave={() => setHoveredMatchId(null)}
                className={`flex flex-col gap-1.5 p-2 rounded-xl border transition-all duration-200 cursor-pointer bg-[#120e1d]/90 ${
                  selectedMatchId === match.id 
                    ? 'border-brand-pink shadow-[0_0_15px_rgba(236,72,153,0.35)] bg-pink-950/15'
                    : hoveredMatchId === match.id 
                      ? 'border-brand-pink/50 shadow-[0_0_12px_rgba(236,72,153,0.15)] bg-purple-950/10' 
                      : 'border-brand-orange/50 shadow-[0_0_12px_rgba(249,115,22,0.1)] hover:border-brand-pink/30'
                }`}
              >
                <TeamSlot 
                  team={match.team1} 
                  score={match.score1} 
                  isWinner={match.winnerId === match.team1?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
                <TeamSlot 
                  team={match.team2} 
                  score={match.score2} 
                  isWinner={match.winnerId === match.team2?.id}
                  onHover={(t) => { onHoverTeam?.(t); if(t) setSelectedRosterTeam(t); }}
                  activeHoverId={activeHoverTeam?.id}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Right Pending Trophy Cup or Final slot */}
        <div className="flex flex-col justify-center h-full w-[15%] items-center z-13">
          <div className="w-12 h-12 rounded-xl bg-purple-950/20 border border-purple-900/30 flex items-center justify-center text-gray-600 cursor-pointer hover:border-brand-pink hover:text-brand-pink transition-all">
            <span className="font-bold text-lg">+</span>
          </div>
        </div>

        {/* ROSTER HOVER POPUP OVERLAY */}
        {selectedRosterTeam && currentRoster && (
          <div 
            id="roster-overlay-popover"
            className="absolute top-2 right-[20%] w-72 bg-[#171328] border border-brand-orange/60 rounded-2xl p-4 shadow-[0_15px_30px_rgba(0,0,0,0.8),0_0_15px_rgba(249,115,22,0.15)] z-40 animate-fade-in text-left flex flex-col gap-3.5 select-none"
            onMouseLeave={() => setSelectedRosterTeam(null)}
          >
            {/* Header: Team Title */}
            <div className="flex items-center justify-between pb-2 border-b border-purple-900/40">
              <div className="flex items-center gap-2">
                <TeamLogo name={selectedRosterTeam.logo} size={22} />
                <span className="font-display font-bold text-white text-base">{selectedRosterTeam.name}</span>
              </div>
              <button 
                onClick={() => setSelectedRosterTeam(null)}
                className="text-gray-500 hover:text-white transition-colors"
                title="Close"
              >
                &times;
              </button>
            </div>

            {/* Active Roster players block */}
            <div>
              <div className="text-[10px] text-gray-400 font-mono tracking-wider uppercase mb-1.5 flex items-center gap-1.5">
                <Layers size={10} className="text-brand-orange" />
                <span>Hover over rosters</span>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                {currentRoster.active.map((player) => (
                  <div key={player.id} className="flex items-center gap-1.5 bg-[#120e1f] px-2.5 py-1.5 rounded-lg border border-purple-950/50 hover:border-brand-pink hover:bg-purple-950/10 transition-colors">
                    <User size={11} className="text-gray-500" />
                    <span className="text-gray-200 font-medium truncate">{player.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Substitute/Coach players block */}
            {currentRoster.substitutes && currentRoster.substitutes.length > 0 && (
              <div>
                <div className="text-[10px] text-gray-400 font-mono tracking-wider uppercase mb-1.5 flex items-center gap-1.5">
                  <User size={10} className="text-brand-orange" />
                  <span>Player rosteres</span>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {currentRoster.substitutes.map((player) => (
                    <div key={player.id} className="flex items-center gap-1.5 bg-[#120e1f] px-2.5 py-1.5 rounded-lg border border-purple-950/50 hover:border-brand-pink hover:bg-purple-950/10 transition-colors">
                      <User size={11} className="text-brand-pink" />
                      <span className="text-gray-300 truncate">{player.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Opponent Linkage comparison in Footer Card */}
            {opponentTeam && (
              <div className="bg-[#120e1f] rounded-xl p-2 flex items-center justify-between border border-[#27213d] hover:border-brand-pink transition-colors">
                <div className="flex items-center gap-1.5">
                  <TeamLogo name={opponentTeam.logo} size={15} />
                  <span className="text-[10px] text-gray-400 font-medium font-mono truncate max-w-[130px]">{opponentTeam.name}</span>
                </div>
                <span className="text-[9px] text-[#ef4444] border border-[#ef4444]/30 px-1 bg-red-950/10 rounded uppercase font-bold tracking-widest font-mono">VS Matches</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Subordinate Slot component for Team representation inside bracket matchups
interface TeamSlotProps {
  team?: Team;
  score?: number;
  isWinner: boolean;
  onHover: (team: Team | null) => void;
  activeHoverId?: string;
}

function TeamSlot({ team, score, isWinner, onHover, activeHoverId }: TeamSlotProps) {
  if (!team) {
    return (
      <div className="flex justify-between items-center py-2 px-2 text-gray-600 bg-gray-900/10 border border-transparent rounded-lg text-xs font-mono select-none">
        <span>TBD Team</span>
        <span>-</span>
      </div>
    );
  }

  const isCurrentHovered = activeHoverId === team.id;

  return (
    <div 
      onMouseEnter={() => onHover(team)}
      className={`flex justify-between items-center py-1.5 px-2 rounded-lg border transition-all text-xs cursor-pointer select-none ${
        isCurrentHovered 
          ? 'bg-purple-950/30 border-brand-pink text-white shadow-[0_0_10px_rgba(236,72,153,0.25)]' 
          : 'bg-[#181329] border-[#29223c] text-gray-300 hover:bg-purple-950/10 hover:border-gray-600'
      }`}
    >
      <div className="flex items-center gap-2 overflow-hidden truncate">
        <TeamLogo name={team.logo} size={16} className="flex-shrink-0" />
        <span className="font-medium font-display truncate">{team.name}</span>
      </div>
      
      {score !== undefined && (
        <span className={`font-semibold font-mono text-xs ml-2 ${isWinner ? 'text-brand-pink font-bold' : 'text-gray-500'}`}>
          {score}
        </span>
      )}
    </div>
  );
}

function ChevronDownIcon() {
  return (
    <svg className="w-2.5 h-2.5 ml-1 select-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
    </svg>
  );
}
