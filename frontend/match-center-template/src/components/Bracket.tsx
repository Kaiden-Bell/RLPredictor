/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { User, Layers, Trophy, Grid3X3, Plus } from 'lucide-react';
import { BracketMatch, Team, TeamRoster } from '../types';
import TeamLogo from './TeamLogo';

interface BracketProps {
  bracketMatches: BracketMatch[];
  rosters: Record<string, TeamRoster>;
  onHoverTeam?: (team: Team | null) => void;
  activeHoverTeam: Team | null;
  selectedMatchId?: string | null;
  onSelectMatch?: (match: BracketMatch) => void;
  onManualMatchup?: () => void;
}

/** Group an array by a key function, preserving insertion order. */
function groupByOrdered<T>(items: T[], keyFn: (item: T) => string): { key: string; items: T[] }[] {
  const map = new Map<string, T[]>();
  for (const item of items) {
    const k = keyFn(item);
    if (!map.has(k)) map.set(k, []);
    map.get(k)!.push(item);
  }
  return Array.from(map.entries()).map(([key, items]) => ({ key, items }));
}

export default function Bracket({
  bracketMatches,
  rosters,
  onHoverTeam,
  activeHoverTeam,
  selectedMatchId,
  onSelectMatch,
  onManualMatchup,
}: BracketProps) {
  const [selectedRosterTeam, setSelectedRosterTeam] = useState<Team | null>(null);
  const [hoveredMatchId, setHoveredMatchId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Derive section/round groupings dynamically from the match data
  const bracketLayout = useMemo(() => {
    // Group by section first (e.g. "Playoffs", "Group A", "Group B")
    const sections = groupByOrdered(bracketMatches, (m) => m.section || 'Bracket');

    return sections.map((sec) => {
      // Within each section, group by round label preserving scraper order
      const rounds = groupByOrdered(sec.items, (m) => m.round || `Round ${m.roundIndex}`);
      return { section: sec.key, rounds };
    });
  }, [bracketMatches]);

  const currentRoster = selectedRosterTeam ? rosters[selectedRosterTeam.id] : null;

  // Find opponent for roster popover
  const getComparativeOpponent = (team: Team): Team | null => {
    const match = bracketMatches.find(
      (m) => m.team1?.id === team.id || m.team2?.id === team.id
    );
    if (!match) return null;
    return match.team1?.id === team.id ? match.team2 || null : match.team1 || null;
  };

  const opponentTeam = selectedRosterTeam ? getComparativeOpponent(selectedRosterTeam) : null;

  // Determine if this is a group stage (multiple sections) or playoff (single section)
  const isGroupStage = bracketLayout.length > 1;
  const totalRoundCount = bracketLayout.reduce((sum, s) => sum + s.rounds.length, 0);

  return (
    <div className="bg-app-surface border border-app-border rounded-2xl h-full flex flex-col overflow-hidden relative select-none">
      {/* Bracket Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-3 flex-shrink-0 z-10">
        <div className="flex items-center gap-1.5">
          {isGroupStage ? (
            <Grid3X3 size={18} className="text-brand-pink" />
          ) : (
            <Layers size={18} className="text-brand-pink" />
          )}
          <h2 className="font-display font-semibold text-gray-200 text-sm md:text-base tracking-wide uppercase">
            {isGroupStage ? 'Group Stage Brackets' : 'Tournament Bracket'}
          </h2>
        </div>

        <div className="flex items-center gap-2.5">
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

          {onManualMatchup && (
            <button
              onClick={onManualMatchup}
              className="flex items-center gap-1.5 bg-purple-950/20 hover:bg-purple-900/30 text-gray-400 hover:text-white border border-app-border px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors"
              title="Manual Matchup Input"
            >
              <Plus size={13} />
              <span>Custom</span>
            </button>
          )}
        </div>
      </div>

      {/* Round count badge */}
      <div className="px-6 pb-3 flex items-center gap-2">
        <span className="text-[9px] font-mono text-gray-500 bg-[#110e1a] border border-purple-950/30 px-2 py-0.5 rounded">
          {bracketMatches.length} matches across {totalRoundCount} rounds
        </span>
        {bracketLayout.length > 1 && (
          <span className="text-[9px] font-mono text-brand-pink bg-pink-950/15 border border-brand-pink/20 px-2 py-0.5 rounded">
            {bracketLayout.length} groups
          </span>
        )}
      </div>

      {/* Bracket Content — scrollable */}
      <div className="flex-1 overflow-y-auto overflow-x-auto px-4 pb-4" ref={scrollRef}>
        {bracketLayout.map((sec, secIdx) => (
          <div key={sec.section} className={secIdx > 0 ? 'mt-6' : ''}>
            {/* Section Header (only show if multiple sections) */}
            {bracketLayout.length > 1 && (
              <div className="flex items-center gap-2 mb-3 px-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand-pink" />
                <h3 className="text-[11px] font-display font-bold text-gray-300 uppercase tracking-wider">
                  {sec.section}
                </h3>
                <div className="flex-1 h-px bg-purple-950/40" />
              </div>
            )}

            {/* Rounds as horizontal columns */}
            <div className="flex gap-3 items-stretch min-w-max">
              {sec.rounds.map((round, roundIdx) => {
                const isLastRound = roundIdx === sec.rounds.length - 1;
                const isFirstRound = roundIdx === 0;
                const matchCount = round.items.length;

                return (
                  <div
                    key={`${sec.section}-${round.key}`}
                    className="flex flex-col flex-shrink-0"
                    style={{ width: matchCount <= 1 ? 210 : 210 }}
                  >
                    {/* Round header */}
                    <div className={`text-[9px] font-mono tracking-widest uppercase mb-2 text-center px-1 py-1 rounded-lg border ${
                      isLastRound
                        ? 'text-brand-pink bg-pink-950/10 border-brand-pink/20 font-bold'
                        : 'text-gray-500 bg-[#0e0b18] border-purple-950/20'
                    }`}>
                      {round.key}
                    </div>

                    {/* Match cards in this round */}
                    <div className={`flex-1 flex flex-col gap-2.5 ${
                      matchCount <= 2 ? 'justify-around' : 'justify-start'
                    }`}>
                      {round.items.map((match) => (
                        <MatchCard
                          key={match.id}
                          match={match}
                          isSelected={selectedMatchId === match.id}
                          isHovered={hoveredMatchId === match.id}
                          isLastRound={isLastRound}
                          onSelect={() => onSelectMatch?.(match)}
                          onHoverStart={() => setHoveredMatchId(match.id)}
                          onHoverEnd={() => setHoveredMatchId(null)}
                          onTeamHover={(t) => {
                            onHoverTeam?.(t);
                            if (t) setSelectedRosterTeam(t);
                          }}
                          activeHoverId={activeHoverTeam?.id}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}

              {/* Trophy/Winner column for last section */}
              {!isGroupStage && (
                <div className="flex flex-col justify-center items-center w-16 flex-shrink-0">
                  <div className="w-12 h-12 rounded-xl bg-purple-950/20 border border-purple-900/30 flex items-center justify-center text-gray-600 hover:border-brand-pink hover:text-brand-pink transition-all">
                    <Trophy size={20} />
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ROSTER HOVER POPUP OVERLAY */}
      {selectedRosterTeam && currentRoster && (
        <div
          id="roster-overlay-popover"
          className="absolute top-14 right-4 w-72 bg-[#171328] border border-brand-orange/60 rounded-2xl p-4 shadow-[0_15px_30px_rgba(0,0,0,0.8),0_0_15px_rgba(249,115,22,0.15)] z-40 animate-fade-in text-left flex flex-col gap-3.5 select-none"
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
              <span>Active Roster</span>
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

          {/* Substitute players block */}
          {currentRoster.substitutes && currentRoster.substitutes.length > 0 && (
            <div>
              <div className="text-[10px] text-gray-400 font-mono tracking-wider uppercase mb-1.5 flex items-center gap-1.5">
                <User size={10} className="text-brand-orange" />
                <span>Substitutes</span>
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

          {/* Opponent Linkage */}
          {opponentTeam && (
            <div className="bg-[#120e1f] rounded-xl p-2 flex items-center justify-between border border-[#27213d] hover:border-brand-pink transition-colors">
              <div className="flex items-center gap-1.5">
                <TeamLogo name={opponentTeam.logo} size={15} />
                <span className="text-[10px] text-gray-400 font-medium font-mono truncate max-w-[130px]">{opponentTeam.name}</span>
              </div>
              <span className="text-[9px] text-[#ef4444] border border-[#ef4444]/30 px-1 bg-red-950/10 rounded uppercase font-bold tracking-widest font-mono">VS</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ─── Individual Match Card ───────────────────────────────────────────

interface MatchCardProps {
  match: BracketMatch;
  isSelected: boolean;
  isHovered: boolean;
  isLastRound: boolean;
  onSelect: () => void;
  onHoverStart: () => void;
  onHoverEnd: () => void;
  onTeamHover: (team: Team | null) => void;
  activeHoverId?: string;
}

function MatchCard({
  match,
  isSelected,
  isHovered,
  isLastRound,
  onSelect,
  onHoverStart,
  onHoverEnd,
  onTeamHover,
  activeHoverId,
}: MatchCardProps & { key?: React.Key }) {
  return (
    <div
      onClick={onSelect}
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      className={`flex flex-col gap-0.5 rounded-xl border transition-all duration-200 cursor-pointer bg-[#120e1d]/90 overflow-hidden ${
        isSelected
          ? 'border-brand-pink shadow-[0_0_15px_rgba(236,72,153,0.35)] bg-pink-950/15'
          : isHovered
            ? 'border-brand-pink/50 shadow-[0_0_12px_rgba(236,72,153,0.15)] bg-purple-950/10'
            : isLastRound
              ? 'border-brand-orange/50 shadow-[0_0_8px_rgba(249,115,22,0.08)] hover:border-brand-pink/30'
              : 'border-[#27213d] hover:border-brand-pink/30'
      }`}
    >
      {/* Best-of badge */}
      {match.bestOf && (
        <div className="flex items-center justify-between px-2 pt-1.5">
          <span className="text-[8px] font-mono text-gray-600 uppercase tracking-wider">
            Bo{match.bestOf}
          </span>
          {match.status === 'completed' && (
            <span className="text-[8px] font-mono text-emerald-500/70 uppercase tracking-wider">✓</span>
          )}
          {match.status === 'live' && (
            <span className="text-[8px] font-mono text-red-400 uppercase tracking-wider animate-pulse">LIVE</span>
          )}
        </div>
      )}

      {/* Team slots */}
      <div className="px-1.5 pb-1.5 flex flex-col gap-0.5">
        <TeamSlot
          team={match.team1}
          score={match.score1}
          isWinner={match.winnerId === match.team1?.id}
          onHover={onTeamHover}
          activeHoverId={activeHoverId}
        />
        <div className="h-px bg-[#1e1833] mx-1" />
        <TeamSlot
          team={match.team2}
          score={match.score2}
          isWinner={match.winnerId === match.team2?.id}
          onHover={onTeamHover}
          activeHoverId={activeHoverId}
        />
      </div>
    </div>
  );
}


// ─── Team Slot within a Match Card ───────────────────────────────────

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
      <div className="flex justify-between items-center py-1.5 px-2 text-gray-600 bg-gray-900/10 border border-transparent rounded-lg text-xs font-mono select-none">
        <span>TBD</span>
        <span>-</span>
      </div>
    );
  }

  const isCurrentHovered = activeHoverId === team.id;

  return (
    <div
      onMouseEnter={() => onHover(team)}
      className={`flex justify-between items-center py-1.5 px-2 rounded-lg border transition-all text-xs cursor-pointer select-none ${
        isWinner
          ? 'bg-[#181329] border-emerald-900/30 text-white'
          : isCurrentHovered
            ? 'bg-purple-950/30 border-brand-pink text-white shadow-[0_0_10px_rgba(236,72,153,0.25)]'
            : 'bg-[#181329] border-[#29223c] text-gray-300 hover:bg-purple-950/10 hover:border-gray-600'
      }`}
    >
      <div className="flex items-center gap-1.5 overflow-hidden truncate">
        <TeamLogo name={team.logo} size={14} className="flex-shrink-0" />
        <span className={`font-medium font-display truncate text-[11px] ${isWinner ? 'font-bold' : ''}`}>
          {team.shortName || team.name}
        </span>
      </div>

      {score !== undefined && (
        <span className={`font-semibold font-mono text-xs ml-2 ${
          isWinner ? 'text-emerald-400 font-bold' : 'text-gray-500'
        }`}>
          {score}
        </span>
      )}
    </div>
  );
}
