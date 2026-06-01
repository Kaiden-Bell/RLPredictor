/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Sparkles, BarChart2, Star } from 'lucide-react';
import { PlayerRating, TeamForm } from '../types';
import TeamLogo from './TeamLogo';

interface StatsFooterProps {
  winProbability: {
    team1: { name: string; logo: string };
    team2: { name: string; logo: string };
    prob1: number;
    prob2: number;
  };
  playerRatings: PlayerRating[];
  teamForms: TeamForm[];
}

export default function StatsFooter({ winProbability, playerRatings, teamForms }: StatsFooterProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-shrink-0 select-none">
      {/* CARD 1: MATCH WIN PROBABILITY */}
      <div className="bg-app-surface rounded-2xl p-5 cyan-glow-border flex flex-col justify-between min-h-[145px] hover:shadow-[0_0_20px_rgba(34,211,238,0.25)] transition-all duration-300">
        <div>
          <h4 className="text-gray-200 font-display font-semibold text-xs uppercase tracking-wide">
            Match win probability
          </h4>
          <p className="text-[10px] text-gray-500 font-medium font-mono leading-none mt-1">
            Match win probability ratio
          </p>
        </div>

        <div className="flex items-baseline justify-between my-2">
          <div className="flex flex-col">
            <span className="text-3xl font-display font-extrabold text-brand-glow glow-text-cyan">
              {winProbability.prob1}%
            </span>
            <span className="text-[9px] text-gray-500 font-mono font-medium tracking-wider uppercase mt-0.5">
              {winProbability.team1.name}
            </span>
          </div>

          <span className="text-sm font-display font-bold text-gray-600 font-mono italic">
            VS
          </span>

          <div className="flex flex-col items-end">
            <span className="text-3xl font-display font-extrabold text-brand-orange glow-text-orange">
              {winProbability.prob2}%
            </span>
            <span className="text-[9px] text-gray-500 font-mono font-medium tracking-wider uppercase mt-0.5">
              {winProbability.team2.name}
            </span>
          </div>
        </div>

        {/* Custom Duel bar */}
        <div>
          <div className="h-2 w-full rounded-full bg-[#1e1933] overflow-hidden flex">
            <div 
              className="h-full bg-brand-glow shadow-[0_0_8px_#22d3ee]" 
              style={{ width: `${winProbability.prob1 / (winProbability.prob1 + winProbability.prob2) * 100}%` }}
            />
            <div 
              className="h-full bg-brand-orange shadow-[0_0_8px_#f97316]" 
              style={{ width: `${winProbability.prob2 / (winProbability.prob1 + winProbability.prob2) * 100}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[10px] text-gray-500 font-mono mt-1.5 leading-none">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* CARD 2: PLAYER PERFORMANCE RATING */}
      <div className="bg-app-surface rounded-2xl p-5 cyan-glow-border flex flex-col justify-between min-h-[145px] hover:shadow-[0_0_20px_rgba(34,211,238,0.25)] transition-all duration-300">
        <div>
          <h4 className="text-gray-200 font-display font-semibold text-xs uppercase tracking-wide">
            Player performance rating
          </h4>
          <p className="text-[10px] text-gray-500 font-medium font-mono leading-none mt-1">
            Player performance ratings
          </p>
        </div>

        <div className="flex flex-col gap-3 my-2">
          {playerRatings.map((player, idx) => {
            const isCyan = idx === 0;
            return (
              <div key={player.name} className="flex items-center justify-between gap-3 text-xs">
                {/* Name */}
                <span className="text-gray-300 font-medium font-display min-w-[70px] truncate">
                  {player.name}
                </span>

                {/* Bar */}
                <div className="flex-1 h-2 rounded-full bg-[#1e1933] overflow-hidden relative">
                  <div 
                    className={`h-full rounded-full ${
                      isCyan 
                        ? 'bg-brand-glow shadow-[0_0_8px_#22d3ee]' 
                        : 'bg-brand-orange shadow-[0_0_8px_#f97316]'
                    }`}
                    style={{ width: `${(player.rating / 10) * 100}%` }}
                  />
                </div>

                {/* Value */}
                <span className={`font-bold font-mono min-w-[45px] text-right inline-flex items-center justify-end gap-0.5 ${
                  isCyan ? 'text-brand-glow glow-text-cyan' : 'text-brand-orange glow-text-orange'
                }`}>
                  {player.rating.toFixed(2)}
                  <span className="text-[9px] opacity-75 font-normal">+</span>
                </span>
              </div>
            );
          })}
        </div>

        {/* Footer info text */}
        <div className="text-[10px] text-gray-500 font-mono text-center pt-0.5 border-t border-purple-950/20 leading-none">
          Live statistics updated by Gemini ML Rating
        </div>
      </div>

      {/* CARD 3: RECENT TEAM FORM */}
      <div className="bg-app-surface rounded-2xl p-5 cyan-glow-border flex flex-col justify-between min-h-[145px] hover:shadow-[0_0_20px_rgba(34,211,238,0.25)] transition-all duration-300">
        <div>
          <h4 className="text-gray-200 font-display font-semibold text-xs uppercase tracking-wide">
            Recent team form
          </h4>
          <p className="text-[10px] text-gray-500 font-medium font-mono leading-none mt-1">
            Recent team form ratight
          </p>
        </div>

        <div className="flex flex-col gap-3.5 my-2">
          {teamForms.map((teamForm) => (
            <div key={teamForm.teamName} className="flex items-center justify-between gap-2.5 text-xs">
              <div className="flex items-center gap-1.5 min-w-[70px]">
                <TeamLogo name={teamForm.logo || 'generic'} size={15} />
                <span className="text-gray-300 font-medium font-display truncate max-w-[65px]">
                  {teamForm.teamName}
                </span>
              </div>

              {/* Form Streaks */}
              <div className="flex items-center gap-1">
                {teamForm.form.map((outcome, idx) => {
                  const isWin = outcome === 'W';
                  return (
                    <div 
                      key={idx}
                      className={`w-5 h-5 rounded-full flex items-center justify-center font-bold text-[9px] font-mono border select-none transition-transform duration-200 hover:scale-110 ${
                        isWin
                          ? 'bg-cyan-950/20 text-brand-glow border-brand-glow/40 shadow-[0_0_8px_rgba(34,211,238,0.2)]'
                          : 'bg-red-950/15 text-brand-orange border-brand-orange/40 shadow-[0_0_8px_rgba(249,115,22,0.1)]'
                      }`}
                      title={isWin ? 'Victory' : 'Defeat'}
                    >
                      {outcome}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="text-[10px] text-gray-500 font-mono text-center pt-0.5 border-t border-purple-950/20 leading-none">
          Performance matrix over last 6 games
        </div>
      </div>
    </div>
  );
}
