/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Search, Users, Loader2, AlertCircle, CheckCircle, X } from 'lucide-react';

interface ManualMatchInputProps {
  onMatchCreated: (data: {
    team1: any;
    team2: any;
    rosters: Record<string, any>;
  }) => void;
  onClose: () => void;
}

export default function ManualMatchInput({ onMatchCreated, onClose }: ManualMatchInputProps) {
  const [team1Name, setTeam1Name] = useState('');
  const [team2Name, setTeam2Name] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!team1Name.trim() || !team2Name.trim()) {
      setError('Both team names are required.');
      return;
    }

    setError('');
    setIsLoading(true);
    setStatus('Looking up rosters...');

    try {
      const response = await fetch('/api/roster/lookup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          team1_name: team1Name.trim(),
          team2_name: team2Name.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Lookup failed with status ${response.status}`);
      }

      const data = await response.json();

      if (!data.team1 || !data.team2) {
        throw new Error('Could not resolve one or both teams.');
      }

      setStatus('Rosters loaded!');
      setTimeout(() => {
        onMatchCreated({
          team1: data.team1,
          team2: data.team2,
          rosters: data.rosters,
        });
      }, 500);
    } catch (err: any) {
      setError(err.message || 'Failed to look up rosters.');
      setIsLoading(false);
      setStatus('');
    }
  };

  return (
    <div className="fixed inset-0 bg-[#06040a]/85 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="w-full max-w-md bg-app-surface border border-app-border rounded-2xl p-6 shadow-2xl relative">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors cursor-pointer"
        >
          <X size={18} />
        </button>

        {/* Header */}
        <div className="flex items-center gap-2 mb-5">
          <div className="w-10 h-10 rounded-xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center text-brand-pink">
            <Users size={20} />
          </div>
          <div>
            <h3 className="font-display font-bold text-white text-base">Manual Matchup</h3>
            <p className="text-[10px] text-gray-500 font-mono uppercase tracking-wider">
              Enter team names to fetch rosters
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {/* Team 1 Input */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-mono text-gray-400 uppercase tracking-wider font-semibold">
              Team 1
            </label>
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                value={team1Name}
                onChange={(e) => { setTeam1Name(e.target.value); setError(''); }}
                placeholder="e.g. Karmine Corp"
                disabled={isLoading}
                className="w-full bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/50 text-sm text-gray-200 pl-9 pr-4 py-2.5 rounded-xl outline-none transition-all placeholder-gray-600 disabled:opacity-50"
              />
            </div>
          </div>

          {/* VS Divider */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-purple-950/40" />
            <span className="text-[10px] font-display font-bold text-gray-500 uppercase tracking-widest">vs</span>
            <div className="flex-1 h-px bg-purple-950/40" />
          </div>

          {/* Team 2 Input */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-mono text-gray-400 uppercase tracking-wider font-semibold">
              Team 2
            </label>
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                value={team2Name}
                onChange={(e) => { setTeam2Name(e.target.value); setError(''); }}
                placeholder="e.g. Team Vitality"
                disabled={isLoading}
                className="w-full bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/50 text-sm text-gray-200 pl-9 pr-4 py-2.5 rounded-xl outline-none transition-all placeholder-gray-600 disabled:opacity-50"
              />
            </div>
          </div>

          {/* Error display */}
          {error && (
            <div className="flex items-center gap-2 text-rose-400 text-xs bg-rose-950/15 border border-rose-500/20 px-3 py-2 rounded-xl">
              <AlertCircle size={14} className="flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Status display */}
          {status && !error && (
            <div className="flex items-center gap-2 text-emerald-400 text-xs bg-emerald-950/15 border border-emerald-500/20 px-3 py-2 rounded-xl">
              {isLoading ? (
                <Loader2 size={14} className="flex-shrink-0 animate-spin" />
              ) : (
                <CheckCircle size={14} className="flex-shrink-0" />
              )}
              <span>{status}</span>
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading || !team1Name.trim() || !team2Name.trim()}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-display font-bold py-3 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer shadow-[0_4px_20px_rgba(236,72,153,0.2)] active:scale-98"
          >
            {isLoading ? (
              <>
                <Loader2 size={15} className="animate-spin" />
                <span>Looking up rosters...</span>
              </>
            ) : (
              <>
                <Users size={15} />
                <span>Fetch Rosters & Create Match</span>
              </>
            )}
          </button>
        </form>

        {/* Helper text */}
        <p className="text-[10px] text-gray-600 mt-4 text-center leading-relaxed">
          Uses Liquipedia team pages to resolve active rosters. Cached results are returned instantly.
        </p>
      </div>
    </div>
  );
}
