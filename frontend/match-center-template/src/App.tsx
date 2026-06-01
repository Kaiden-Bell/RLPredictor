/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { TournamentData, Team, BracketMatch, Match } from './types';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Onboarding from './components/Onboarding';
import Bracket from './components/Bracket';
import MatchesList from './components/MatchesList';
import StatsFooter from './components/StatsFooter';
import { generatePlaceholderData } from './utils/placeholderGenerator';
import { Sparkles } from 'lucide-react';

export default function App() {
  const [tournamentData, setTournamentData] = useState<TournamentData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState('');
  const [activeHoverTeam, setActiveHoverTeam] = useState<Team | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<BracketMatch | Match | null>(null);

  const handleStartAnalysis = (url: string) => {
    setIsLoading(true);
    setApiError('');
    setSelectedMatch(null);
    
    // Transition to the placeholder structure/layout for the inputted tournament specific url cleanly
    setTimeout(() => {
      try {
        const data = generatePlaceholderData(url);
        setTournamentData(data);
      } catch (err: any) {
        console.error('Failed setting up tournament specific structure:', err);
        setApiError('Failed to compile local layout structure.');
      } finally {
        setIsLoading(false);
      }
    }, 1200); // Sleek simulated parse duration to display state loading transition
  };

  const handleFailsafeFallback = (url: string) => {
    // Generate default local mockup matching image if API fails
    setTournamentData({
      name: "RLPredictor - Rocket League Major Championship",
      url: url,
      game: "Rocket League",
      bracketMatches: [
        { id: "q1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" }, team2: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality" }, score1: 3, score2: 2, winnerId: "g2", status: "completed", roundIndex: 0 },
        { id: "q2", matchIndex: 1, team1: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, score1: 3, score2: 1, winnerId: "vit", status: "completed", roundIndex: 0 },
        { id: "q3", matchIndex: 2, team1: { id: "swn", name: "Swnder Esports", shortName: "Swnder", logo: "Swnder" }, team2: { id: "nop", name: "Noppes Esports", shortName: "Noppes", logo: "Noppes" }, score1: 1, score2: 3, winnerId: "nop", status: "completed", roundIndex: 0 },
        { id: "q4", matchIndex: 3, team1: { id: "sin", name: "Sinzline Gaming", shortName: "Sinzline", logo: "Sinzline" }, team2: { id: "t5w", name: "Team 5WS", shortName: "Team 5WS", logo: "Team 5WS" }, score1: 1, score2: 3, winnerId: "t5w", status: "completed", roundIndex: 0 },
        { id: "s1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" }, team2: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality" }, score1: 4, score2: 3, winnerId: "g2", status: "completed", roundIndex: 1 },
        { id: "s2", matchIndex: 1, team1: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, team2: { id: "nop", name: "Noppes Esports", shortName: "Noppes", logo: "Noppes" }, score1: 4, score2: 0, winnerId: "kc", status: "completed", roundIndex: 1 },
        { id: "f1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, status: "live", roundIndex: 2 }
      ],
      upcomingMatches: [
        { id: "u1", team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" }, team2: { id: "swn", name: "Swnder Esports", shortName: "Swnder", logo: "Swnder" }, status: "upcoming", time: "19:00", date: "12:00" },
        { id: "u2", team1: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality" }, team2: { id: "aud", name: "Audacity Team", shortName: "Audacity", logo: "Audacity" }, status: "completed", score1: 3, score2: 1, time: "19:30", date: "12:00" }
      ],
      finishedMatches: [
        { id: "fi1", team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, status: "finished", time: "13:00", date: "17:00" }
      ],
      completedMatches: [
        { id: "c1", team1: { id: "sol", name: "Solary Esports", shortName: "Solary", logo: "Solary" }, team2: { id: "aud", name: "Audacity Team", shortName: "Audacity", logo: "Audacity" }, status: "completed", score1: 1, score2: 1, time: "13:00", date: "13:00" },
        { id: "c2", team1: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, team2: { id: "swn", name: "Swnder Esports", shortName: "Swnder", logo: "Swnder" }, status: "completed", score1: 0, score2: 2, time: "13:00", date: "10:00" },
        { id: "c3", team1: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" }, status: "completed", score1: 2, score2: 2, time: "13:00", date: "9:30" }
      ],
      winProbability: {
        team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2" },
        team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine" },
        prob1: 80,
        prob2: 68
      },
      playerRatings: [
        { name: "Jounal", rating: 7.86 },
        { name: "Kamerian", rating: 6.70 }
      ],
      teamForms: [
        { teamName: "Team", logo: "Vitality", form: ["L", "W", "W", "L", "L", "L"] },
        { teamName: "Karmine", logo: "Karmine", form: ["L", "L", "W", "W", "W", "L"] }
      ],
      rosters: {
        g2: {
          teamId: "g2",
          active: [
            { id: "g1", name: "Alorin", role: "Active Roster" },
            { id: "g2", name: "Binder", role: "Active Roster" },
            { id: "g3", name: "Data", role: "Active Roster" },
            { id: "g4", name: "Sugan", role: "Active Roster" },
            { id: "g5", name: "Rauksl", role: "Active Roster" },
            { id: "g6", name: "Markeen", role: "Active Roster" }
          ],
          substitutes: [
            { id: "g7", name: "Porth", role: "Bench/Reserve" },
            { id: "g8", name: "Evinik", role: "Bench/Reserve" },
            { id: "g9", name: "Nersoc", role: "Bench/Reserve" }
          ]
        },
        vit: {
          teamId: "vit",
          active: [
            { id: "v1", name: "Zen", role: "Active Roster" },
            { id: "v2", name: "Alpha54", role: "Active Roster" },
            { id: "v3", name: "Radosin", role: "Active Roster" },
            { id: "v6", name: "ExoTiiK", role: "Active Roster" }
          ],
          substitutes: [
            { id: "v8", name: "Kaydop", role: "Bench" }
          ]
        },
        kc: {
          teamId: "kc",
          active: [
            { id: "k1", name: "Vatira", role: "Active Roster" },
            { id: "k2", name: "Atow.", role: "Active Roster" },
            { id: "k3", name: "Rise.", role: "Active Roster" }
          ],
          substitutes: [
            { id: "k7", name: "Itachi", role: "Bench" },
            { id: "k8", name: "Chausette", role: "Bench" }
          ]
        }
      }
    });
  };

  const handleReset = () => {
    setTournamentData(null);
    setApiError('');
    setSelectedMatch(null);
  };

  // Dynamic stats calculation for clicked matchups
  const getSelectedMatchStats = () => {
    if (!selectedMatch || !tournamentData) return null;

    const t1 = selectedMatch.team1;
    const t2 = selectedMatch.team2;

    if (!t1 || !t2) return null;

    // Stable deterministic pseudo-metrics based on character codes of name strings
    const code1 = t1.name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    const code2 = t2.name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);

    const prob1 = 48 + (code1 % 25); // 48% - 72%
    const prob2 = 100 - prob1;

    // Grab first real roster player name from data, fallback elegantly if roster is missing
    const p1 = tournamentData.rosters[t1.id]?.active?.[0]?.name || `${t1.shortName} MVP`;
    const p2 = tournamentData.rosters[t2.id]?.active?.[0]?.name || `${t2.shortName} MVP`;

    const ratingVal1 = 6.4 + ((code1 * 3) % 21) / 10; // 6.4 - 8.4
    const ratingVal2 = 6.2 + ((code2 * 7) % 19) / 10; // 6.2 - 8.0

    const playerRatings = [
      { name: p1, rating: Number(ratingVal1.toFixed(2)) },
      { name: p2, rating: Number(ratingVal2.toFixed(2)) }
    ];

    // Build stable historic form
    const formsArr: ('W' | 'L')[][] = [
      ['W', 'W', 'L', 'W', 'W', 'L'],
      ['W', 'L', 'W', 'W', 'L', 'W'],
      ['L', 'W', 'W', 'L', 'W', 'W'],
      ['W', 'W', 'W', 'L', 'L', 'W']
    ];
    const form1 = formsArr[code1 % formsArr.length];
    const form2 = formsArr[code2 % formsArr.length];

    const teamForms = [
      { teamName: t1.shortName, logo: t1.logo, form: form1 },
      { teamName: t2.shortName, logo: t2.logo, form: form2 }
    ];

    return {
      winProbability: {
        team1: t1,
        team2: t2,
        prob1,
        prob2
      },
      playerRatings,
      teamForms
    };
  };

  const hasData = tournamentData !== null;

  return (
    <div className="min-h-screen bg-[#06040a] text-[#f3f4f6]" id="app-viewport">
      {/* 1. Header Typography title blocks (Outside App client wrapper) */}
      <div className="flex flex-col items-center justify-center pt-8 pb-5 text-center select-none">
        <h1 className="font-display font-extrabold text-white text-3xl md:text-[38px] tracking-tight leading-none uppercase">
          Match Center
        </h1>
        <p className="font-sans font-normal text-xs md:text-sm text-gray-500 tracking-wider mt-1.5 uppercase">
          AI-powered esports platform UI design
        </p>
      </div>

      {/* 2. Mockup Frame Box container representing gaming client */}
      <div className="max-w-7xl mx-auto px-4 pb-12">
        <div className="flex border border-gray-900 rounded-[28px] bg-app-bg shadow-[0_25px_60px_rgba(0,0,0,0.85)] h-[840px] overflow-hidden relative">
          
          {/* Left panel control sidebar */}
          <Sidebar onReset={handleReset} hasData={hasData} />

          {/* Right main viewing context */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Context Header bar */}
            <Header 
              currentUrl={tournamentData?.url} 
              onSearchUrl={handleStartAnalysis} 
              isLoading={isLoading} 
            />

            {/* Inner frame context viewports */}
            <div className="flex-1 overflow-y-auto bg-app-bg">
              {!hasData ? (
                /* 2a. Welcome onboarding input stage */
                <Onboarding onStartAnalysis={handleStartAnalysis} isLoading={isLoading} />
              ) : (
                /* 2b. Fully structured Match Center Dashboard */
                <main className="p-6 md:p-8 flex flex-col gap-6 animate-fade-in">
                  
                  {/* Page sub-banner: Active Scraped Info */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#130f21] border border-brand-pink/20 rounded-xl px-5 py-3 shadow-[0_0_15px_rgba(147,51,234,0.05)] select-none">
                    <div className="flex flex-col">
                      <span className="text-[10px] text-brand-pink font-mono tracking-widest font-bold uppercase">Connected Tournament Source</span>
                      <span className="text-[15px] font-display font-bold text-gray-100 mt-0.5 leading-snug">
                        {tournamentData.name}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-2 py-1 bg-cyan-950/20 text-brand-glow border border-brand-glow/30 rounded font-mono font-medium lowercase">
                        {tournamentData.game} platform
                      </span>
                      <button 
                        onClick={handleReset}
                        className="text-xs text-gray-400 hover:text-white border border-gray-800 bg-app-surface hover:bg-app-surface/80 px-3 py-1 pb-1.5 rounded-lg font-medium transition-colors"
                      >
                        Disconnect Source URL
                      </button>
                    </div>
                  </div>

                  {apiError && (
                    <div className="bg-rose-950/20 border border-rose-500/25 text-rose-400 text-xs px-4 py-2.5 rounded-xl flex items-center select-none">
                      {apiError}
                    </div>
                  )}

                  {/* Main Grid: Brackets + Upcoming Matches Sidebars */}
                  <div className="grid grid-cols-1 xl:grid-cols-10 gap-6">
                    {/* Active Brackets quadrant */}
                    <div className="xl:col-span-7 h-[530px]">
                      <Bracket 
                        bracketMatches={tournamentData.bracketMatches} 
                        rosters={tournamentData.rosters}
                        activeHoverTeam={activeHoverTeam}
                        onHoverTeam={setActiveHoverTeam}
                        selectedMatchId={selectedMatch?.id}
                        onSelectMatch={(match) => setSelectedMatch(match)}
                      />
                    </div>

                    {/* Upcoming sidebar panel quadrant */}
                    <div className="xl:col-span-3 h-[530px]">
                      <MatchesList 
                        upcomingMatches={tournamentData.upcomingMatches} 
                        finishedMatches={tournamentData.finishedMatches} 
                        completedMatches={tournamentData.completedMatches}
                        activeHoverTeam={activeHoverTeam}
                        onHoverTeam={setActiveHoverTeam}
                        selectedMatchId={selectedMatch?.id}
                        onSelectMatch={(match) => setSelectedMatch(match)}
                      />
                    </div>
                  </div>

                  {/* Bottom Stats Row panels */}
                  {selectedMatch ? (
                    (() => {
                      const activeMatchStats = getSelectedMatchStats();
                      return activeMatchStats ? (
                        <StatsFooter 
                          winProbability={activeMatchStats.winProbability}
                          playerRatings={activeMatchStats.playerRatings}
                          teamForms={activeMatchStats.teamForms}
                        />
                      ) : (
                        /* If selectedMatch exists but lacks teams (e.g. pending TBD slots) */
                        <div className="bg-[#120f21]/80 border border-purple-950 rounded-2xl p-6 py-10 flex flex-col items-center justify-center text-center select-none gap-4 animate-fade-in">
                          <div className="w-12 h-12 rounded-xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center text-brand-pink">
                            <Sparkles size={24} className="animate-pulse" />
                          </div>
                          <div className="space-y-1.5 max-w-md">
                            <h4 className="font-display font-semibold text-gray-200 text-sm uppercase">
                              Sufficient Seeding Required
                            </h4>
                            <p className="text-xs text-gray-500 leading-relaxed font-sans">
                              This tournament card is waiting for preceding rounds to conclude. Click an active, fully seeded matchup in the bracket to generate matrix metrics.
                            </p>
                          </div>
                        </div>
                      );
                    })()
                  ) : (
                    /* Initial Landing: Stats are hidden until bracket node interaction */
                    <div className="bg-[#0e0a1b]/60 border border-[#231b34] rounded-2xl p-6 py-10 flex flex-col items-center justify-center text-center select-none gap-4 animate-fade-in">
                      <div className="w-12 h-12 rounded-xl bg-purple-950/40 border border-[#ab47bc]/20 flex items-center justify-center text-brand-pink relative">
                        <Sparkles size={24} className="animate-pulse text-brand-pink" />
                        <div className="absolute inset-0 rounded-xl border border-brand-pink/30 animate-ping opacity-25" />
                      </div>
                      <div className="space-y-1.5 max-w-lg">
                        <h4 className="font-display font-extrabold text-[#eae8ef] text-sm uppercase tracking-wide">
                          Predictive Analysis Cockpit
                        </h4>
                        <p className="text-xs text-gray-500 leading-relaxed font-sans">
                          Click any active matchup row in the <span className="text-gray-300 font-semibold font-display">Tournament Bracket diagram</span> or sidebar list above to load live win probabilities, player performance ratings, and recent team form curves.
                        </p>
                      </div>
                    </div>
                  )}
                  
                </main>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
