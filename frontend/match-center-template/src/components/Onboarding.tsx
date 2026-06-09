/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Sparkles, Trophy, ArrowRight, Compass, ShieldAlert, Cpu, Layers, Grid3X3 } from 'lucide-react';

interface OnboardingProps {
  onStartAnalysis: (url: string, sections?: string[]) => void;
  isLoading: boolean;
}

export default function Onboarding({ onStartAnalysis, isLoading }: OnboardingProps) {
  const [showPrompt, setShowPrompt] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [errorText, setErrorText] = useState('');
  const [loadingStep, setLoadingStep] = useState(0);
  const [scrapeMode, setScrapeMode] = useState<'playoffs' | 'group'>('playoffs');

  const quickExamples = [
    {
      title: 'RLCS 2026 Boston Major (NA Open 3)',
      game: 'Rocket League (Default)',
      url: 'https://liquipedia.net/rocketleague/Rocket_League_Championship_Series/2026/Boston_Major/North_America/Open_3',
      badgeColor: 'border-yellow-500/30 text-yellow-400 bg-yellow-950/10'
    }
  ];

  // Loading process visual messages
  React.useEffect(() => {
    if (!isLoading) {
      setLoadingStep(0);
      return;
    }
    const interval = setInterval(() => {
      setLoadingStep((prev) => (prev < 3 ? prev + 1 : prev));
    }, 1500);
    return () => clearInterval(interval);
  }, [isLoading]);

  const loadingMessages = [
    'Initializing server-side Gemini live session...',
    `Scraping ${scrapeMode === 'playoffs' ? 'Playoff bracket' : 'Group Stage bracket'} nodes from Liquipedia...`,
    'Parsing team rosters, calculating individual player power ratings...',
    'Composing match win probabilities and form multipliers...',
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!urlInput.trim()) {
      setErrorText('Please paste a tournament URL first to begin.');
      return;
    }
    if (!urlInput.toLowerCase().startsWith('http://') && !urlInput.toLowerCase().startsWith('https://')) {
      setErrorText('Please enter a valid URL beginning with http:// or https://');
      return;
    }
    setErrorText('');
    const sections = scrapeMode === 'playoffs' ? ['Playoffs'] : ['Group'];
    onStartAnalysis(urlInput.trim(), sections);
  };

  const handleQuickClick = (url: string) => {
    setUrlInput(url);
    setErrorText('');
    const sections = scrapeMode === 'playoffs' ? ['Playoffs'] : ['Group'];
    onStartAnalysis(url, sections);
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 relative overflow-hidden select-none">
      {/* Decorative background glows */}
      <div className="absolute top-[30%] left-[50%] -translate-x-[50%] -translate-y-[50%] w-[500px] h-[500px] bg-purple-950/15 rounded-full blur-[120px] pointer-events-none z-0" />
      <div className="absolute bottom-[10%] left-[20%] w-[300px] h-[300px] bg-cyan-950/10 rounded-full blur-[90px] pointer-events-none z-0" />

      <div className="w-full max-w-xl bg-app-surface border border-app-border rounded-3xl p-8 shadow-2xl relative z-10 text-center flex flex-col gap-8 transition-transform duration-300">
        
        {/* Animated Loading screen */}
        {isLoading ? (
          <div className="py-12 flex flex-col items-center justify-center gap-6 animate-pulse">
            {/* Spinning vector halo */}
            <div className="relative flex items-center justify-center">
              <div className="w-16 h-16 rounded-full border-2 border-brand-pink/20 border-t-brand-pink animate-spin" />
              <Cpu className="absolute text-brand-pink" size={24} />
            </div>

            <div className="space-y-2 maxw-sm mx-auto">
              <h3 className="font-display font-bold text-gray-200 text-lg tracking-tight">
                Parsing Tournament URL
              </h3>
              <p className="text-sm text-gray-400 font-medium px-4 min-h-[40px] flex items-center justify-center font-mono py-1">
                {loadingMessages[loadingStep]}
              </p>
            </div>

            <div className="w-full bg-[#1e1933] h-1.5 rounded-full overflow-hidden max-w-[280px]">
              <div 
                className="h-full bg-brand-pink transition-all duration-500 ease-out shadow-[0_0_8px_#ec4899]" 
                style={{ width: `${((loadingStep + 1) / 4) * 100}%` }}
              />
            </div>
          </div>
        ) : !showPrompt ? (
          /* Landing Screen: Initially, display a "Get Started" button */
          <div className="py-6 flex flex-col items-center gap-6 animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center text-brand-pink relative">
              <Compass size={32} className="animate-pulse" />
              <div className="absolute inset-0 rounded-2xl border border-brand-pink/40 animate-ping opacity-25" />
            </div>

            <div className="space-y-2">
              <h2 className="font-display font-extrabold text-2xl tracking-tight text-white uppercase">
                Match Center Console
              </h2>
              <p className="text-sm text-gray-400 max-w-sm mx-auto leading-relaxed font-sans">
                A highly polished virtual dashboard for scraping live brackets, tracking historical results, and calculating real-time match predictions.
              </p>
            </div>

            <button
              id="get-started-landing"
              onClick={() => setShowPrompt(true)}
              className="w-full max-w-xs bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-display font-bold py-4 px-8 rounded-2xl transition-all duration-300 flex items-center justify-center gap-2 group cursor-pointer shadow-[0_4px_25px_rgba(236,72,153,0.4)] hover:shadow-[0_4px_35px_rgba(236,72,153,0.6)] active:scale-98 hover:-translate-y-0.5"
            >
              <span>Get Started</span>
              <ArrowRight size={18} className="transition-transform duration-200 group-hover:translate-x-1" />
            </button>
          </div>
        ) : (
          /* Input and trigger screen */
          <>
            {/* Heading and badge icons */}
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-purple-950/30 border border-purple-500/20 flex items-center justify-center text-brand-pink">
                <Trophy size={28} className="animate-bounce" />
              </div>

              <div className="space-y-2 mt-2">
                <h2 className="font-display font-extrabold text-2xl tracking-tight text-white">
                  Match Center Engine
                </h2>
                <p className="text-sm text-gray-400 max-w-md mx-auto leading-relaxed">
                  Provide a tournament URL below to scan brackets, reconstruct roster players, and calculate AI win statistics.
                </p>
              </div>
            </div>

            {/* ── Bracket Type Toggle ── */}
            <div className="flex flex-col gap-2.5">
              <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest font-semibold text-left">
                Bracket Section to Scrape
              </span>
              <div className="grid grid-cols-2 gap-2 bg-[#110e1a] p-1.5 rounded-xl border border-purple-950/30">
                <button
                  type="button"
                  onClick={() => setScrapeMode('playoffs')}
                  className={`flex items-center justify-center gap-2 py-2.5 px-3 text-xs font-display font-bold rounded-lg transition-all cursor-pointer ${
                    scrapeMode === 'playoffs'
                      ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-[0_2px_12px_rgba(236,72,153,0.25)]'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Trophy size={14} />
                  <span>Playoffs</span>
                </button>
                <button
                  type="button"
                  onClick={() => setScrapeMode('group')}
                  className={`flex items-center justify-center gap-2 py-2.5 px-3 text-xs font-display font-bold rounded-lg transition-all cursor-pointer ${
                    scrapeMode === 'group'
                      ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-[0_2px_12px_rgba(236,72,153,0.25)]'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Grid3X3 size={14} />
                  <span>Group Stage</span>
                </button>
              </div>
              <p className="text-[10px] text-gray-600 font-sans text-left leading-relaxed">
                {scrapeMode === 'playoffs'
                  ? 'Scrapes the elimination bracket (Quarterfinals → Semifinals → Grand Final). Best for predicting late-stage matchups.'
                  : 'Scrapes group stage brackets (GSL double-elimination pools). Best for round-robin and early-stage analysis.'}
              </p>
            </div>

            {/* URL Input Form */}
            <form onSubmit={handleSubmit} className="flex flex-col gap-3.5">
              <div className="relative">
                <input
                  id="onboarding-url-input"
                  type="text"
                  value={urlInput}
                  onChange={(e) => { setUrlInput(e.target.value); setErrorText(''); }}
                  placeholder="Paste Liquipedia tournament URL..."
                  className="w-full bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/50 text-sm text-gray-200 px-4 py-3.5 rounded-2xl outline-none transition-all placeholder-gray-500 shadow-inner"
                />
              </div>

              {errorText && (
                <div className="flex items-center gap-2 text-rose-500 text-xs text-left bg-rose-950/15 border border-rose-500/20 px-3.5 py-2.5 rounded-xl">
                  <ShieldAlert size={14} className="flex-shrink-0" />
                  <span>{errorText}</span>
                </div>
              )}

              <button
                type="submit"
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-display font-bold py-3.5 px-6 rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 group cursor-pointer shadow-[0_4px_20px_rgba(236,72,153,0.3)] active:scale-98"
              >
                <span>Scrape {scrapeMode === 'playoffs' ? 'Playoffs' : 'Group Stage'}</span>
                <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </form>

            <div className="relative">
              <div className="absolute inset-0 flex items-center" aria-hidden="true">
                <div className="w-full border-t border-purple-950/40"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-app-surface px-3 text-gray-500 font-mono tracking-widest">Or Click a Quick Example</span>
              </div>
            </div>

            {/* Quick Click Demo presets list */}
            <div className="flex flex-col gap-2.5">
              {quickExamples.map((ex) => (
                <div
                  key={ex.title}
                  onClick={() => handleQuickClick(ex.url)}
                  className="bg-[#110e1a]/80 hover:bg-purple-950/10 border border-purple-950 hover:border-brand-pink/40 p-3.5 rounded-xl flex items-center justify-between text-left cursor-pointer group transition-all"
                >
                  <div className="flex flex-col gap-1 overflow-hidden pr-2">
                    <span className="font-display font-bold text-xs text-gray-200 group-hover:text-white transition-colors">
                      {ex.title}
                    </span>
                    <span className="text-[10px] text-gray-500 font-mono font-medium truncate">
                      {ex.url}
                    </span>
                  </div>

                  <span className={`text-[9px] px-2 py-1 rounded-md border font-mono font-bold uppercase flex-shrink-0 ${ex.badgeColor}`}>
                    {ex.game}
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
