/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  Trophy, Brain, Cpu, MessageSquare, Sparkles, 
  ArrowRight, Shield, Database, Gauge, Zap 
} from 'lucide-react';

interface LandingPageProps {
  onEnterConsole: () => void;
  onEnterAuth: () => void;
}

export default function LandingPage({ onEnterConsole, onEnterAuth }: LandingPageProps) {
  const [demoStat, setDemoStat] = useState<'goals' | 'saves' | 'demos'>('demos');
  const [demoOver, setDemoOver] = useState(true);
  const [demoThreshold, setDemoThreshold] = useState(2.5);

  // Pseudo-dynamic calculation for live demo widget
  const getDemoExpectation = () => {
    let base = 50;
    if (demoStat === 'demos') base = demoOver ? 73.4 : 26.6;
    if (demoStat === 'goals') base = demoOver ? 61.2 : 38.8;
    if (demoStat === 'saves') base = demoOver ? 54.8 : 45.2;
    return base;
  };

  return (
    <div className="flex-1 flex flex-col p-6 md:p-12 relative overflow-hidden select-none animate-fade-in">
      {/* Dynamic background lighting */}
      <div className="absolute top-[10%] left-[25%] w-[450px] h-[450px] bg-purple-950/15 rounded-full blur-[130px] pointer-events-none z-0" />
      <div className="absolute bottom-[15%] right-[20%] w-[350px] h-[350px] bg-cyan-950/10 rounded-full blur-[110px] pointer-events-none z-0" />

      {/* Main Grid: Hero banner + Interactive prediction demo */}
      <div className="max-w-6xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-10 items-center relative z-10 py-4">
        
        {/* Left Column: Marketing & Explanatory Context */}
        <div className="lg:col-span-7 flex flex-col text-left gap-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-950/30 border border-brand-pink/20 rounded-full w-fit">
            <Sparkles size={13} className="text-brand-pink animate-pulse" />
            <span className="text-[10px] font-mono font-bold text-brand-pink uppercase tracking-widest">
              NEXT-GEN ESPORTS PREDICTION
            </span>
          </div>

          <div className="space-y-3">
            <h1 className="font-display font-extrabold text-4xl md:text-5xl lg:text-6xl text-white tracking-tight uppercase leading-none">
              Predict The <br />
              <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-cyan-400 bg-clip-text text-transparent">
                Unpredictable
              </span>
            </h1>
            <p className="text-sm md:text-base text-gray-400 max-w-xl leading-relaxed font-sans mt-2">
              RLPredictor fuses real-time professional telemetry scraping, public sentiment data, and a custom trained PyTorch Neural Network to output high-fidelity statistical probability projections for Rocket League Esports.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap items-center gap-4 mt-2">
            <button
              onClick={onEnterConsole}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-display font-bold py-3.5 px-7 rounded-2xl transition-all duration-300 flex items-center justify-center gap-2 group cursor-pointer shadow-[0_4px_25px_rgba(236,72,153,0.35)] hover:shadow-[0_4px_35px_rgba(236,72,153,0.55)] active:scale-98 hover:-translate-y-0.5"
            >
              <span>Launch Match Center</span>
              <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-1" />
            </button>

            <button
              onClick={onEnterAuth}
              className="bg-app-surface/90 hover:bg-app-surface border border-app-border hover:border-gray-700 text-gray-200 hover:text-white font-display font-bold py-3.5 px-6 rounded-2xl transition-all duration-300 cursor-pointer flex items-center gap-2"
            >
              <Shield size={16} className="text-brand-pink" />
              <span>Access Secure Portal</span>
            </button>
          </div>

          {/* Core Technical Highlights row */}
          <div className="grid grid-cols-3 gap-4 border-t border-purple-950/40 pt-6 mt-4">
            <div className="flex flex-col">
              <span className="font-display font-bold text-lg text-white">13-Dim</span>
              <span className="text-[10px] text-gray-500 font-medium uppercase font-sans mt-0.5">Feature vectors</span>
            </div>
            <div className="flex flex-col">
              <span className="font-display font-bold text-lg text-brand-glow">PRO=TRUE</span>
              <span className="text-[10px] text-gray-500 font-medium uppercase font-sans mt-0.5">Telemetry focus</span>
            </div>
            <div className="flex flex-col">
              <span className="font-display font-bold text-lg text-brand-pink">NLTK VADER</span>
              <span className="text-[10px] text-gray-500 font-medium uppercase font-sans mt-0.5">Social sentiment</span>
            </div>
          </div>
        </div>

        {/* Right Column: Premium AI Prediction Live Widget */}
        <div className="lg:col-span-5 w-full flex justify-center">
          <div className="w-full max-w-sm bg-[#110e1a]/85 border border-[#2e2645] p-6 rounded-3xl shadow-2xl flex flex-col gap-5 relative group overflow-hidden">
            {/* Glossy top-lighting border lines */}
            <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
            
            <div className="flex justify-between items-center select-none">
              <span className="text-[10px] font-mono text-brand-pink font-bold uppercase tracking-widest flex items-center gap-1.5">
                <Zap size={10} className="fill-brand-pink animate-pulse" />
                Live Demo Sandbox
              </span>
              <span className="text-[9px] bg-cyan-950/20 text-brand-glow border border-brand-glow/20 px-2 py-0.5 rounded font-mono font-medium">
                PyTorch Inference
              </span>
            </div>

            {/* Simulated Match Setup */}
            <div className="bg-[#0b0813] border border-purple-950/40 p-4 rounded-2xl flex justify-between items-center shadow-inner">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-gray-950 flex items-center justify-center font-display font-extrabold text-sm border border-gray-800 text-white">G2</div>
                <div className="flex flex-col">
                  <span className="font-display font-bold text-xs text-gray-200">G2 Esports</span>
                  <span className="text-[9px] text-gray-500 font-mono">Platform ID Verified</span>
                </div>
              </div>
              <span className="font-mono text-gray-600 text-xs font-bold">vs</span>
              <div className="flex items-center gap-2.5 text-right">
                <div className="flex flex-col">
                  <span className="font-display font-bold text-xs text-gray-200">Vitality</span>
                  <span className="text-[9px] text-gray-500 font-mono">Zen resolved</span>
                </div>
                <div className="w-8 h-8 rounded-lg bg-yellow-950/30 flex items-center justify-center font-display font-extrabold text-sm border border-yellow-500/20 text-yellow-400">VIT</div>
              </div>
            </div>

            {/* Demo Controller Tabs */}
            <div className="flex flex-col gap-3">
              <span className="text-[10px] text-gray-500 uppercase font-mono tracking-wider font-semibold">Choose Stat Parameter:</span>
              <div className="grid grid-cols-3 gap-2">
                {(['goals', 'saves', 'demos'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setDemoStat(s)}
                    className={`py-2 rounded-xl text-xs font-display font-bold uppercase transition-all duration-200 border cursor-pointer ${
                      demoStat === s 
                        ? 'bg-purple-950/20 text-brand-pink border-brand-pink/35 shadow-[0_0_10px_rgba(236,72,153,0.1)]' 
                        : 'bg-app-bg text-gray-500 border-purple-950 hover:text-gray-300'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>

              {/* Slider for Over/Under choice */}
              <div className="flex items-center justify-between border border-purple-950/40 p-2.5 rounded-xl bg-app-bg select-none">
                <button
                  onClick={() => setDemoOver(true)}
                  className={`flex-1 py-1.5 text-center text-xs font-bold rounded-lg transition-colors cursor-pointer ${
                    demoOver ? 'bg-emerald-950/30 text-emerald-400 border border-emerald-500/15' : 'text-gray-600 hover:text-gray-400'
                  }`}
                >
                  OVER
                </button>
                <button
                  onClick={() => setDemoOver(false)}
                  className={`flex-1 py-1.5 text-center text-xs font-bold rounded-lg transition-colors cursor-pointer ${
                    !demoOver ? 'bg-rose-950/30 text-rose-400 border border-rose-500/15' : 'text-gray-600 hover:text-gray-400'
                  }`}
                >
                  UNDER
                </button>
              </div>

              {/* Threshold values toggle */}
              <div className="flex items-center justify-between bg-app-bg border border-purple-950/40 p-3 rounded-xl">
                <span className="text-[10px] text-gray-500 font-mono uppercase">Target Threshold:</span>
                <div className="flex items-center gap-2">
                  {[1.5, 2.5, 3.5].map((t) => (
                    <button
                      key={t}
                      onClick={() => setDemoThreshold(t)}
                      className={`w-8 h-8 rounded-lg text-xs font-mono font-bold flex items-center justify-center cursor-pointer transition-colors ${
                        demoThreshold === t 
                          ? 'bg-cyan-950/30 text-brand-glow border border-brand-glow/20' 
                          : 'text-gray-500 hover:text-gray-300'
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Neural Net Output Circular Gauge Representation */}
            <div className="flex flex-col items-center justify-center border border-[#2e2645] bg-[#0c0915] p-5 rounded-2xl select-none relative gap-3 shadow-inner">
              <div className="relative w-28 h-28 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" stroke="rgba(236, 72, 153, 0.06)" strokeWidth="6" fill="transparent" />
                  <circle cx="50" cy="50" r="42" stroke="url(#demoGradient)" strokeWidth="6" fill="transparent"
                    strokeDasharray={263.8}
                    strokeDashoffset={263.8 - (263.8 * getDemoExpectation()) / 100}
                    className="transition-all duration-700 ease-out"
                  />
                  <defs>
                    <linearGradient id="demoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#9333ea" />
                      <stop offset="100%" stopColor="#ec4899" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute flex flex-col items-center justify-center">
                  <span className="font-display font-extrabold text-xl text-white tracking-tighter leading-none">
                    {getDemoExpectation()}%
                  </span>
                  <span className="text-[8px] text-gray-500 uppercase tracking-widest font-mono mt-0.5">
                    Probability
                  </span>
                </div>
              </div>

              <div className="flex flex-col items-center text-center gap-1.5">
                <span className="text-xs font-display font-semibold text-gray-200">
                  AI Pick: <span className={demoOver ? 'text-emerald-400' : 'text-rose-400'}>
                    {demoOver ? 'Over' : 'Under'} {demoThreshold} {demoStat}
                  </span>
                </span>
                <span className="text-[9px] text-gray-500 leading-relaxed font-sans max-w-[220px]">
                  Engineered from H2H averages, 2s momentum indices, and NLTK reddit sentiment.
                </span>
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* Footer / Value Prop Cards Section */}
      <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 relative z-10 border-t border-purple-950/30 pt-10">
        
        {/* Card 1 */}
        <div className="bg-app-surface/40 border border-purple-950/40 p-5 rounded-2xl flex flex-col text-left gap-3.5 group hover:border-[#ec4899]/30 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-purple-950/20 border border-purple-500/20 flex items-center justify-center text-brand-pink">
            <Brain size={20} />
          </div>
          <div className="space-y-1">
            <h4 className="font-display font-bold text-sm text-gray-200 uppercase tracking-wide">
              Neural Prediction Engine
            </h4>
            <p className="text-xs text-gray-500 leading-relaxed font-sans">
              Custom PyTorch MLP model trained chronologically over historical performance matrices to output exact betting odds probability expectations.
            </p>
          </div>
        </div>

        {/* Card 2 */}
        <div className="bg-app-surface/40 border border-purple-950/40 p-5 rounded-2xl flex flex-col text-left gap-3.5 group hover:border-[#22d3ee]/30 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-cyan-950/20 border border-brand-glow/20 flex items-center justify-center text-brand-glow">
            <Database size={20} />
          </div>
          <div className="space-y-1">
            <h4 className="font-display font-bold text-sm text-gray-200 uppercase tracking-wide">
              Scraping & Caching
            </h4>
            <p className="text-xs text-gray-500 leading-relaxed font-sans">
              Automated Liquipedia and Ballchasing API pipelines pulling recent scrimmage telemetry with smart SQLite caching layers.
            </p>
          </div>
        </div>

        {/* Card 3 */}
        <div className="bg-app-surface/40 border border-purple-950/40 p-5 rounded-2xl flex flex-col text-left gap-3.5 group hover:border-[#eab308]/30 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-yellow-950/20 border border-yellow-500/20 flex items-center justify-center text-yellow-500">
            <MessageSquare size={20} />
          </div>
          <div className="space-y-1">
            <h4 className="font-display font-bold text-sm text-gray-200 uppercase tracking-wide">
              Reddit Sentiment Analyzer
            </h4>
            <p className="text-xs text-gray-500 leading-relaxed font-sans">
              Uses NLTK VADER sentiment analyzer to gauge public forum opinions and scaled momentum matrices for targeted roster grids.
            </p>
          </div>
        </div>

      </div>

    </div>
  );
}
