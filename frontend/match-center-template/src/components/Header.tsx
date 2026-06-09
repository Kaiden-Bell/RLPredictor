/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Search, Bell, CalendarRange, ChevronDown } from 'lucide-react';

interface HeaderProps {
  currentUrl?: string;
  onSearchUrl: (url: string) => void;
  isLoading?: boolean;
  onProfileClick: () => void;
}

export default function Header({ currentUrl = '', onSearchUrl, isLoading = false, onProfileClick }: HeaderProps) {
  const [inputValue, setInputValue] = useState(currentUrl);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSearchUrl(inputValue.trim());
    }
  };

  return (
    <header className="h-16 border-b border-app-border bg-app-bg px-6 flex items-center justify-between gap-4 z-20 flex-shrink-0">
      {/* Left Platform logo */}
      <div className="flex items-center gap-2">
        <h1 className="font-display font-bold text-xl tracking-tight text-white select-none">
          RL<span className="text-gray-300 font-medium">Predictor</span>
        </h1>
      </div>

      {/* Middle Scraping Bar form */}
      <form onSubmit={handleSubmit} className="flex-1 max-w-xl">
        <div className="relative group">
          <div className="absolute inset-y-0 left-3.5 flex items-center pointer-events-none text-gray-500 group-focus-within:text-brand-pink transition-colors">
            <Search size={16} />
          </div>
          <input
            id="header-search-bar"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={isLoading}
            type="text"
            placeholder="Search for Rocket League tournaments on Liquipedia..."
            className="w-full bg-app-surface border border-gray-800 focus:border-brand-pink text-sm text-gray-200 pl-11 pr-4 py-2 rounded-lg outline-none transition-all placeholder-gray-500 group-hover:border-gray-700 focus:group-hover:border-brand-pink"
          />
          {inputValue && (
            <button
              type="submit"
              disabled={isLoading}
              className="absolute right-2 top-1.5 bg-brand-pink/20 hover:bg-brand-pink/35 text-brand-pink text-xs font-semibold px-2.5 py-1 rounded transition-colors hidden sm:inline"
            >
              Scrape
            </button>
          )}
        </div>
      </form>

      {/* Right User Utility Items */}
      <div className="flex items-center gap-3">
        {/* Calendar button */}
        <button className="w-9 h-9 flex items-center justify-center rounded-lg text-gray-400 hover:text-white hover:bg-app-surface transition-colors">
          <CalendarRange size={18} />
        </button>

        {/* Notif Bell with status dot */}
        <button className="w-9 h-9 flex items-center justify-center rounded-lg text-gray-400 hover:text-white hover:bg-app-surface transition-colors relative">
          <Bell size={18} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-[#ef4444] rounded-full ring-2 ring-app-bg" />
        </button>

        <div className="h-6 w-px bg-app-border mx-1 hidden sm:block" />

        {/* Profile Avatar Trigger dropdown */}
        <div 
          onClick={onProfileClick}
          className="flex items-center gap-2 cursor-pointer group hover:bg-app-surface p-1 rounded-lg transition-colors select-none"
        >
          <div className="w-8 h-8 rounded-full border border-purple-500/30 overflow-hidden bg-brand-purple">
            <img 
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=256&auto=format&fit=crop" 
              alt="User" 
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover"
            />
          </div>
          <ChevronDown size={14} className="text-gray-400 group-hover:text-white transition-colors" />
        </div>
      </div>
    </header>
  );
}
