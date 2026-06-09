/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Home, Trophy, Sparkles, Settings, LogOut } from 'lucide-react';

interface SidebarProps {
  onReset?: () => void;
  hasData?: boolean;
  activeItem: string;
  setActiveItem: (item: string) => void;
}

export default function Sidebar({ onReset, hasData = false, activeItem, setActiveItem }: SidebarProps) {

  const navItems = [
    { id: 'landing', icon: Home, label: 'Landing Home' },
    { id: 'bracket', icon: Trophy, label: 'Match Center' },
    { id: 'prediction', icon: Sparkles, label: 'AI Predictor' },
    { id: 'settings', icon: Settings, label: 'Platform Options' },
  ];

  return (
    <aside className="w-16 md:w-20 bg-app-bg border-r border-app-border flex flex-col items-center justify-between py-6 h-full flex-shrink-0 z-30">
      {/* Logos */}
      <div 
        onClick={onReset}
        className="cursor-pointer group flex flex-col items-center justify-center relative mb-4"
        title="Reset to Getting Started"
      >
        <div className="w-10 h-10 rounded-xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center transition-all duration-300 group-hover:border-purple-400 group-hover:shadow-[0_0_15px_rgba(147,51,234,0.3)]">
          <span className="font-display font-bold text-lg bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">RL</span>
          <span className="font-display font-medium text-xs text-brand-pink absolute -bottom-1 -right-1 bg-app-bg border border-app-border rounded px-0.5 scale-75 select-none font-bold">P</span>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 flex flex-col justify-center gap-4 w-full px-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeItem === item.id;
          return (
            <button
              id={`sidebar-${item.id}`}
              key={item.id}
              onClick={() => setActiveItem(item.id)}
              className={`w-full py-3 rounded-xl flex items-center justify-center relative group transition-all duration-200 outline-none ${
                isActive 
                  ? 'text-brand-pink bg-purple-950/20 border border-purple-500/10' 
                  : 'text-gray-500 hover:text-gray-200 hover:bg-white/5'
              }`}
            >
              {/* Active Indicator bar */}
              {isActive && (
                <div className="absolute left-0 top-3 bottom-3 w-1 bg-brand-pink rounded-r-full shadow-[0_0_8px_#ec4899]" />
              )}
              
              <Icon size={isActive ? 22 : 20} className="transition-transform duration-300 group-hover:scale-110" />

              {/* Tooltip */}
              <div className="absolute left-16 md:left-20 bg-[#171324] border border-app-border text-white text-xs font-medium py-1.5 px-3 rounded-lg opacity-0 translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 shadow-2xl whitespace-nowrap z-50">
                {item.label}
              </div>
            </button>
          );
        })}
      </nav>

      {/* Power Off/Log Out at Bottom */}
      <div className="w-full px-2">
        <button
          id="sidebar-logout"
          onClick={onReset}
          className="w-full py-3 rounded-xl flex items-center justify-center text-gray-500 hover:text-red-400 hover:bg-red-950/10 transition-all duration-200 group relative outline-none"
        >
          <LogOut size={20} className="transition-transform duration-300 group-hover:translate-x-0.5" />
          <div className="absolute left-16 md:left-20 bg-[#171324] border border-app-border text-[#ef4444] text-xs font-medium py-1.5 px-3 rounded-lg opacity-0 translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 shadow-2xl whitespace-nowrap z-50">
            Disconnect URL
          </div>
        </button>
      </div>
    </aside>
  );
}
