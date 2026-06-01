/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';

interface TeamLogoProps {
  name: string;
  className?: string;
  size?: number;
}

export default function TeamLogo({ name, className = '', size = 20 }: TeamLogoProps) {
  const normName = name.toLowerCase();

  // Custom vector SVGs representing realistic esports emblems
  if (normName.includes('g2')) {
    // G2 - Stylized Samurai helmet mask
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L4 7L12 9V2Z" fill="#F3F4F6" />
        <path d="M12 2L20 7L12 9V2Z" fill="#D1D5DB" />
        <path d="M4 7V17L12 22V9L4 7Z" fill="#E5E7EB" />
        <path d="M20 7V17L12 22V9L20 7Z" fill="#9CA3AF" />
        <path d="M7 11H17" stroke="#EF4444" strokeWidth="2" strokeLinecap="round" />
        <path d="M9 14H15" stroke="#F3F4F6" strokeWidth="2" strokeLinecap="round" />
        <circle cx="10" cy="9.5" r="1.5" fill="#EF4444" />
        <circle cx="14" cy="9.5" r="1.5" fill="#EF4444" />
      </svg>
    );
  }

  if (normName.includes('vital') || normName.includes('vit')) {
    // Vitality - Wasp Chevron logo
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 19H7L12 10L17 19H22L12 2Z" fill="#EAB308" />
        <path d="M12 10L9 15H15L12 10Z" fill="#111" />
        <line x1="12" y1="2" x2="12" y2="10" stroke="#000" strokeWidth="1.5" />
      </svg>
    );
  }

  if (normName.includes('karmine') || normName.includes('kc')) {
    // Karmine Corp - Stylized Monogram Ribbon "K" / "C"
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 4H8V20H4V4Z" fill="#3B82F6" />
        <path d="M8 12L16 4H20L12 12L20 20H16L8 12Z" fill="#60A5FA" />
        <circle cx="12" cy="12" r="3" fill="#1D4ED8" stroke="#3B82F6" strokeWidth="1" />
        <path d="M12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15" stroke="#F3F4F6" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }

  if (normName.includes('swnd') || normName.includes('swn')) {
    // Swnder - Orange Spider vector
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="5" fill="#F97316" />
        <circle cx="12" cy="7" r="2.5" fill="#F97316" />
        <path d="M6 12C6 12 8 8 12 8C16 8 18 12 18 12" stroke="#110E1C" strokeWidth="1.5" />
        <path d="M5 8C7 10 7 13 5 15" stroke="#F97316" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M19 8C17 10 17 13 19 15" stroke="#F97316" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M4 11C7 12 7 14 4 16" stroke="#F97316" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M20 11C17 12 17 14 20 16" stroke="#F97316" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }

  if (normName.includes('nopp') || normName.includes('nop')) {
    // Noppes - Purple Crest/Shield
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 22C12 22 20 18 20 10V5L12 2L4 5V10C4 18 12 22 12 22Z" fill="#8B5CF6" />
        <path d="M12 4L6 6.25V9.5C6 14.85 10.05 18.25 12 19.5V4Z" fill="#A78BFA" />
        <path d="M12 8L15 11" stroke="#F3F4F6" strokeWidth="2" strokeLinecap="round" />
        <path d="M12 14L9 11" stroke="#111" strokeWidth="2" strokeLinecap="round" />
      </svg>
    );
  }

  if (normName.includes('sinz') || normName.includes('sin')) {
    // Sinzline - Red stripes
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="24" height="24" rx="4" fill="#EF4444" />
        <line x1="4" y1="20" x2="20" y2="4" stroke="#F3F4F6" strokeWidth="3" />
        <line x1="8" y1="20" x2="20" y2="8" stroke="#F3F4F6" strokeWidth="1.5" />
        <line x1="4" y1="16" x2="16" y2="4" stroke="#F3F4F6" strokeWidth="1.5" />
      </svg>
    );
  }

  if (normName.includes('5w')) {
    // Team 5WS - Compass circular green
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" stroke="#22C55E" strokeWidth="2.5" />
        <polygon points="12,6 15,12 12,14 12,6" fill="#22C55E" />
        <polygon points="12,18 9,12 12,14 12,18" fill="#15803D" />
        <circle cx="12" cy="11.5" r="1.5" fill="#F3F4F6" />
      </svg>
    );
  }

  if (normName.includes('solar')) {
    // Solary - Blue Compass shield
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="12,2 22,12 12,22 2,12" fill="#2563EB" />
        <circle cx="12" cy="12" r="4" fill="#F3F4F6" />
        <polygon points="12,10 14,14 10,14" fill="#EF4444" />
      </svg>
    );
  }

  if (normName.includes('aud')) {
    // Audacity - Gold / Orange Abstract block
    return (
      <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 19V5C4 5 10 3 12 7C14 3 20 5 20 5V19C20 19 14 17 12 21C10 17 4 19 4 19Z" fill="#F59E0B" />
        <path d="M12 7C10.5 7 8 10 8 13C8 16 10 17 12 17V7Z" fill="#FBBF24" />
      </svg>
    );
  }

  // Generic fallback emblem
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="24" height="24" rx="6" fill="#4B5563" />
      <circle cx="12" cy="12" r="5" fill="#9CA3AF" />
      <path d="M12 9V15M9 12H15" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
