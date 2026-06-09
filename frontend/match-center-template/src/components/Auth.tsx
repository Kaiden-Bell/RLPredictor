/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Shield, ShieldAlert, Cpu, Key, Lock, Mail, ArrowRight, ShieldCheck, CheckCircle } from 'lucide-react';

interface AuthProps {
  onAuthSuccess: () => void;
  onBackToLanding: () => void;
}

type AuthMode = 'signin' | 'signup' | 'mfa';

export default function Auth({ onAuthSuccess, onBackToLanding }: AuthProps) {
  const [mode, setMode] = useState<AuthMode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [mfaCode, setMfaCode] = useState(['', '', '', '', '', '']);
  const [isLoading, setIsLoading] = useState(false);
  const [passStrength, setPassStrength] = useState(0); // 0 to 4

  const mfaRefs = [
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null)
  ];

  // Dynamic password strength computation
  useEffect(() => {
    if (!password) {
      setPassStrength(0);
      return;
    }
    let strength = 0;
    if (password.length >= 6) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[0-9]/.test(password)) strength += 1;
    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    setPassStrength(strength);
  }, [password]);

  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setErrorMsg('Please populate all credential inputs to proceed.');
      return;
    }
    if (!email.includes('@')) {
      setErrorMsg('Please specify a valid email address.');
      return;
    }
    setErrorMsg('');
    setIsLoading(true);

    // Simulated network delay
    setTimeout(() => {
      setIsLoading(false);
      // Route to MFA verification step
      setMode('mfa');
    }, 1000);
  };

  const handleMfaChange = (index: number, value: string) => {
    if (value.length > 1) value = value.slice(-1);
    const newCode = [...mfaCode];
    newCode[index] = value;
    setMfaCode(newCode);
    setErrorMsg('');

    // Auto-focus next field
    if (value && index < 5) {
      mfaRefs[index + 1].current?.focus();
    }
  };

  const handleMfaKeyDown = (index: number, e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Backspace' && !mfaCode[index] && index > 0) {
      mfaRefs[index - 1].current?.focus();
    }
  };

  const handleMfaSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const finalCode = mfaCode.join('');
    if (finalCode.length < 6) {
      setErrorMsg('Please populate all 6 numeric digits of your verification code.');
      return;
    }
    setErrorMsg('');
    setIsLoading(true);

    setTimeout(() => {
      setIsLoading(false);
      if (finalCode === '123456' || finalCode.startsWith('7') || finalCode.endsWith('7') || finalCode === '000000' || finalCode.length === 6) {
        // Successful verification!
        onAuthSuccess();
      } else {
        setErrorMsg('Invalid token code. Enter any 6 digits to verify mock authentication.');
      }
    }, 1200);
  };

  const getStrengthLabel = () => {
    if (passStrength === 0) return 'None';
    if (passStrength === 1) return 'Weak';
    if (passStrength === 2) return 'Fair';
    if (passStrength === 3) return 'Strong';
    return 'Hyper-Secure';
  };

  const getStrengthColor = () => {
    if (passStrength <= 1) return 'bg-rose-500 shadow-[0_0_8px_#f43f5e]';
    if (passStrength === 2) return 'bg-yellow-500 shadow-[0_0_8px_#eab308]';
    return 'bg-emerald-500 shadow-[0_0_8px_#10b981]';
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 relative overflow-hidden select-none animate-fade-in">
      {/* Background glow halos */}
      <div className="absolute top-[30%] left-[50%] -translate-x-[50%] -translate-y-[50%] w-[500px] h-[500px] bg-purple-950/15 rounded-full blur-[130px] pointer-events-none z-0" />
      
      <div className="w-full max-w-md bg-app-surface border border-app-border rounded-3xl p-8 shadow-2xl relative z-10 text-center flex flex-col gap-6">
        
        {/* Core Header */}
        <div className="flex flex-col items-center gap-2 select-none">
          <div className="w-12 h-12 rounded-xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center text-brand-pink relative">
            <Shield size={24} className="animate-pulse" />
            <div className="absolute inset-0 rounded-xl border border-brand-pink/30 animate-ping opacity-25" />
          </div>
          
          <h2 className="font-display font-extrabold text-2xl tracking-tight text-white uppercase mt-1">
            {mode === 'signin' && 'Member Access'}
            {mode === 'signup' && 'Register Account'}
            {mode === 'mfa' && 'Verify Identity'}
          </h2>
          <p className="text-xs text-gray-500 leading-relaxed font-sans max-w-xs mx-auto">
            {mode === 'signin' && 'Verify your credentials to unlock custom esports modeling prediction vaults.'}
            {mode === 'signup' && 'Create secure keys to manage models, scraper rates, and custom statistics pools.'}
            {mode === 'mfa' && 'Enter your 6-digit authenticator passcode to establish a secure database session.'}
          </p>
        </div>

        {/* Dynamic Forms */}
        {mode !== 'mfa' ? (
          <form onSubmit={handleAuthSubmit} className="flex flex-col gap-4 text-left">
            {/* Email Field */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-mono text-gray-500 uppercase tracking-widest font-semibold">Email Credentials:</label>
              <div className="relative">
                <span className="absolute left-4 top-[50%] -translate-y-[50%] text-gray-500"><Mail size={15} /></span>
                <input
                  type="text"
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); setErrorMsg(''); }}
                  placeholder="name@agency.com"
                  className="w-full bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/40 text-xs text-gray-200 pl-11 pr-4 py-3 rounded-xl outline-none transition-all placeholder-gray-600 shadow-inner"
                />
              </div>
            </div>

            {/* Password Field */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-mono text-gray-500 uppercase tracking-widest font-semibold">Passphrase Key:</label>
              <div className="relative">
                <span className="absolute left-4 top-[50%] -translate-y-[50%] text-gray-500"><Lock size={15} /></span>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setErrorMsg(''); }}
                  placeholder="••••••••••••••"
                  className="w-full bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/40 text-xs text-gray-200 pl-11 pr-4 py-3 rounded-xl outline-none transition-all placeholder-gray-600 shadow-inner"
                />
              </div>
              
              {/* Active strength meter for Sign Up Mode */}
              {mode === 'signup' && password.length > 0 && (
                <div className="flex flex-col gap-1 mt-1 animate-fade-in">
                  <div className="flex justify-between items-center text-[9px] font-mono">
                    <span className="text-gray-500">Security Level:</span>
                    <span className={passStrength <= 1 ? 'text-rose-400' : passStrength === 2 ? 'text-yellow-400' : 'text-emerald-400'}>
                      {getStrengthLabel()}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-1 h-1 bg-[#110e1a] rounded-full overflow-hidden mt-0.5">
                    {[1, 2, 3, 4].map((step) => (
                      <div
                        key={step}
                        className={`h-full rounded-full transition-all duration-300 ${
                          step <= passStrength ? getStrengthColor() : 'bg-transparent'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Error alerts */}
            {errorMsg && (
              <div className="flex items-center gap-2 text-rose-500 text-[11px] bg-rose-950/15 border border-rose-500/20 px-3.5 py-2.5 rounded-xl animate-shake">
                <ShieldAlert size={14} className="flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 text-white font-display font-bold py-3 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 group cursor-pointer shadow-[0_4px_20px_rgba(236,72,153,0.3)] active:scale-98 mt-2"
            >
              {isLoading ? (
                <>
                  <Cpu className="animate-spin text-white" size={16} />
                  <span>Decrypting Keyring...</span>
                </>
              ) : (
                <>
                  <span>{mode === 'signin' ? 'Verify Credentials' : 'Initialize Account'}</span>
                  <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
                </>
              )}
            </button>

            {/* Toggle Sign In vs Sign Up */}
            <div className="flex items-center justify-between text-[10px] font-mono border-t border-purple-950/40 pt-4 mt-2">
              {mode === 'signin' ? (
                <>
                  <span className="text-gray-500">Need secure credentials?</span>
                  <button
                    type="button"
                    onClick={() => { setMode('signup'); setErrorMsg(''); }}
                    className="text-brand-pink hover:text-pink-400 font-semibold cursor-pointer outline-none"
                  >
                    Register Account
                  </button>
                </>
              ) : (
                <>
                  <span className="text-gray-500">Already a registered agent?</span>
                  <button
                    type="button"
                    onClick={() => { setMode('signin'); setErrorMsg(''); }}
                    className="text-brand-pink hover:text-pink-400 font-semibold cursor-pointer outline-none"
                  >
                    Member Log In
                  </button>
                </>
              )}
            </div>
          </form>
        ) : (
          /* MFA OTP PASSCODE INPUT PANEL */
          <form onSubmit={handleMfaSubmit} className="flex flex-col gap-5 text-left animate-fade-in">
            <div className="flex flex-col gap-2.5 items-center justify-center">
              <div className="flex gap-2">
                {mfaCode.map((val, idx) => (
                  <input
                    ref={mfaRefs[idx]}
                    key={idx}
                    type="text"
                    pattern="[0-9]*"
                    maxLength={1}
                    value={val}
                    onChange={(e) => handleMfaChange(idx, e.target.value)}
                    onKeyDown={(e) => handleMfaKeyDown(idx, e)}
                    className="w-11 h-12 bg-[#110e1a] border border-gray-800 focus:border-brand-pink focus:ring-1 focus:ring-brand-pink/40 text-center font-display font-extrabold text-lg text-white rounded-xl outline-none transition-all shadow-inner"
                  />
                ))}
              </div>
              <span className="text-[10px] font-mono text-gray-500 text-center leading-relaxed">
                Hint: Enter <span className="text-brand-pink font-semibold">123456</span> or any 6 digits to verify.
              </span>
            </div>

            {errorMsg && (
              <div className="flex items-center gap-2 text-rose-500 text-[11px] bg-rose-950/15 border border-rose-500/20 px-3.5 py-2.5 rounded-xl animate-shake">
                <ShieldAlert size={14} className="flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            <div className="flex flex-col gap-2 mt-2">
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 text-white font-display font-bold py-3 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 group cursor-pointer shadow-[0_4px_20px_rgba(236,72,153,0.3)] active:scale-98"
              >
                {isLoading ? (
                  <>
                    <Cpu className="animate-spin text-white" size={16} />
                    <span>Verifying Session Token...</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck size={16} />
                    <span>Establish Session</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => { setMode('signin'); setMfaCode(['', '', '', '', '', '']); setErrorMsg(''); }}
                className="w-full py-2 bg-transparent text-gray-500 hover:text-gray-300 text-center text-xs font-semibold outline-none cursor-pointer"
              >
                Back to Credentials Input
              </button>
            </div>
          </form>
        )}

        {/* Back to Home Button */}
        <button
          onClick={onBackToLanding}
          className="text-xs text-gray-500 hover:text-gray-200 flex items-center justify-center gap-1.5 outline-none cursor-pointer transition-colors border-t border-purple-950/40 pt-4"
        >
          <span>← Back to Marketing Homepage</span>
        </button>

      </div>
    </div>
  );
}
