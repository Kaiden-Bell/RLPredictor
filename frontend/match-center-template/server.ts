/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialize Gemini client to avoid crashes if API key is missing
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI | null {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (apiKey && apiKey !== 'MY_GEMINI_API_KEY') {
      aiClient = new GoogleGenAI({
        apiKey: apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });
      console.log('Gemini client initialized successfully.');
    } else {
      console.log('No valid GEMINI_API_KEY environment variable. App will run in standalone high-fidelity mode.');
    }
  }
  return aiClient;
}

// Default high-fidelity mockup data matching the reference image exactly
const fallbackTournamentData = {
  name: "RLPredictor - Rocket League Championship Series",
  url: "https://liquipedia.net/rocketleague/Rocket_League_Championship_Series/2026/Major_1",
  game: "Rocket League",
  bracketMatches: [
    // Quarterfinals (roundIndex: 0)
    { id: "q1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2", color: "#111" }, team2: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality", color: "#eab308" }, score1: 3, score2: 2, winnerId: "g2", status: "completed", roundIndex: 0 },
    { id: "q2", matchIndex: 1, team1: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality", color: "#eab308" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine", color: "#3b82f6" }, score1: 3, score2: 1, winnerId: "vit", status: "completed", roundIndex: 0 },
    { id: "q3", matchIndex: 2, team1: { id: "swn", name: "Swnder Esports", shortName: "Swnder", logo: "Swnder", color: "#f97316" }, team2: { id: "nop", name: "Noppes Esports", shortName: "Noppes", logo: "Noppes", color: "#a855f7" }, score1: 1, score2: 3, winnerId: "nop", status: "completed", roundIndex: 0 },
    { id: "q4", matchIndex: 3, team1: { id: "sin", name: "Sinzline Gaming", shortName: "Sinzline", logo: "Sinzline", color: "#ef4444" }, team2: { id: "t5w", name: "Team 5WS", shortName: "Team 5WS", logo: "Team 5WS", color: "#22c55e" }, score1: 1, score2: 3, winnerId: "t5w", status: "completed", roundIndex: 0 },
    
    // Semifinals (roundIndex: 1)
    { id: "s1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2", color: "#111" }, team2: { id: "vit", name: "Team Vitality", shortName: "Vitality", logo: "Vitality", color: "#eab308" }, score1: 4, score2: 3, winnerId: "g2", status: "completed", roundIndex: 1 },
    { id: "s2", matchIndex: 1, team1: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine", color: "#3b82f6" }, team2: { id: "nop", name: "Noppes Esports", shortName: "Noppes", logo: "Noppes", color: "#a855f7" }, score1: 4, score2: 0, winnerId: "kc", status: "completed", roundIndex: 1 },
    
    // Finals (roundIndex: 2)
    { id: "f1", matchIndex: 0, team1: { id: "g2", name: "G2 Esports", shortName: "G2", logo: "G2", color: "#111" }, team2: { id: "kc", name: "Karmine Corp", shortName: "Karmine", logo: "Karmine", color: "#3b82f6" }, status: "live", roundIndex: 2 }
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
        { id: "v4", name: "Saizen", role: "Active Roster" },
        { id: "v5", name: "FairyPeak", role: "Coach" },
        { id: "v6", name: "ExoTiiK", role: "Active Roster" }
      ],
      substitutes: [
        { id: "v7", name: "Mizu", role: "Bench" },
        { id: "v8", name: "Kaydop", role: "Bench" }
      ]
    },
    kc: {
      teamId: "kc",
      active: [
        { id: "k1", name: "Vatira", role: "Active Roster" },
        { id: "k2", name: "Atow.", role: "Active Roster" },
        { id: "k3", name: "Rise.", role: "Active Roster" },
        { id: "k4", name: "Kamet0", role: "Manager" },
        { id: "k5", name: "Ferra05", role: "Coach" },
        { id: "k6", name: "Eversax", role: "Strategy" }
      ],
      substitutes: [
        { id: "k7", name: "Itachi", role: "Bench" },
        { id: "k8", name: "Chausette", role: "Bench" },
        { id: "k9", name: "Sly", role: "Reserve" }
      ]
    },
    swn: {
      teamId: "swn",
      active: [
        { id: "s1", name: "Swndr_1", role: "Striker" },
        { id: "s2", name: "Spider", role: "Midfield" },
        { id: "s3", name: "Cobweb", role: "Defense" }
      ],
      substitutes: []
    },
    nop: {
      teamId: "nop",
      active: [
        { id: "n1", name: "Noppe_1", role: "Striker" },
        { id: "n2", name: "Shield", role: "Midfield" },
        { id: "n3", name: "Guard", role: "Defense" }
      ],
      substitutes: []
    },
    sin: {
      teamId: "sin",
      active: [
        { id: "sn1", name: "Sinz_Red", role: "Striker" },
        { id: "sn2", name: "Line", role: "Midfield" },
        { id: "sn3", name: "Track", role: "Defense" }
      ],
      substitutes: []
    },
    t5w: {
      teamId: "t5w",
      active: [
        { id: "t1", name: "Five", role: "Striker" },
        { id: "t2", name: "Wave", role: "Midfield" },
        { id: "t3", name: "Storm", role: "Defense" }
      ],
      substitutes: []
    },
    aud: {
      teamId: "aud",
      active: [
        { id: "a1", name: "Aud_Pro", role: "Striker" },
        { id: "a2", name: "Bold", role: "Midfield" },
        { id: "a3", name: "Courage", role: "Defense" }
      ],
      substitutes: []
    }
  }
};

// API Route for Tournament analysis
app.post('/api/tournament/analyze', async (req, res) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }

  const client = getGeminiClient();
  if (!client) {
    // Graceful fallback if API key is missing
    console.log('Using robust mockup data matching reference image.');
    return res.json({
      data: fallbackTournamentData,
      isMock: true,
      message: 'Running in high-fidelity sandbox mode. Configure your GEMINI_API_KEY for dynamic real-time URL bracket Generation.'
    });
  }

  try {
    // Generate content using Gemini
    const prompt = `
You are an expert esports data analyst and scraper. The user has provided an esports tournament URL: "${url}".
Please analyze this URL. Based on the URL contents, paths, or names, reconstruct a highly realistic tournament representation for this event.

If the tournament looks like a Rocket League event (e.g. "RLCS", "Rocket League", etc.), use genuine teams like G2 Esports, Team Vitality, Karmine Corp, Team BDS, Spacestation Gaming, Gen.G, Team Falcons, Gentle Mates, etc.
If the tournament looks like League of Legends, Counter-Strike (CS2 or CS:GO), Valorant, or any other game, adapt the teams, names, ratings, and rosters perfectly to fit that specific game!

Generate a comprehensive JSON response matching the following strict schema:
{
  "name": "The actual full tournament name (e.g. 'RLCS 24 Copenhagen Major' or 'PGL CS2 Major Copenhagen')",
  "game": "The name of the competitive game (e.g. 'Rocket League', 'Counter-Strike 2', 'Valorant', 'League of Legends')",
  "url": "${url}",
  "bracketMatches": [
    // Provide a bracket consisting of 4 Quarterfinals (roundIndex: 0, matchIndex: 0..3),
    // 2 Semifinals (roundIndex: 1, matchIndex: 0..1),
    // and 1 Final match (roundIndex: 2, matchIndex: 0) which is currently live or scheduled.
    // Ensure accurate ids (like "q1", "q2", "s1", "f1"). Keep winnerId populated for completed matches.
    // All team objects should have id, name, shortName, logo (key like 'G2', 'Vitality', 'Karmine', 'BDS', 'GenG' or custom based on the teams), color (hex color).
    // Let's make G2 and Karmine Corp match references as we need to support identical styling where possible.
  ],
  "upcomingMatches": [
    // Array of 2 upcoming or recently scheduled matches: { id, team1: Team, team2: Team, status: 'upcoming' | 'completed', score1?, score2?, time, date }
  ],
  "finishedMatches": [
    // Array of 1 finished match: { id, team1: Team, team2: Team, status: 'finished', time, date }
  ],
  "completedMatches": [
    // Array of 3 matches that are fully completed, with score1, score2, status: 'completed', time, date
  ],
  "winProbability": {
    "team1": Team,
    "team2": Team,
    "prob1": 80, // percentage integer
    "prob2": 68  // percentage integer
  },
  "playerRatings": [
    // Top 2 standout player ratings: { name: string, rating: float (e.g. 7.86, 6.70) }
  ],
  "teamForms": [
    // Top 2 team form representations: { teamName: string, logo: string (logo name), form: Array of 6 chars, either 'W' or 'L' }
  ],
  "rosters": {
    // Map with rosters for every team ID referenced in the bracket:
    "teamId": {
      "teamId": "teamId",
      "active": [
        // Array of 3 to 6 active player objects: { id: string, name: string, role: string }
      ],
      "substitutes": [
        // Array of 2 to 3 backup player/coach/manager objects: { id: string, name: string, role: string }
      ]
    }
  }
}

Respond ONLY with raw JSON. Excellent formatting is required. Must be directly parseable. Do not wrap in markdown \`\`\`json blocks.
`;

    const response = await client.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      }
    });

    const text = response.text?.trim() || '';
    let parsedData;
    try {
      parsedData = JSON.parse(text);
    } catch (parseError) {
      console.error('Gemini JSON parsing expired, text response was:', text);
      throw new Error('Received malformed JSON from Gemini');
    }

    return res.json({
      data: parsedData,
      isMock: false,
    });

  } catch (error: any) {
    console.error('Error generating predictive brackets with Gemini:', error);
    return res.json({
      data: fallbackTournamentData,
      isMock: true,
      error: error.message || 'Error occurred during generation',
      message: 'Fell back to default high-fidelity tournament setup.'
    });
  }
});

// Setup Vite Dev server or Production static handlers
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
    console.log('Vite middleware mounted in development.');
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
    console.log('Serving production build assets from /dist.');
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on status http://localhost:${PORT}`);
  });
}

startServer();
