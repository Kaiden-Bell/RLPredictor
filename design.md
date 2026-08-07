# RLPredictor Refactoring Design: Selenium + BS4 -> Async Playwright

## Overview
This document outlines the phased approach for migrating the RLPredictor web scraping pipeline from a synchronous Selenium and Beautiful Soup (BS4) stack to an entirely asynchronous Playwright architecture. 

By migrating in phases, we can ensure stability, test each individual component, and gather the necessary B-roll for your video content (including moments where things might break as we refactor!).

## Core Objectives
1. **Speed & Efficiency:** Use Playwright's network interception to block images, fonts, and CSS, prioritizing raw data extraction speed for brackets.
2. **Code Cleanliness:** Eventually remove BS4 and strict regex matching, replacing them with Playwright's robust `page.locator()` DOM traversal mechanisms.
3. **Async Architecture:** Leverage FastAPI's async nature by maintaining a persistent global Playwright browser instance, rather than spinning up a new browser per request.

---

## Phase 1: Foundation & Backend Setup
**Goal:** Introduce Playwright to the stack and set up the global async browser lifecycle without breaking existing scrapers.

1. **Dependencies:**
   - Add `playwright` to `requirements.txt`.
   - Run `playwright install chromium` in the environment.
2. **Server Lifecycle (`server.py`):**
   - Implement a FastAPI `lifespan` event (or `@app.on_event("startup")`) to launch a global `async_playwright` instance and a persistent `Browser`.
   - Ensure the browser gracefully closes on shutdown.
3. **State Management:**
   - Attach the global browser to the FastAPI `app.state` so that incoming requests can access it.

*Expected outcome:* The server starts up with a persistent headless Chromium instance running in the background. The existing Selenium/BS4 scrapers remain untouched and functional.

---

## Phase 2: Hybrid Migration (Playwright + BS4)
**Goal:** Eliminate Selenium by using Playwright to fetch the JS-rendered HTML, then passing it to our existing BS4 logic. This is a safe intermediate step.

1. **Playwright Integration in Scrapers:**
   - Convert `scrape_playoffs` to an `async def`.
   - Instead of launching `selenium.webdriver`, use `await app.state.browser.new_page()`.
2. **Network Interception for Speed:**
   - Implement a route handler on the Playwright page to instantly abort requests for images, stylesheets, media, and fonts.
3. **The Hybrid Handoff:**
   - Call `await page.content()` to get the fully rendered DOM.
   - Pass this HTML string directly into our existing `BeautifulSoup` parsing logic.
4. **Server Updates:**
   - Update the `/api/scrape` endpoint in `server.py` to `await` the new hybrid `scrape_playoffs`.

*Expected outcome:* Selenium is completely removed. Scraping is significantly faster due to Playwright's speed and network interception, but the convoluted BS4 parsing logic is preserved for now. *(Great spot for B-roll showing the speed improvement!)*

---

## Phase 3: Pure Playwright (Removing BS4 from Brackets)
**Goal:** Rewrite the complex bracket extraction logic to natively use Playwright locators, removing the need for BS4 in the primary scraper.

1. **DOM Traversal Refactor:**
   - Replace BS4 `soup.select()` and `soup.find_all()` with Playwright's `page.locator()`.
   - Refactor `round_map`, `nearest_sect`, and `extract_roster` to use Playwright's async element evaluation (e.g., `await locator.inner_text()`, `await locator.get_attribute('aria-label')`).
2. **Dropping Regex:**
   - Use Playwright's powerful CSS and text selectors (e.g., `page.locator("text=Active")`) to avoid messy regex matching where possible.
3. **Eliminating the "Light" Route:**
   - Since Playwright is now extremely fast and handles all JS rendering natively, we can deprecate and remove the `scrape_tournament_light` fallback route in `server.py`.

*Expected outcome:* The primary playoff scraper is purely Playwright. This phase is where things might break during development as we map BS4 logic to Playwright locators. *(Perfect for "breaking" B-roll).*

---

## Phase 4: Consolidating Remaining Scrapers
**Goal:** Complete the migration by updating the remaining supplementary scrapers and stripping out legacy dependencies.

1. **H2H & Ballchasing (`h2h_ballchasing.py`):**
   - Convert the Liquipedia H2H fetching to use the global Playwright instance.
   - Keep standard HTTP REST calls (for the Ballchasing API) using `requests` or swap to `httpx` for full async.
2. **Player Profiles (`liquipedia_players.py`):**
   - Convert `scrape_player_profile` to use Playwright locators for pulling alt IDs and Steam links.
3. **Cleanup:**
   - Remove `beautifulsoup4` and `requests` (if fully migrated to `httpx`/Playwright) from `requirements.txt`.

*Expected outcome:* A clean, 100% async backend architecture. No Selenium, no Beautiful Soup.
