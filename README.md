# Trip Planner Capstone (Streamlit)

A consumer-style “fun” capstone project for an AI Engineer career path: a trip planner app that uses an LLM agent + tool-calling to build a multi-day itinerary for any city, pulls live points-of-interest (POIs) from OpenStreetMap, optionally grounds suggestions with Wikivoyage snippets (RAG), and includes a lightweight feedback loop that improves results over time.

---

## What this demonstrates (AI Engineer skills)

- **LLMs + Agents (tool calling):** Uses the OpenAI Responses API with **strict** tool schemas.
- **Retrieval / Grounding (RAG):** Optional runtime retrieval from Wikivoyage + TF-IDF ranking.
- **External data ingestion:** Live POIs via OpenStreetMap (Nominatim geocoding + Overpass queries).
- **Deployment:** Streamlit-only UI (no backend service required).
- **Feedback loop:** Up/downvotes saved locally and used to boost POI ranking for that destination.
- **Reliability + UX:** Caching, retries, timeouts, step trace panel, and autosave.

---

## Features

### Trip planning
- Enter a city, trip length, pace, interests, and constraints.
- The LLM generates itinerary JSON **only using POIs returned from `search_pois`** (tool-validated).

### Map + itinerary display
- Day-by-day itinerary view (Morning / Afternoon / Evening columns).
- Interactive map (pydeck) with:
  - **Scaled dots** (radius in meters + pixel clamps so dots don’t blow up at different zoom levels)
  - Day filter (All / Day 1 / Day 2 …)
  - Optional day “path” lines (morning → afternoon → evening)

### Optional RAG (Wikivoyage)
- If enabled, the app fetches Wikivoyage text at runtime and retrieves relevant chunks.
- If Wikivoyage blocks requests (403) or returns no content, the planner proceeds with `sources=[]`.

### Feedback loop
- Users can upvote/downvote itinerary POIs.
- Votes are saved locally and used as a ranking “boost” for future POI searches for the same destination.

### Don’t lose your itinerary
- Autosaves to `data/app_state.json`
- Reloads on restart so your itinerary persists across reruns/restarts.

---

## Folder structure

capstone/
 - app.py
 - requirements.txt
 - data/
   - app_state.json # created after first save/autosave
   - feedback.jsonl # created after first vote



---

## Setup

### 1) Create and activate an environment

**Conda**
```bash
conda create -n tripcapstone python=3.11 -y
conda activate tripcapstone
```

### 2) Install requirements
```
pip install -r requirements.txt
```

### 3) Run Streamlit
```bash
streamlit run app.py
```

### OpenAI key (Bring Your Own Key)

This app is BYO-key by default:

- Paste your OpenAI API key into the sidebar.
- The key is stored only in `st.session_state` (memory) for that session.
- If you uncheck **“Remember for this session”**, the app clears the key after each run.

✅ This is the simplest safe option for demos and Streamlit Community Cloud.

---

## Public API etiquette (important)

This app uses public services:

- **Nominatim (OpenStreetMap)** for geocoding  
- **Overpass API** for POIs  
- **Wikivoyage/Wikimedia** for optional RAG  

To reduce 403/429 blocks:

- Set a real email in the sidebar **User-Agent contact**.
- Avoid repeatedly spamming cities/queries (caching is enabled).
- Expect occasional rate limits; the app retries lightly and continues gracefully.

---

## How the agent works

### Tools

- `search_pois(city, interests, radius_km, limit, query)`  
  Calls OSM services to return POIs with IDs + lat/lon.

- `retrieve_guides(city, query, k)`  
  Optional: retrieves relevant chunks from Wikivoyage for grounding.

### Guardrails

- Tool schema is strict (`required` includes every property + `additionalProperties: false`).
- Itinerary validation rejects any `poi_id` that wasn’t returned by `search_pois`.

### Speed controls

- **Fast mode** reduces forced tool calls and defaults to fewer agent steps.
- You can raise `max_steps` if the model needs more turns.

---

## Troubleshooting

### “Everything disappears when I change the map dropdown”

This is fixed by rendering the itinerary/map from saved session state (outside the button handler).  
If it still happens:

- Confirm an itinerary exists (Plan tab)
- Confirm Autosave is enabled
- Don’t clear state (sidebar **“Clear saved”**)

### Dots too big/small on the map

Adjust in `render_map()`:

- `get_radius` (meters)
- `radius_min_pixels` / `radius_max_pixels`

### Wikivoyage returns 403

- Set a real email in User-Agent contact (sidebar)
- Or disable RAG (default)

### Overpass / Nominatim rate limits

- Try again after a short pause
- Reduce radius/limit
- Keep Fast mode on

---

## Deployment notes (Streamlit Community Cloud)

Recommended: **BYO-key** (users paste their own OpenAI key).  
If you prefer a shared key:

- Use Streamlit **Secrets**
- Never commit the key to the repo

---

## Suggested capstone extensions (optional)

- Add budgets (restaurants/hotels) and daily spending.
- Add travel time estimation between POIs (OSRM API).
- Add multi-city itineraries with travel days.
- Replace TF-IDF RAG with embeddings + FAISS for a stronger retrieval demo.
- Add a “save trip” library (SQLite) with multiple saved itineraries per user.

---

## License

Use however you like for learning/demo purposes.

