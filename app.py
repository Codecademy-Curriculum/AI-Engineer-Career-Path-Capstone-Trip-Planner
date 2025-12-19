# app.py
"""
Trip Planner Capstone (Streamlit-only) with:
- Agent + tool calling (OpenAI Responses API)
- Live POIs via OpenStreetMap (Nominatim + Overpass)
- Optional Wikivoyage RAG (runtime fetch + TF-IDF retrieval)
- Feedback loop that boosts POI ranking per destination
- Faster defaults + visible step trace
- Nicer itinerary rendering + readable light map
- FIXES:
  1) Dots are smaller and scale nicely (meters + pixel clamps)
  2) Changing "Map: show day" no longer wipes the UI (itinerary render is outside the button)
  3) Itinerary persists: saved to disk (data/app_state.json) + reloaded on app restart
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydeck as pdk
import requests
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Config
# ============================================================
DATA_DIR = Path("data")
APP_STATE_PATH = DATA_DIR / "app_state.json"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"

DEFAULT_MODEL = "gpt-4.1-mini"
MAX_TOOL_STEPS_DEFAULT = 5  # faster default

MAP_STYLE_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
MAP_STYLE_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


# ============================================================
# Session state persistence (prevents losing itinerary on reruns/restarts)
# ============================================================
def save_app_state() -> None:
    if not st.session_state.get("autosave_enabled", True):
        return
    state = {
        "itinerary": st.session_state.get("itinerary"),
        "allowed_pois": st.session_state.get("allowed_pois"),
        "center": st.session_state.get("center"),
        "city_key": st.session_state.get("city_key"),
    }
    # Only save if we actually have an itinerary
    if not state["itinerary"] or not state["allowed_pois"]:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APP_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def load_app_state() -> None:
    if st.session_state.get("_loaded_app_state"):
        return
    st.session_state["_loaded_app_state"] = True

    if "itinerary" in st.session_state and st.session_state.get("itinerary"):
        return

    if not APP_STATE_PATH.exists():
        return

    try:
        state = json.loads(APP_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(state, dict) and state.get("itinerary") and state.get("allowed_pois"):
            st.session_state["itinerary"] = state["itinerary"]
            st.session_state["allowed_pois"] = state["allowed_pois"]
            st.session_state["center"] = state.get("center", {}) or {}
            st.session_state["city_key"] = state.get("city_key", "") or ""
    except Exception:
        # If corrupt, ignore
        return


def clear_app_state() -> None:
    st.session_state.pop("itinerary", None)
    st.session_state.pop("allowed_pois", None)
    st.session_state.pop("center", None)
    st.session_state.pop("city_key", None)
    if APP_STATE_PATH.exists():
        try:
            APP_STATE_PATH.unlink()
        except Exception:
            pass


# ============================================================
# BYO-key helpers
# ============================================================
def get_openai_client() -> OpenAI:
    key = (st.session_state.get("user_openai_key") or "").strip()
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=key)


def maybe_clear_key_after_use() -> None:
    if not st.session_state.get("remember_key", True):
        st.session_state["user_openai_key"] = ""


# ============================================================
# Trace (show steps + timings)
# ============================================================
def reset_trace() -> None:
    st.session_state["trace"] = []


def trace_event(kind: str, data: Dict[str, Any]) -> None:
    st.session_state.setdefault("trace", []).append({"ts": time.time(), "kind": kind, **data})


def render_trace() -> None:
    with st.expander("Show execution trace", expanded=False):
        trace = st.session_state.get("trace", [])
        if not trace:
            st.caption("No trace yet.")
            return
        for ev in trace:
            ts = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
            if ev["kind"] == "model_call":
                st.markdown(f"**{ts}** 🤖 model_call — step **{ev.get('step')}**")
            elif ev["kind"] == "tool_call":
                st.markdown(f"**{ts}** 🔧 tool_call `{ev.get('name')}`")
                st.code(json.dumps(ev.get("args", {}), indent=2))
            elif ev["kind"] == "tool_result":
                st.markdown(f"**{ts}** ✅ tool_result `{ev.get('name')}` in **{ev.get('elapsed_s')}s**")
            elif ev["kind"] == "tool_error":
                st.markdown(f"**{ts}** ❌ tool_error `{ev.get('name')}` in **{ev.get('elapsed_s')}s**")
                st.code(ev.get("error", ""))
            elif ev["kind"] == "note":
                st.markdown(f"**{ts}** 📝 {ev.get('message')}")


# ============================================================
# Feedback storage + boosts
# ============================================================
def append_feedback(event: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event["ts"] = time.time()
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def feedback_boost_map(city_key: str) -> Dict[str, float]:
    """
    Simple boosting:
      +0.25 per upvote, -0.35 per downvote (per destination).
    """
    if not FEEDBACK_PATH.exists():
        return {}

    pos, neg = {}, {}
    for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if (e.get("city_key") or "") != city_key:
            continue
        poi_id = e.get("poi_id")
        if not poi_id:
            continue
        if e.get("vote") == "up":
            pos[poi_id] = pos.get(poi_id, 0) + 1
        elif e.get("vote") == "down":
            neg[poi_id] = neg.get(poi_id, 0) + 1

    boost = {}
    for poi_id in set(list(pos.keys()) + list(neg.keys())):
        boost[poi_id] = 0.25 * pos.get(poi_id, 0) - 0.35 * neg.get(poi_id, 0)
    return boost


# ============================================================
# JSON extraction + validation
# ============================================================
def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def validate_itinerary_poi_ids(itin: Dict[str, Any], allowed_pois: Dict[str, Any]) -> List[str]:
    valid = set(allowed_pois.keys())
    bad = []
    for day in itin.get("days", []) or []:
        for block in ["morning", "afternoon", "evening"]:
            for item in day.get(block, []) or []:
                pid = item.get("poi_id")
                if pid and pid not in valid:
                    bad.append(pid)
    return sorted(set(bad))


def other_days_unchanged(old_itin: Dict[str, Any], new_itin: Dict[str, Any], target_day: int) -> Tuple[bool, List[int]]:
    old_days = {int(d.get("day")): d for d in (old_itin.get("days") or []) if d.get("day") is not None}
    new_days = {int(d.get("day")): d for d in (new_itin.get("days") or []) if d.get("day") is not None}

    changed = []
    for day_num, old_d in old_days.items():
        if day_num == target_day:
            continue
        new_d = new_days.get(day_num)
        if new_d is None:
            changed.append(day_num)
            continue
        if json.dumps(old_d, sort_keys=True) != json.dumps(new_d, sort_keys=True):
            changed.append(day_num)

    return (len(changed) == 0), sorted(changed)


# ============================================================
# Wikimedia headers (fixes Wikivoyage 403)
# ============================================================
def wikimedia_headers(user_agent: str) -> Dict[str, str]:
    ua = (user_agent or "").strip() or "trip-planner-capstone/1.0 (contact: unknown)"
    return {"User-Agent": ua, "Accept": "application/json"}


# ============================================================
# Nominatim geocoding + Overpass POIs (OpenStreetMap)
# ============================================================
@st.cache_data(ttl=24 * 3600)
def geocode_city(city: str, user_agent: str) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": (user_agent or "trip-planner-capstone/1.0 (contact: unknown)")}
    params = {"q": city, "format": "json", "limit": 1}
    time.sleep(1.0)  # be polite
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    top = data[0]
    return {"lat": float(top["lat"]), "lon": float(top["lon"]), "display_name": top.get("display_name", city)}


INTEREST_TO_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "outdoors": [("leisure", "park|nature_reserve|garden"), ("tourism", "viewpoint"), ("natural", "peak|wood|spring|beach|cave_entrance")],
    "museums": [("tourism", "museum|gallery")],
    "food": [("amenity", "restaurant|cafe|fast_food")],
    "coffee": [("amenity", "cafe")],
    "history": [("historic", ".+"), ("tourism", "attraction")],
    "art": [("tourism", "gallery|museum")],
    "nightlife": [("amenity", "bar|pub|nightclub")],
    "scenic": [("tourism", "viewpoint"), ("natural", "peak")],
}

DEFAULT_TAGS: List[Tuple[str, str]] = [
    ("tourism", "museum|attraction|viewpoint"),
    ("leisure", "park|garden|nature_reserve"),
    ("amenity", "cafe|restaurant"),
    ("historic", ".+"),
]


def _merge_tag_filters(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    merged: Dict[str, List[str]] = {}
    for k, v in pairs:
        merged.setdefault(k, []).append(v)
    return {k: "|".join(vs) for k, vs in merged.items()}


def _overpass_query(lat: float, lon: float, radius_m: int, tag_filters: Dict[str, str]) -> str:
    parts = []
    for k, v in tag_filters.items():
        parts.append(f'node(around:{radius_m},{lat},{lon})["{k}"~"{v}"];')
        parts.append(f'way(around:{radius_m},{lat},{lon})["{k}"~"{v}"];')
        parts.append(f'relation(around:{radius_m},{lat},{lon})["{k}"~"{v}"];')
    body = "\n".join(parts)
    return f"""
[out:json][timeout:35];
(
{body}
);
out center tags;
"""


def _category_from_tags(tags: Dict[str, str]) -> str:
    for key in ["tourism", "amenity", "leisure", "historic", "natural"]:
        if key in tags:
            return f"{key}:{tags.get(key)}"
    return "other"


@st.cache_data(ttl=6 * 3600)
def fetch_pois(city: str, interests: Tuple[str, ...], radius_km: float, limit: int, user_agent: str) -> Dict[str, Any]:
    geo = geocode_city(city, user_agent=user_agent)
    if not geo:
        return {"city_key": city.strip().lower(), "display_name": city, "lat": None, "lon": None, "pois": [], "error": "No geocode result"}

    lat, lon = geo["lat"], geo["lon"]
    display_name = geo["display_name"]
    city_key = display_name.strip().lower()

    pairs: List[Tuple[str, str]] = []
    for intr in interests:
        pairs.extend(INTEREST_TO_TAGS.get(intr, []))
    if not pairs:
        pairs = DEFAULT_TAGS

    tag_filters = _merge_tag_filters(pairs)
    radius_m = int(max(500, radius_km * 1000))
    q = _overpass_query(lat, lon, radius_m, tag_filters)

    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(OVERPASS_URL, data={"data": q}, timeout=30)
            if r.status_code == 429:
                time.sleep(1.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()

            seen = set()
            out = []
            for el in data.get("elements", []):
                tags = el.get("tags") or {}
                name = tags.get("name")
                if not name:
                    continue

                if "lat" in el and "lon" in el:
                    plat, plon = el["lat"], el["lon"]
                else:
                    c = el.get("center") or {}
                    plat, plon = c.get("lat"), c.get("lon")
                if plat is None or plon is None:
                    continue

                poi_id = f"osm_{el['type']}_{el['id']}"
                if poi_id in seen:
                    continue
                seen.add(poi_id)

                out.append(
                    {
                        "poi_id": poi_id,
                        "name": name,
                        "category": _category_from_tags(tags),
                        "lat": float(plat),
                        "lon": float(plon),
                        "url": tags.get("website") or tags.get("url") or "",
                    }
                )

            return {"city_key": city_key, "display_name": display_name, "lat": lat, "lon": lon, "pois": out[: max(1, min(limit, 200))], "error": ""}

        except Exception as e:
            last_err = e
            time.sleep(1.0 * (2 ** attempt))

    return {"city_key": city_key, "display_name": display_name, "lat": lat, "lon": lon, "pois": [], "error": str(last_err)}


# ============================================================
# Wikivoyage RAG (optional)
# ============================================================
@st.cache_data(ttl=7 * 24 * 3600)
def wikivoyage_resolve_title(city: str, user_agent: str) -> Optional[str]:
    params = {"action": "query", "list": "search", "srsearch": city, "srlimit": 1, "format": "json"}
    r = requests.get(WIKIVOYAGE_API, params=params, headers=wikimedia_headers(user_agent), timeout=15)
    if r.status_code == 403:
        return None
    r.raise_for_status()
    hits = (r.json().get("query", {}).get("search") or [])
    return hits[0]["title"] if hits else None


@st.cache_data(ttl=7 * 24 * 3600)
def wikivoyage_plaintext(title: str, user_agent: str) -> str:
    params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
    r = requests.get(WIKIVOYAGE_API, params=params, headers=wikimedia_headers(user_agent), timeout=20)
    if r.status_code == 403:
        return ""
    r.raise_for_status()
    html = r.json()["parse"]["text"]["*"]

    text = re.sub(r"<(script|style).*?>.*?</\\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<br\\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<.*?>", " ", text, flags=re.S)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 850, min_chars: int = 240) -> List[str]:
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in raw:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
            buf = p
    if buf and len(buf) >= min_chars:
        chunks.append(buf)
    return chunks


def get_city_rag_index(city: str, user_agent: str) -> Dict[str, Any]:
    cache = st.session_state.setdefault("_rag_cache", {})
    if city in cache:
        return cache[city]

    title = wikivoyage_resolve_title(city, user_agent=user_agent)
    if not title:
        cache[city] = {"title": None, "chunks": [], "vectorizer": None, "X": None}
        return cache[city]

    text = wikivoyage_plaintext(title, user_agent=user_agent)
    chunks = chunk_text(text) if text else []
    if not chunks:
        cache[city] = {"title": title, "chunks": [], "vectorizer": None, "X": None}
        return cache[city]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vectorizer.fit_transform(chunks)

    cache[city] = {"title": title, "chunks": chunks, "vectorizer": vectorizer, "X": X}
    return cache[city]


def rag_retrieve(city: str, query: str, user_agent: str, k: int = 4) -> List[Dict[str, Any]]:
    idx = get_city_rag_index(city, user_agent=user_agent)
    if not idx.get("title") or idx.get("vectorizer") is None or idx.get("X") is None:
        return []

    vectorizer: TfidfVectorizer = idx["vectorizer"]
    X = idx["X"]
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()

    topk = np.argsort(-sims)[: max(1, min(k, 8))]
    return [
        {"chunk_id": f"{idx['title']}__{int(j)}", "source": idx["title"], "text": idx["chunks"][int(j)], "score": float(sims[int(j)])}
        for j in topk
    ]


# ============================================================
# Tools (model calls these)
# ============================================================
def tool_search_pois(city: str, interests: List[str], radius_km: float, limit: int, query: str, user_agent: str) -> Dict[str, Any]:
    data = fetch_pois(city=city, interests=tuple(interests), radius_km=radius_km, limit=limit, user_agent=user_agent)
    city_key = data.get("city_key", city.strip().lower())
    boost = feedback_boost_map(city_key)

    pois = data.get("pois", [])
    if query.strip():
        q = query.lower().strip()
        for p in pois:
            p["_match"] = 1 if q in (p.get("name", "").lower()) else 0
    else:
        for p in pois:
            p["_match"] = 1
    for p in pois:
        p["_boost"] = float(boost.get(p["poi_id"], 0.0))

    pois = sorted(pois, key=lambda p: (p["_match"], p["_boost"], p.get("name", "")), reverse=True)
    pois = pois[: max(1, min(limit, 60))]

    for p in pois:
        p.pop("_match", None)
        p.pop("_boost", None)

    return {
        "city_key": city_key,
        "display_name": data.get("display_name", city),
        "center": {"lat": data.get("lat"), "lon": data.get("lon")},
        "pois": pois,
        "error": data.get("error", ""),
    }


def tool_retrieve_guides(city: str, query: str, k: int, user_agent: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"city": city, "hits": [], "note": "RAG disabled by user."}
    hits = rag_retrieve(city=city, query=query, user_agent=user_agent, k=k)
    return {"city": city, "hits": hits, "note": "If hits empty, proceed with sources=[]"}


# STRICT MODE: required must include every property key
TOOLS = [
    {
        "type": "function",
        "name": "search_pois",
        "description": "Find POIs near a city. Returns poi_id, name, category, lat/lon, url. Use poi_id in the itinerary.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "interests": {"type": "array", "items": {"type": "string"}},
                "radius_km": {"type": "number", "minimum": 1, "maximum": 50},
                "limit": {"type": "integer", "minimum": 1, "maximum": 60},
                "query": {"type": "string"},
            },
            "required": ["city", "interests", "radius_km", "limit", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "retrieve_guides",
        "description": "Retrieve relevant Wikivoyage snippets for the city (RAG). Returns chunk_id, source, text, score.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["city", "query", "k"],
            "additionalProperties": False,
        },
    },
]


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def call_tool(name: str, args: Dict[str, Any], *, user_agent: str, tool_state: Dict[str, Any], rag_enabled: bool) -> str:
    t0 = time.time()
    trace_event("tool_call", {"name": name, "args": args})

    try:
        if name == "search_pois":
            result = tool_search_pois(
                city=args["city"],
                interests=args["interests"],
                radius_km=float(args["radius_km"]),
                limit=int(args["limit"]),
                query=args["query"],
                user_agent=user_agent,
            )
            for p in result.get("pois", []):
                tool_state.setdefault("pois", {})[p["poi_id"]] = p

            tool_state["city_key"] = result.get("city_key", tool_state.get("city_key", ""))
            tool_state["display_name"] = result.get("display_name", tool_state.get("display_name", ""))
            tool_state["center"] = result.get("center", tool_state.get("center", {}))

            out = json.dumps(result, ensure_ascii=False)

        elif name == "retrieve_guides":
            result = tool_retrieve_guides(
                city=args["city"],
                query=args["query"],
                k=int(args["k"]),
                user_agent=user_agent,
                enabled=rag_enabled,
            )
            for h in result.get("hits", []):
                tool_state.setdefault("chunks", {})[h["chunk_id"]] = {"source": h.get("source", ""), "score": h.get("score", 0.0)}
            out = json.dumps(result, ensure_ascii=False)

        else:
            out = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)

        trace_event("tool_result", {"name": name, "elapsed_s": round(time.time() - t0, 3)})
        return out

    except Exception as e:
        trace_event("tool_error", {"name": name, "elapsed_s": round(time.time() - t0, 3), "error": str(e)})
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def run_trip_agent(
    client: OpenAI,
    model: str,
    user_prompt: str,
    user_agent: str,
    max_steps: int,
    rag_enabled: bool,
    status: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    tool_state: Dict[str, Any] = {"pois": {}, "chunks": {}, "center": {}}

    for step in range(1, max_steps + 1):
        trace_event("model_call", {"step": step})
        if status is not None:
            status.update(label=f"Model step {step}/{max_steps}", state="running")
            status.write("Calling model…")

        resp = client.responses.create(model=model, tools=TOOLS, input=input_items, store=False)
        input_items += resp.output

        tool_calls = [it for it in resp.output if _item_get(it, "type") == "function_call"]
        if not tool_calls:
            if status is not None:
                status.update(label="Done", state="complete")
            return resp.output_text, tool_state

        for tc in tool_calls:
            if status is not None:
                status.write(f"Tool: `{_item_get(tc,'name','')}`")

            try:
                args = json.loads(_item_get(tc, "arguments") or "{}")
            except Exception:
                args = {}

            output_str = call_tool(
                _item_get(tc, "name", ""),
                args,
                user_agent=user_agent,
                tool_state=tool_state,
                rag_enabled=rag_enabled,
            )

            input_items.append({"type": "function_call_output", "call_id": _item_get(tc, "call_id"), "output": output_str})

    raise RuntimeError("Agent hit max_steps; enable Fast mode, reduce constraints, or increase max_steps.")


# ============================================================
# Itinerary rendering (nice UI)
# ============================================================
def format_poi(pid: str, allowed: Dict[str, Any]) -> str:
    p = allowed.get(pid) or {}
    name = p.get("name") or pid
    cat = p.get("category") or ""
    return f"{name} ({cat})" if cat else name


def render_itinerary_pretty(itin: Dict[str, Any], allowed: Dict[str, Any]) -> None:
    st.subheader(itin.get("title") or "Itinerary")
    st.caption(itin.get("city", ""))

    for day in itin.get("days", []) or []:
        dnum = day.get("day")
        st.markdown(f"### Day {dnum}")
        cols = st.columns(3)
        for i, block in enumerate(["morning", "afternoon", "evening"]):
            with cols[i]:
                st.markdown(f"**{block.title()}**")
                items = day.get(block, []) or []
                if not items:
                    st.caption("—")
                for item in items[:4]:
                    pid = item.get("poi_id", "")
                    why = (item.get("why") or "").strip()
                    st.markdown(f"- **{format_poi(pid, allowed)}**  \n  {why}")

        notes = (day.get("notes") or "").strip()
        if notes:
            st.caption(notes)

        sources = day.get("sources") or []
        if sources:
            with st.expander("Sources (RAG)", expanded=False):
                for s in sources[:6]:
                    st.markdown(f"- `{s.get('chunk_id','')}` — **{s.get('source','')}**")


# ============================================================
# Map helpers (smaller dots that scale + stable UI)
# ============================================================
def itinerary_points(itin: Dict[str, Any], allowed: Dict[str, Any], day_filter: Optional[int] = None) -> List[Dict[str, Any]]:
    pts = []
    for day in itin.get("days", []) or []:
        dnum = int(day.get("day", -1))
        if day_filter is not None and dnum != day_filter:
            continue
        for block in ["morning", "afternoon", "evening"]:
            for item in day.get(block, []) or []:
                pid = item.get("poi_id")
                if not pid:
                    continue
                p = allowed.get(pid)
                if not p:
                    continue
                pts.append(
                    {
                        "day": dnum,
                        "block": block,
                        "poi_id": pid,
                        "name": p.get("name", pid),
                        "category": p.get("category", ""),
                        "lat": float(p.get("lat")),
                        "lon": float(p.get("lon")),
                    }
                )
    return pts


def itinerary_paths(itin: Dict[str, Any], allowed: Dict[str, Any], day_filter: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for day in itin.get("days", []) or []:
        dnum = int(day.get("day", -1))
        if day_filter is not None and dnum != day_filter:
            continue

        coords = []
        for block in ["morning", "afternoon", "evening"]:
            items = day.get(block, []) or []
            if not items:
                continue
            pid = (items[0] or {}).get("poi_id")
            p = allowed.get(pid) if pid else None
            if p and p.get("lat") is not None and p.get("lon") is not None:
                coords.append([float(p["lon"]), float(p["lat"])])

        if len(coords) >= 2:
            out.append({"day": dnum, "path": coords})
    return out


def _approx_zoom(points: List[Dict[str, Any]]) -> int:
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    span = max(lat_span, lon_span)
    if span < 0.01:
        return 14
    if span < 0.03:
        return 13
    if span < 0.08:
        return 12
    if span < 0.18:
        return 11
    if span < 0.35:
        return 10
    return 9


def render_map(points: List[Dict[str, Any]], paths: List[Dict[str, Any]], center: Dict[str, Any], dark: bool) -> None:
    if not points:
        st.info("No mappable POIs yet.")
        return

    clat = center.get("lat")
    clon = center.get("lon")
    if clat is None or clon is None:
        clat = float(np.mean([p["lat"] for p in points]))
        clon = float(np.mean([p["lon"] for p in points]))

    zoom = _approx_zoom(points)
    view_state = pdk.ViewState(latitude=clat, longitude=clon, zoom=zoom, pitch=0)

    # ✅ FIX: Smaller dots that still scale with zoom:
    # - radius is in meters (so it scales naturally)
    # - clamp pixel sizes so it never gets huge or vanishes
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_radius=35,             # meters (smaller than before)
        radius_min_pixels=3,       # never disappear
        radius_max_pixels=10,      # never huge
        get_fill_color=[0, 120, 255, 190],
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=1,
        pickable=True,
    )

    layers = [point_layer]

    if paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_color=[255, 80, 80, 170],
                width_min_pixels=2,
                width_max_pixels=4,
                pickable=False,
            )
        )

    tooltip = {"text": "{name}\nDay {day} • {block}\n{category}\n{poi_id}"}
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=(MAP_STYLE_DARK if dark else MAP_STYLE_LIGHT),
    )
    st.pydeck_chart(deck, use_container_width=True)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Trip Planner Capstone", layout="wide")

# Load persisted itinerary (if present)
load_app_state()

st.title("Trip Planner Capstone (Agent + POIs + Optional RAG + Feedback)")
st.caption("BYO-key: Your key is stored in this session’s memory only (st.session_state).")

with st.sidebar:
    st.header("Settings")

    st.subheader("OpenAI (Bring your own key)")
    st.text_input("OpenAI API key", type="password", key="user_openai_key", help="Starts with sk- ...")
    st.checkbox("Remember for this session", value=True, key="remember_key")
    if st.button("Clear key"):
        st.session_state["user_openai_key"] = ""
        st.toast("Cleared key")

    st.subheader("Persistence")
    st.checkbox("Autosave itinerary locally (data/app_state.json)", value=True, key="autosave_enabled")
    cols = st.columns(2)
    if cols[0].button("Save now"):
        save_app_state()
        st.toast("Saved.")
    if cols[1].button("Clear saved"):
        clear_app_state()
        st.toast("Cleared.")

    st.subheader("Speed / UX")
    fast_mode = st.checkbox("Fast mode (fewer tool calls)", value=True)
    rag_enabled = st.checkbox("Enable Wikivoyage RAG (slower, may 403)", value=False)
    show_steps = st.checkbox("Show step trace", value=True)

    model = st.text_input("Model", value=DEFAULT_MODEL)
    max_steps = st.slider("Max tool steps", min_value=3, max_value=12, value=(5 if fast_mode else 8))

    st.subheader("External API etiquette")
    user_agent_email = st.text_input(
        "User-Agent contact (recommended)",
        value="your-email@example.com",
        help="Used in User-Agent for Nominatim + Wikivoyage. Put a real email to reduce 403 blocks.",
    )
    if user_agent_email.strip().lower() in {"your-email@example.com", "you@example.com", "email@example.com"}:
        st.warning("Set a real email in User-Agent contact to reduce 403 blocks from public APIs.")
    user_agent = f"trip-planner-capstone/1.0 ({user_agent_email})"

    st.subheader("Map")
    dark_map = st.checkbox("Dark map style", value=False)

tab1, tab2, tab3 = st.tabs(["Plan Trip", "Refine / Regenerate", "Feedback"])

# -------------------------
# Tab 1: Plan (IMPORTANT: itinerary display is OUTSIDE the button handler)
# -------------------------
with tab1:
    colA, colB = st.columns(2)
    with colA:
        city = st.text_input("Destination city", value="Santa Fe, NM")
        days = st.slider("Trip length (days)", 1, 7, 3)
        pace = st.selectbox("Pace", ["relaxed", "balanced", "packed"], index=1)
        radius_km = st.slider("POI search radius (km)", 1, 30, 8)
    with colB:
        interests = st.multiselect(
            "Interests (drives POI tags)",
            ["outdoors", "food", "coffee", "museums", "history", "art", "nightlife", "scenic"],
            default=["outdoors", "food"],
        )
        constraints = st.text_area("Constraints", value="No early mornings. Prefer 1–2 big activities/day.")
        notes = st.text_area("Extra notes", value="Include at least one iconic highlight and one hidden gem.")

    generate_clicked = st.button("Generate itinerary", type="primary")

    if generate_clicked:
        reset_trace()
        client = get_openai_client()

        poi_limit = 40 if fast_mode else 30
        rag_k = 4

        if fast_mode:
            tool_rules = f"""
Tooling rules (FAST):
- Call search_pois ONCE at the start with:
  city="{city}", interests={interests}, radius_km={radius_km}, limit={poi_limit}, query=""
- Only call search_pois again if you cannot satisfy a specific slot (keep total calls <= 2).
- If RAG is enabled, call retrieve_guides once with a helpful query (k={rag_k}). If disabled or empty, proceed with sources=[].
"""
        else:
            tool_rules = f"""
Tooling rules:
- Call search_pois at least 2 times with different queries (keep total calls <= 4).
- If RAG is enabled, call retrieve_guides at least once (k={rag_k}). If disabled or empty, proceed with sources=[].
"""

        user_prompt = f"""
You are a trip-planning assistant.

Create a {days}-day itinerary for: {city}
Pace: {pace}
Interests: {interests}
Constraints: {constraints}
Notes: {notes}

Hard rules:
1) You MUST ONLY use POIs you saw from search_pois, referencing them by poi_id.
2) Output MUST be valid JSON (no markdown, no commentary).
3) Keep each block (morning/afternoon/evening) to 1–2 items max. Prefer concise "why".

{tool_rules}

JSON schema:
{{
  "title": str,
  "city": str,
  "days": [
    {{
      "day": int,
      "morning": [{{"poi_id": str, "why": str}}],
      "afternoon": [{{"poi_id": str, "why": str}}],
      "evening": [{{"poi_id": str, "why": str}}],
      "notes": str,
      "sources": [{{"chunk_id": str, "source": str}}]
    }}
  ]
}}
"""

        status = st.status("Starting…", expanded=True) if show_steps else None
        try:
            with st.spinner("Planning…"):
                raw, tool_state = run_trip_agent(
                    client=client,
                    model=model,
                    user_prompt=user_prompt,
                    user_agent=user_agent,
                    max_steps=max_steps,
                    rag_enabled=rag_enabled,
                    status=status,
                )
        finally:
            maybe_clear_key_after_use()
            if status is not None:
                status.update(label="Done", state="complete")

        if show_steps:
            render_trace()

        # Parse/validate and SAVE to session_state
        try:
            itin = extract_json(raw)
            allowed = dict(tool_state.get("pois", {}))
            bad = validate_itinerary_poi_ids(itin, allowed)
            if bad:
                st.error(f"Itinerary referenced unknown poi_id(s) (not returned by tools): {bad}")
                st.stop()
            if not allowed:
                st.error("No POIs were returned by tools. The model may not have called search_pois.")
                st.stop()

            st.session_state["itinerary"] = itin
            st.session_state["allowed_pois"] = allowed
            st.session_state["city_key"] = tool_state.get("city_key", city.strip().lower())
            st.session_state["center"] = tool_state.get("center", {})

            save_app_state()
            st.success("Itinerary saved.")

        except Exception as e:
            st.error(f"Could not parse/validate itinerary JSON: {e}")
            with st.expander("Raw output"):
                st.code(raw)

    # ✅ ALWAYS render saved itinerary if present (so map dropdown reruns don't wipe everything)
    itin = st.session_state.get("itinerary")
    allowed = st.session_state.get("allowed_pois", {})
    center = st.session_state.get("center", {})

    if itin and allowed:
        st.divider()
        render_itinerary_pretty(itin, allowed)

        st.subheader("Map")
        day_options = ["All"] + [d.get("day") for d in itin.get("days", []) if d.get("day") is not None]

        # ✅ FIX: if options change, reset invalid selection key so UI doesn't blank/error
        if "map_day_filter_plan" in st.session_state and st.session_state["map_day_filter_plan"] not in day_options:
            del st.session_state["map_day_filter_plan"]

        day_filter = st.selectbox("Map: show day", options=day_options, index=0, key="map_day_filter_plan")
        df = None if day_filter == "All" else int(day_filter)

        pts = itinerary_points(itin, allowed, df)
        paths = itinerary_paths(itin, allowed, df)
        render_map(pts, paths, center, dark=dark_map)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.download_button(
                "Download itinerary.json",
                data=json.dumps(itin, ensure_ascii=False, indent=2),
                file_name="itinerary.json",
                mime="application/json",
            )
        with c2:
            with st.expander("Raw / JSON output", expanded=False):
                st.json(itin)
    else:
        st.info("No saved itinerary yet. Generate one above (or enable Autosave and reload).")

# -------------------------
# Tab 2: Refine / Regenerate
# -------------------------
with tab2:
    itin = st.session_state.get("itinerary")
    allowed_prev = st.session_state.get("allowed_pois", {})
    center_prev = st.session_state.get("center", {})

    if not itin or not allowed_prev:
        st.info("Generate an itinerary first.")
    else:
        st.subheader("Current itinerary")
        render_itinerary_pretty(itin, allowed_prev)

        st.subheader("Map")
        day_options = ["All"] + [d.get("day") for d in itin.get("days", []) if d.get("day") is not None]
        if "map_day_filter_refine" in st.session_state and st.session_state["map_day_filter_refine"] not in day_options:
            del st.session_state["map_day_filter_refine"]
        day_filter = st.selectbox("Map: show day", options=day_options, index=0, key="map_day_filter_refine")
        df = None if day_filter == "All" else int(day_filter)
        pts = itinerary_points(itin, allowed_prev, df)
        paths = itinerary_paths(itin, allowed_prev, df)
        render_map(pts, paths, center_prev, dark=dark_map)

        st.divider()
        colL, colR = st.columns(2)

        with colL:
            st.markdown("### Refine entire itinerary")
            refine = st.text_input("Refinement request", value="Make it more outdoorsy, keep evenings chill, and reduce walking.")
            if st.button("Apply refinement"):
                reset_trace()
                client = get_openai_client()
                status = st.status("Refining…", expanded=True) if show_steps else None
                try:
                    refine_prompt = f"""
You will edit an existing itinerary JSON for: {itin.get("city","")}.

Hard rules:
- Keep the same JSON schema.
- You MUST ONLY use poi_id values you have obtained via search_pois.
- If you need alternatives, call search_pois (keep total calls <= 2 in fast mode).
- If RAG enabled, you may call retrieve_guides once; if empty/disabled, sources=[] is fine.
- Output JSON only.

Refinement request: {refine}

Existing JSON:
{json.dumps(itin, ensure_ascii=False)}
"""
                    with st.spinner("Refining…"):
                        raw2, tool_state2 = run_trip_agent(
                            client=client,
                            model=model,
                            user_prompt=refine_prompt,
                            user_agent=user_agent,
                            max_steps=max_steps,
                            rag_enabled=rag_enabled,
                            status=status,
                        )
                finally:
                    maybe_clear_key_after_use()
                    if status is not None:
                        status.update(label="Done", state="complete")

                if show_steps:
                    render_trace()

                try:
                    itin2 = extract_json(raw2)
                    allowed = dict(allowed_prev)
                    allowed.update(tool_state2.get("pois", {}))

                    bad = validate_itinerary_poi_ids(itin2, allowed)
                    if bad:
                        st.error(f"Refined itinerary referenced unknown poi_id(s): {bad}")
                        st.stop()

                    st.session_state["itinerary"] = itin2
                    st.session_state["allowed_pois"] = allowed
                    st.session_state["center"] = tool_state2.get("center", center_prev)
                    st.session_state["city_key"] = tool_state2.get("city_key", st.session_state.get("city_key", ""))

                    save_app_state()
                    st.success("Itinerary updated and saved.")
                    render_itinerary_pretty(itin2, allowed)

                except Exception as e:
                    st.error(f"Could not parse/validate refined JSON: {e}")
                    with st.expander("Raw output"):
                        st.code(raw2)

        with colR:
            st.markdown("### Regenerate just one day")
            day_nums = [int(d.get("day")) for d in (itin.get("days") or []) if d.get("day") is not None]
            if not day_nums:
                st.info("No day numbers found in itinerary.")
            else:
                target_day = st.selectbox("Which day?", options=sorted(day_nums), index=0)
                day_request = st.text_area(
                    "Day-specific request",
                    value="Swap in a different afternoon activity and add a cozy dinner option; keep walking minimal.",
                    height=120,
                )
                if st.button("Regenerate Day"):
                    reset_trace()
                    client = get_openai_client()
                    status = st.status("Regenerating…", expanded=True) if show_steps else None
                    try:
                        regen_prompt = f"""
You will edit an existing itinerary JSON for: {itin.get("city","")}.

Goal: ONLY modify the content of day == {target_day}. All other days must remain EXACTLY unchanged.
If you need alternatives, call search_pois (keep total calls <= 2 in fast mode).
If RAG enabled, you may call retrieve_guides once; if empty/disabled, sources=[] is fine.

Hard rules:
- Keep the same JSON schema.
- You MUST ONLY use poi_id values you have obtained via search_pois.
- Output JSON only.

Day-specific request: {day_request}

Existing JSON:
{json.dumps(itin, ensure_ascii=False)}
"""
                        with st.spinner(f"Regenerating Day {target_day}…"):
                            raw3, tool_state3 = run_trip_agent(
                                client=client,
                                model=model,
                                user_prompt=regen_prompt,
                                user_agent=user_agent,
                                max_steps=max_steps,
                                rag_enabled=rag_enabled,
                                status=status,
                            )
                    finally:
                        maybe_clear_key_after_use()
                        if status is not None:
                            status.update(label="Done", state="complete")

                    if show_steps:
                        render_trace()

                    try:
                        itin3 = extract_json(raw3)
                        allowed = dict(allowed_prev)
                        allowed.update(tool_state3.get("pois", {}))

                        bad = validate_itinerary_poi_ids(itin3, allowed)
                        if bad:
                            st.error(f"Day-regenerated itinerary referenced unknown poi_id(s): {bad}")
                            st.stop()

                        ok, changed_days = other_days_unchanged(itin, itin3, target_day=int(target_day))
                        if not ok:
                            st.error(f"Model changed other day(s) too: {changed_days}. Not applying.")
                            st.stop()

                        st.session_state["itinerary"] = itin3
                        st.session_state["allowed_pois"] = allowed
                        st.session_state["center"] = tool_state3.get("center", center_prev)
                        st.session_state["city_key"] = tool_state3.get("city_key", st.session_state.get("city_key", ""))

                        save_app_state()
                        st.success(f"Updated Day {target_day} and saved.")
                        render_itinerary_pretty(itin3, allowed)

                    except Exception as e:
                        st.error(f"Could not parse/validate day-regenerated JSON: {e}")
                        with st.expander("Raw output"):
                            st.code(raw3)

# -------------------------
# Tab 3: Feedback
# -------------------------
with tab3:
    itin = st.session_state.get("itinerary")
    allowed_pois = st.session_state.get("allowed_pois", {})
    city_key = st.session_state.get("city_key", "")

    if not itin or not allowed_pois:
        st.info("Generate an itinerary first.")
    else:
        st.subheader("Vote on places (feeds future ranking)")
        st.caption("Upvoted POIs get boosted in future search_pois results for the same destination.")

        referenced: List[str] = []
        for day in itin.get("days", []) or []:
            for block in ["morning", "afternoon", "evening"]:
                for item in day.get(block, []) or []:
                    if item.get("poi_id"):
                        referenced.append(item["poi_id"])
        referenced = list(dict.fromkeys(referenced))

        if not referenced:
            st.info("No POIs found in itinerary JSON.")
        else:
            for pid in referenced:
                p = allowed_pois.get(pid, {})
                name = p.get("name", pid)
                cat = p.get("category", "")
                url = p.get("url", "")

                cols = st.columns([5, 1, 1])
                with cols[0]:
                    line = f"**{name}**  \n`{pid}`  \n{cat}"
                    if url:
                        line += f"  \n{url}"
                    st.markdown(line)

                if cols[1].button("👍", key=f"up_{pid}"):
                    append_feedback({"city_key": city_key, "poi_id": pid, "vote": "up"})
                    st.toast(f"Upvoted: {name}")

                if cols[2].button("👎", key=f"down_{pid}"):
                    append_feedback({"city_key": city_key, "poi_id": pid, "vote": "down"})
                    st.toast(f"Downvoted: {name}")
