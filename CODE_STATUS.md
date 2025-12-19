# Code status checklist

The current `work` branch already includes the validated date parsing, timezone-aware endpoints (including Mercury retrograde alerts and daily notifications), and the pinned dependency set from the latest PR iteration. Use this quick checklist to confirm the branch stays aligned with the approved changes:

1. **Validate dependencies**
   - Keep `requirements.txt` pinned as committed (FastAPI, Pydantic, pyswisseph, OpenAI, tzdata, etc.).
   - Ensure `runtime.txt` stays on Python 3.11.x.

2. **Date and timezone path**
   - Always rely on `_parse_date_yyyy_mm_dd` for request date validation.
   - Prefer IANA timezones; `_tz_offset_for` falls back to `tz_offset_minutes` only when provided.

3. **Feature surface (should already exist)**
   - Roadmap at `/v1/system/roadmap`.
   - Timezone resolver at `/v1/time/resolve-tz`.
   - Mercury retrograde/system alerts at `/v1/alerts/system`.
   - Daily notifications feed at `/v1/notifications/daily`.
   - Sidereal/tropical selection with optional ayanamsa on chart endpoints.

4. **Health and tests**
   - `/` returns version and commit hash; `/health` stays minimal.
   - Run `python -m compileall .` and `pytest` before pushing.

If a merge conflict appears, prefer the "current" version that uses the single validated date parser and timezone-aware cache keys—the protective tests in `tests/test_alerts.py` and `tests/test_timezone.py` assert this behavior.
