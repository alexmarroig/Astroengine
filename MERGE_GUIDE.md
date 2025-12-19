# Merge guidance

This repository already includes the merge resolution you should keep:

- **Date validation**: Keep the `_parse_date_yyyy_mm_dd` helper and route calls that rely on it for Mercury retrograde alerts and other date-sensitive logic. It guarantees a single validated parse path and returns a 400 for malformed dates instead of silently re-splitting strings or re-running transit computations.
- **Mercury retrograde**: The alert helper should only compute transits once after validation; discard any variant that duplicates `compute_transits` after manual `split("-")` parsing.
- **Timezone-aware endpoints**: Preserve the IANA timezone support and offset resolution that power `/v1/time/resolve-tz`, cosmic weather, system alerts, notifications, and chart-related endpoints so DST-aware offsets remain correct.
- **Testing**: Keep the pytest coverage for system alerts, timezone resolution, and health to ensure the chosen implementation stays protected during future merges.

If you encounter a conflicting version, accept the one that aligns with the points above.
