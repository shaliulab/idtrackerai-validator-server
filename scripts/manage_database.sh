#!/bin/bash
set -euo pipefail

DB=/flyhostel_data/videos/pe_annotations.db

if [ $# -lt 2 ]; then
    echo "usage: $0 <ls|rm> <experiment>" >&2
    echo "   e.g. $0 ls FlyHostel1_4X_2026-01-15_17-00-00" >&2
    exit 1
fi

ACTION="$1"
EXPERIMENT="$2"

# Only the first two underscores are separators; the rest belong to the date.
# Accepts either spelling as input and derives the other.
EXPERIMENT_WITHOUT_SLASH="${EXPERIMENT//\//_}"
EXPERIMENT_WITH_SLASH="$(echo "$EXPERIMENT_WITHOUT_SLASH" | sed 's|_|/|; s|_|/|')"

WHERE="experiment IN ('$EXPERIMENT_WITH_SLASH', '$EXPERIMENT_WITHOUT_SLASH')"

case "$ACTION" in
  ls)
    sqlite3 -header -column "$DB" "
      SELECT experiment, identity, start_frame, end_frame, burst_id, verdict, reviewed_at
      FROM pe_annotations
      WHERE $WHERE
      ORDER BY identity, start_frame;
    "
    ;;

  rm)
    N=$(sqlite3 "$DB" "SELECT COUNT(*) FROM pe_annotations WHERE $WHERE;")
    if [ "$N" -eq 0 ]; then
        echo "no annotations for $EXPERIMENT — nothing to delete"
        exit 0
    fi

    echo "About to delete $N annotation(s) for $EXPERIMENT:"
    sqlite3 -header -column "$DB" "
      SELECT identity, verdict, COUNT(*) AS n
      FROM pe_annotations WHERE $WHERE
      GROUP BY identity, verdict ORDER BY identity, verdict;
    "
    read -r -p "This cannot be undone. Type the experiment name to confirm: " CONFIRM
    if [ "$CONFIRM" != "$EXPERIMENT" ]; then
        echo "aborted" >&2
        exit 1
    fi

    BACKUP="${DB}.bak-$(date +%F-%H%M%S)"
    cp "$DB" "$BACKUP"
    echo "backup: $BACKUP"

    sqlite3 "$DB" <<SQL
BEGIN;
DELETE FROM pe_annotations WHERE $WHERE;
SELECT changes() AS deleted;
COMMIT;
SQL
    ;;

  *)
    echo "unknown action '$ACTION' (expected ls or rm)" >&2
    exit 1
    ;;
esac
