#!/usr/bin/env bash
for _ in $(seq 1 240); do
  curl -sf http://localhost:8000/health >/dev/null && exit 0
  sleep 15
done
echo "server not healthy" >&2
exit 1
