from radiacode import RadiaCode
import time
import json
import os
from datetime import datetime, timezone

LOGFILE = "/home/pi/radiacode_logs/spectrum_log.jsonl"
INTERVAL = 5  # seconds

rc = RadiaCode()

print("Connected to detector")

prev_spec = rc.spectrum()

with open(LOGFILE, "a") as logfile:
    try:
        while True:
            time.sleep(INTERVAL)

            curr_spec = rc.spectrum()

            interval_counts = [
                c - p for c, p in zip(curr_spec.counts, prev_spec.counts)
            ]

            total_counts = sum(interval_counts)

            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "counts_total": total_counts,
                "spectrum": interval_counts
            }

            logfile.write(json.dumps(record) + "\n")
            logfile.flush()
            os.fsync(logfile.fileno())

            print("Logged interval:", total_counts, "counts")

            prev_spec = curr_spec

    except KeyboardInterrupt:
        print("\nStopped cleanly.")