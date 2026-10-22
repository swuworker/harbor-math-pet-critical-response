"""JSON stdin/stdout entry point; candidate implementation lives in src."""
import json
import sys
sys.path.insert(0, "/app/src")
from kernel import solve
if __name__ == "__main__":
    result = solve(json.load(sys.stdin))
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
