"""Record setup/wrapper failure without replacing the first traceback."""
import json, os, sys
from pathlib import Path
if os.geteuid() != 0:
    raise RuntimeError('trusted failure recorder requires root')
out = Path('/logs/verifier')
out.mkdir(parents=True, exist_ok=True)
record = {'execution_status': 'invalid', 'stage': sys.argv[1],
          'exit_code': int(sys.argv[2]), 'reason': 'Trusted verifier did not complete; inspect preserved stdout/stderr.'}
(out/'verifier-failure.json').write_text(json.dumps(record, indent=2)+'\n')
if not (out/'case-results.json').exists():
    (out/'case-results.json').write_text(json.dumps(dict(record, case_results=[]), indent=2)+'\n')
