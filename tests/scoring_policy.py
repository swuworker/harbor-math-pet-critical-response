"""Exact, shared manifest weights for host scoring and the container verifier."""
from fractions import Fraction
import math


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('weight must be a positive finite rational number')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite weight')
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError('invalid rational weight') from exc
    if result <= 0:
        raise ValueError('every item/category must have strictly positive weight')
    return result


def weights(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema_version') not in ('1.0', '2.0'):
        raise ValueError('manifest schema_version must be 1.0 or 2.0')
    tests = manifest.get('tests')
    if not isinstance(tests, list) or not tests:
        raise ValueError('nonempty tests required')
    ids = []
    for item in tests:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
            raise ValueError('test needs nonempty ID')
        if not isinstance(item.get('pass_criteria'), str) or not item['pass_criteria'].strip():
            raise ValueError('pass_criteria required')
        ids.append(item['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate test ID')
    if manifest['schema_version'] == '1.0':
        if any('weight' in t and (type(t['weight']) not in (int, float) or t['weight'] != 1) for t in tests):
            raise ValueError('legacy schema 1.0 requires equal weights')
        return dict.fromkeys(ids, Fraction(1, len(ids)))
    policy = manifest.get('scoring', {})
    if not isinstance(policy, dict) or policy.get('method') != 'weighted_sum':
        raise ValueError('schema 2.0 requires weighted_sum')
    groups = policy.get('category_weights')
    if not isinstance(groups, dict) or not groups or any(not isinstance(k, str) or not k for k in groups):
        raise ValueError('nonempty named category_weights required')
    groups = {k: rational(v) for k, v in groups.items()}
    if sum(groups.values()) != 1:
        raise ValueError('category weights must sum exactly to one')
    sums = dict.fromkeys(groups, Fraction(0))
    result = {}
    for t in tests:
        group = t.get('category')
        if not isinstance(group, str) or group not in groups:
            raise ValueError('unknown category')
        w = rational(t.get('weight'))
        result[t['id']] = w
        sums[group] += w
    if sums != groups:
        raise ValueError('item weights must exactly match each frozen category budget')
    return result


def earned(manifest, cases):
    table = weights(manifest)
    if not isinstance(cases, list) or any(not isinstance(c, dict) for c in cases):
        raise ValueError('case results required')
    ids = [c.get('id') for c in cases]
    if len(ids) != len(table) or any(not isinstance(i, str) for i in ids) or set(ids) != set(table):
        raise ValueError('missing, duplicate or unknown case results')
    if any(c.get('status') not in {'pass', 'fail', 'candidate_error', 'candidate_timeout', 'skipped'} for c in cases):
        raise ValueError('invalid or unclassified case result')
    return sum((table[c['id']] for c in cases if c['status'] == 'pass'), Fraction(0))
