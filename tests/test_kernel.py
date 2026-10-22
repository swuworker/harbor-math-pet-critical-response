"""Forty declared equal-weight items; 46 calls share per-item process budgets."""
import json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'test-manifest.json').read_text(encoding='utf8'))
IDS=[x['id'] for x in MANIFEST['tests']]

def numeric_structure(actual,expected):
    if isinstance(expected,list):
        assert type(actual) is list, 'Array must be a JSON list'
        assert len(actual)==len(expected), 'Array length mismatch'
        for a,e in zip(actual,expected):numeric_structure(a,e)
    else:
        assert type(actual) in (int,float), 'Scalar must be a JSON real number, not bool or string'

def validate(actual,expected,action):
    assert isinstance(actual,dict),'Output must be a dictionary'
    for key,value in expected.items():
        assert key in actual,f'Missing {key}'
        numeric_structure(actual[key],value)
        try:
            a=np.asarray(actual[key],dtype=float);e=np.asarray(value,dtype=float)
        except (TypeError,ValueError,OverflowError) as error: raise AssertionError(f'{key} is not numeric') from error
        assert a.shape==e.shape,f'{key}: shape {a.shape}, expected {e.shape}'
        assert np.isfinite(a).all(),f'{key}: non-finite entries'
        atol,rtol=(1e-9,1e-7) if action=='forward' else ((2e-5,3e-4) if key=='response' else (2e-7,2e-5))
        assert np.all(np.abs(a-e)<=atol+rtol*np.abs(e)),f'{key}: max absolute error {np.max(np.abs(a-e))}'

@pytest.mark.parametrize('case_id',IDS,ids=IDS)
def test_contract(case_id):
    """Check required numeric outputs under `abs(actual-expected)<=atol+rtol*abs(expected)`.

    Each frozen item contains at least one independently specified request;
    candidate-process failures propagate as assertions, never skipped checks.
    """
    from safe_candidate import run_payload
    fixture=json.loads((ROOT/f'{case_id}.json').read_text(encoding='utf8'))
    assert type(fixture['cases']) is list and fixture['cases'], 'Fixture must contain requests'
    budget={'elapsed':0.,'calls':0}
    for case in fixture['cases']:
        actual=run_payload(case['request'],budget)
        validate(actual,case['expected'],case['request']['action'])

