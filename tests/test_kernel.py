"""Independent numerical behavior checks and separate schema/resource checks."""
import json
from pathlib import Path
import numpy as np
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
        try:
            represented = float(actual)
        except (OverflowError, ValueError, TypeError) as error:
            raise AssertionError('Candidate number cannot be represented as a finite real') from error
        assert np.isfinite(represented), 'Candidate number must be finite'

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


def run_cases(case_id, budget_only=False):
    from safe_candidate import run_payload, CandidateFailure
    fixture=json.loads((ROOT/f'{case_id}.json').read_text(encoding='utf8'))
    assert 1 <= len(fixture['cases']) <= 3
    budget={'elapsed':0., 'calls':0}
    observations=[]
    for case in fixture['cases']:
        try:
            actual=run_payload(case['request'],budget,budget_only=budget_only)
        except CandidateFailure as error:
            actual=error
        observations.append((case,actual))
    return observations,budget

def require_output(value):
    assert not isinstance(value,Exception), str(value)
    assert type(value) is dict, 'Output must be a dictionary'

def check_field(actual, expected, field, action):
    require_output(actual)
    validate(actual,{field:expected[field]},action)

def check_schema(actual, expected):
    require_output(actual)
    for key, value in expected.items():
        assert key in actual, f'Missing {key}'
        numeric_structure(actual[key],value)
        assert np.isfinite(np.asarray(actual[key],dtype=float)).all(), f'{key}: nonfinite'

def test_a01_unit_pixel_chord():
    """A01: 单位单像素弦长; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A01')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a02_oblique_pixel_chords():
    """A02: 斜射线跨多个像素; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A02')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a03_translated_rectangular_pixels():
    """A03: 非方形像素及坐标平移; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A03')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'
    case_1, actual_1 = observations[1]
    require_output(actual_1)
    assert 'ray_matrix' in actual_1, 'Missing ray_matrix'
    numeric_structure(actual_1['ray_matrix'], case_1['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_1['ray_matrix'], dtype=float)-np.asarray(case_1['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_1['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'system_matrix' in actual_1, 'Missing system_matrix'
    numeric_structure(actual_1['system_matrix'], case_1['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_1['system_matrix'], dtype=float)-np.asarray(case_1['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_1['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'mean' in actual_1, 'Missing mean'
    numeric_structure(actual_1['mean'], case_1['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_1['mean'], dtype=float)-np.asarray(case_1['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_1['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'adjoint' in actual_1, 'Missing adjoint'
    numeric_structure(actual_1['adjoint'], case_1['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_1['adjoint'], dtype=float)-np.asarray(case_1['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_1['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a04_whole_line_attenuation():
    """A04: 整条符合线衰减; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A04')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a05_asymmetric_psf_adjoint():
    """A05: 非对称PSF伴随; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A05')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a06_blur_projection_order():
    """A06: 模糊/投影/衰减次序; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A06')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a07_nonuniform_sensitivity():
    """A07: 非均匀效率和非均匀灵敏度; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A07')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_a08_signed_cotangent_background():
    """A08: 有符号伴随及背景排除; independently frozen expected values for each stated output."""
    observations, _ = run_cases('A08')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'ray_matrix' in actual_0, 'Missing ray_matrix'
    numeric_structure(actual_0['ray_matrix'], case_0['expected']['ray_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['ray_matrix'], dtype=float)-np.asarray(case_0['expected']['ray_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['ray_matrix'], dtype=float))), 'ray_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'system_matrix' in actual_0, 'Missing system_matrix'
    numeric_structure(actual_0['system_matrix'], case_0['expected']['system_matrix'])
    assert np.all(np.abs(np.asarray(actual_0['system_matrix'], dtype=float)-np.asarray(case_0['expected']['system_matrix'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['system_matrix'], dtype=float))), 'system_matrix violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'mean' in actual_0, 'Missing mean'
    numeric_structure(actual_0['mean'], case_0['expected']['mean'])
    assert np.all(np.abs(np.asarray(actual_0['mean'], dtype=float)-np.asarray(case_0['expected']['mean'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['mean'], dtype=float))), 'mean violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'adjoint' in actual_0, 'Missing adjoint'
    numeric_structure(actual_0['adjoint'], case_0['expected']['adjoint'])
    assert np.all(np.abs(np.asarray(actual_0['adjoint'], dtype=float)-np.asarray(case_0['expected']['adjoint'], dtype=float)) <= 1e-09+1e-07*np.abs(np.asarray(case_0['expected']['adjoint'], dtype=float))), 'adjoint violates the public elementwise tolerance'

def test_b01_unregularized_mirror_step():
    """B01: 无正则单步; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B01')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_b02_nonuniform_mirror_metric():
    """B02: 非均匀灵敏度单步; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B02')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_b03_zero_count_background():
    """B03: 背景与零观测; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B03')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_b04_quadratic_proximal_step():
    """B04: 纯二次正则; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B04')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_b05_step_size_scaling():
    """B05: 不同步长缩放; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B05')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_b06_finite_iteration_history():
    """B06: 有限步与最终最优解区别; independently frozen expected values for each stated output."""
    observations, _ = run_cases('B06')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c01_isotropic_group_coupling():
    """C01: 二维各向同性组; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C01')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c02_boundary_spacing_tv():
    """C02: 网格边界与间距; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C02')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c03_weighted_tv_groups():
    """C03: 非均匀TV权重; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C03')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c04_strictly_fused_plateau():
    """C04: 严格融合的平坦块; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C04')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c05_unfused_regions():
    """C05: 两个未融合像素区域; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C05')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c06_quadratic_tv_interaction():
    """C06: 二次与TV共同作用; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C06')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c07_right_proximal_coupling():
    """C07: 右近端与单步滞后区别; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C07')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_c08_critical_fusion_image():
    """C08: 临界融合点基准图像; independently frozen expected values for each stated output."""
    observations, _ = run_cases('C08')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'history' in actual_0, 'Missing history'
    numeric_structure(actual_0['history'], case_0['expected']['history'])
    assert np.all(np.abs(np.asarray(actual_0['history'], dtype=float)-np.asarray(case_0['expected']['history'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['history'], dtype=float))), 'history violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'objective' in actual_0, 'Missing objective'
    numeric_structure(actual_0['objective'], case_0['expected']['objective'])
    assert np.all(np.abs(np.asarray(actual_0['objective'], dtype=float)-np.asarray(case_0['expected']['objective'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['objective'], dtype=float))), 'objective violates the public elementwise tolerance'

def test_d01_smooth_initial_response():
    """D01: 光滑单像素初值响应; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D01')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d02_attenuation_metric_response():
    """D02: 衰减改变A和镜像度量; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D02')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d03_background_response():
    """D03: 背景方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D03')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d04_zero_count_data_response():
    """D04: 数据方向含基准零计数; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D04')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d05_critical_tv_increase():
    """D05: 临界融合增强TV方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D05')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d06_critical_tv_decrease():
    """D06: 临界融合减弱TV方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D06')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d07_nonzero_group_curvature():
    """D07: 非零二维组曲率; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D07')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d08_fused_block_response():
    """D08: 严格融合块内部方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D08')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d09_critical_group_feasibility():
    """D09: 临界二维组的可行方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D09')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d10_chained_step_response():
    """D10: 多步链式依赖; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D10')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d11_joint_parameter_response():
    """D11: 联合参数方向; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D11')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'
    case_1, actual_1 = observations[1]
    require_output(actual_1)
    assert 'image' in actual_1, 'Missing image'
    numeric_structure(actual_1['image'], case_1['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_1['image'], dtype=float)-np.asarray(case_1['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_1['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'response' in actual_1, 'Missing response'
    numeric_structure(actual_1['response'], case_1['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_1['response'], dtype=float)-np.asarray(case_1['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_1['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'
    case_2, actual_2 = observations[2]
    require_output(actual_2)
    assert 'image' in actual_2, 'Missing image'
    numeric_structure(actual_2['image'], case_2['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_2['image'], dtype=float)-np.asarray(case_2['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_2['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_2)
    assert 'response' in actual_2, 'Missing response'
    numeric_structure(actual_2['response'], case_2['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_2['response'], dtype=float)-np.asarray(case_2['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_2['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_d12_zero_step_response():
    """D12: K=0方向语义; independently frozen expected values for each stated output."""
    observations, _ = run_cases('D12')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e01_blur_attenuation_interaction():
    """E01: 非对称PSF与变衰减; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E01')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e02_mixed_regularizers_efficiency():
    """E02: 混合正则与非均匀效率; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E02')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e03_early_critical_step():
    """E03: 较早一步临界融合影响后续; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E03')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e04_adjacent_critical_groups():
    """E04: 二维组边界与相邻边同时临界; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E04')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'
    case_1, actual_1 = observations[1]
    require_output(actual_1)
    assert 'image' in actual_1, 'Missing image'
    numeric_structure(actual_1['image'], case_1['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_1['image'], dtype=float)-np.asarray(case_1['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_1['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'response' in actual_1, 'Missing response'
    numeric_structure(actual_1['response'], case_1['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_1['response'], dtype=float)-np.asarray(case_1['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_1['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e05_axis_ray_permutation():
    """E05: 网格轴互换与射线重排; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E05')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'
    case_1, actual_1 = observations[1]
    require_output(actual_1)
    assert 'image' in actual_1, 'Missing image'
    numeric_structure(actual_1['image'], case_1['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_1['image'], dtype=float)-np.asarray(case_1['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_1['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'response' in actual_1, 'Missing response'
    numeric_structure(actual_1['response'], case_1['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_1['response'], dtype=float)-np.asarray(case_1['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_1['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_e06_positive_direction_homogeneity():
    """E06: 联合方向的正齐次性及一般响应; independently frozen expected values for each stated output."""
    observations, _ = run_cases('E06')
    case_0, actual_0 = observations[0]
    require_output(actual_0)
    assert 'image' in actual_0, 'Missing image'
    numeric_structure(actual_0['image'], case_0['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_0['image'], dtype=float)-np.asarray(case_0['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_0['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_0)
    assert 'response' in actual_0, 'Missing response'
    numeric_structure(actual_0['response'], case_0['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_0['response'], dtype=float)-np.asarray(case_0['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_0['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'
    case_1, actual_1 = observations[1]
    require_output(actual_1)
    assert 'image' in actual_1, 'Missing image'
    numeric_structure(actual_1['image'], case_1['expected']['image'])
    assert np.all(np.abs(np.asarray(actual_1['image'], dtype=float)-np.asarray(case_1['expected']['image'], dtype=float)) <= 2e-07+2e-05*np.abs(np.asarray(case_1['expected']['image'], dtype=float))), 'image violates the public elementwise tolerance'
    require_output(actual_1)
    assert 'response' in actual_1, 'Missing response'
    numeric_structure(actual_1['response'], case_1['expected']['response'])
    assert np.all(np.abs(np.asarray(actual_1['response'], dtype=float)-np.asarray(case_1['expected']['response'], dtype=float)) <= 2e-05+0.0003*np.abs(np.asarray(case_1['expected']['response'], dtype=float))), 'response violates the public elementwise tolerance'

def test_f01_forward_matrix_vectors_format():
    """Finite real JSON field types and dimensions for forward_matrix_vectors; no value comparison."""
    observations, _ = run_cases('A05')
    for case, actual in observations:
        assert not isinstance(actual, Exception), str(actual)
        assert type(actual) is dict, 'Output must be a JSON object'
        for field, expected in case['expected'].items():
            assert field in actual, f'Missing {field}'
            pending = [(actual[field], expected)]
            while pending:
                value, shape = pending.pop()
                if isinstance(shape, list):
                    assert type(value) is list, f'{field} must use JSON arrays'
                    assert len(value) == len(shape), f'{field} has incorrect dimensions'
                    pending.extend(zip(value, shape))
                else:
                    assert type(value) in (int, float), f'{field} must contain real numbers, not booleans or strings'
                    try:
                        represented = float(value)
                    except (OverflowError, ValueError, TypeError) as error:
                        raise AssertionError(f'{field} cannot be represented as a finite real') from error
                    assert np.isfinite(represented), f'{field} contains a nonfinite number'

def test_p01_forward_matrix_vectors_performance():
    """Successful processes within the public cumulative time and stdout budget for forward_matrix_vectors."""
    observations, budget = run_cases('A05', budget_only=True)
    assert budget['calls'] == len(observations)
    assert budget['elapsed'] <= 10.0
    for _, metrics in observations:
        require_output(metrics)
        assert metrics['returncode'] == 0
        assert metrics['stdout_bytes'] <= 8388608

def test_f02_finite_history_objective_format():
    """Finite real JSON field types and dimensions for finite_history_objective; no value comparison."""
    observations, _ = run_cases('B06')
    for case, actual in observations:
        assert not isinstance(actual, Exception), str(actual)
        assert type(actual) is dict, 'Output must be a JSON object'
        for field, expected in case['expected'].items():
            assert field in actual, f'Missing {field}'
            pending = [(actual[field], expected)]
            while pending:
                value, shape = pending.pop()
                if isinstance(shape, list):
                    assert type(value) is list, f'{field} must use JSON arrays'
                    assert len(value) == len(shape), f'{field} has incorrect dimensions'
                    pending.extend(zip(value, shape))
                else:
                    assert type(value) in (int, float), f'{field} must contain real numbers, not booleans or strings'
                    try:
                        represented = float(value)
                    except (OverflowError, ValueError, TypeError) as error:
                        raise AssertionError(f'{field} cannot be represented as a finite real') from error
                    assert np.isfinite(represented), f'{field} contains a nonfinite number'

def test_p02_finite_history_objective_performance():
    """Successful processes within the public cumulative time and stdout budget for finite_history_objective."""
    observations, budget = run_cases('B06', budget_only=True)
    assert budget['calls'] == len(observations)
    assert budget['elapsed'] <= 10.0
    for _, metrics in observations:
        require_output(metrics)
        assert metrics['returncode'] == 0
        assert metrics['stdout_bytes'] <= 8388608

def test_f03_zero_step_direction_format():
    """Finite real JSON field types and dimensions for zero_step_direction; no value comparison."""
    observations, _ = run_cases('D12')
    for case, actual in observations:
        assert not isinstance(actual, Exception), str(actual)
        assert type(actual) is dict, 'Output must be a JSON object'
        for field, expected in case['expected'].items():
            assert field in actual, f'Missing {field}'
            pending = [(actual[field], expected)]
            while pending:
                value, shape = pending.pop()
                if isinstance(shape, list):
                    assert type(value) is list, f'{field} must use JSON arrays'
                    assert len(value) == len(shape), f'{field} has incorrect dimensions'
                    pending.extend(zip(value, shape))
                else:
                    assert type(value) in (int, float), f'{field} must contain real numbers, not booleans or strings'
                    try:
                        represented = float(value)
                    except (OverflowError, ValueError, TypeError) as error:
                        raise AssertionError(f'{field} cannot be represented as a finite real') from error
                    assert np.isfinite(represented), f'{field} contains a nonfinite number'

def test_p03_zero_step_direction_performance():
    """Successful processes within the public cumulative time and stdout budget for zero_step_direction."""
    observations, budget = run_cases('D12', budget_only=True)
    assert budget['calls'] == len(observations)
    assert budget['elapsed'] <= 10.0
    for _, metrics in observations:
        require_output(metrics)
        assert metrics['returncode'] == 0
        assert metrics['stdout_bytes'] <= 8388608

def test_f04_critical_direction_format():
    """Finite real JSON field types and dimensions for critical_direction; no value comparison."""
    observations, _ = run_cases('D09')
    for case, actual in observations:
        assert not isinstance(actual, Exception), str(actual)
        assert type(actual) is dict, 'Output must be a JSON object'
        for field, expected in case['expected'].items():
            assert field in actual, f'Missing {field}'
            pending = [(actual[field], expected)]
            while pending:
                value, shape = pending.pop()
                if isinstance(shape, list):
                    assert type(value) is list, f'{field} must use JSON arrays'
                    assert len(value) == len(shape), f'{field} has incorrect dimensions'
                    pending.extend(zip(value, shape))
                else:
                    assert type(value) in (int, float), f'{field} must contain real numbers, not booleans or strings'
                    try:
                        represented = float(value)
                    except (OverflowError, ValueError, TypeError) as error:
                        raise AssertionError(f'{field} cannot be represented as a finite real') from error
                    assert np.isfinite(represented), f'{field} contains a nonfinite number'

def test_p04_critical_direction_performance():
    """Successful processes within the public cumulative time and stdout budget for critical_direction."""
    observations, budget = run_cases('D09', budget_only=True)
    assert budget['calls'] == len(observations)
    assert budget['elapsed'] <= 10.0
    for _, metrics in observations:
        require_output(metrics)
        assert metrics['returncode'] == 0
        assert metrics['stdout_bytes'] <= 8388608

def test_f05_joint_directions_format():
    """Finite real JSON field types and dimensions for joint_directions; no value comparison."""
    observations, _ = run_cases('E06')
    for case, actual in observations:
        assert not isinstance(actual, Exception), str(actual)
        assert type(actual) is dict, 'Output must be a JSON object'
        for field, expected in case['expected'].items():
            assert field in actual, f'Missing {field}'
            pending = [(actual[field], expected)]
            while pending:
                value, shape = pending.pop()
                if isinstance(shape, list):
                    assert type(value) is list, f'{field} must use JSON arrays'
                    assert len(value) == len(shape), f'{field} has incorrect dimensions'
                    pending.extend(zip(value, shape))
                else:
                    assert type(value) in (int, float), f'{field} must contain real numbers, not booleans or strings'
                    try:
                        represented = float(value)
                    except (OverflowError, ValueError, TypeError) as error:
                        raise AssertionError(f'{field} cannot be represented as a finite real') from error
                    assert np.isfinite(represented), f'{field} contains a nonfinite number'

def test_p05_joint_directions_performance():
    """Successful processes within the public cumulative time and stdout budget for joint_directions."""
    observations, budget = run_cases('E06', budget_only=True)
    assert budget['calls'] == len(observations)
    assert budget['elapsed'] <= 10.0
    for _, metrics in observations:
        require_output(metrics)
        assert metrics['returncode'] == 0
        assert metrics['stdout_bytes'] <= 8388608

