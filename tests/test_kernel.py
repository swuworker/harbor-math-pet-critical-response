"""Independent numerical behavior checks and separate schema/resource checks."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'test-manifest.json').read_text(encoding='utf8'))
IDS=[x['id'] for x in MANIFEST['tests']]

def finite_real(value):
    """A total numeric predicate: bad candidate values return False, never grader errors."""
    if type(value) not in (int, float):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, ValueError, TypeError):
        return False


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
    '''A01: 单位单像素弦长; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A02: 斜射线跨多个像素; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A03: 非方形像素及坐标平移; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A04: 整条符合线衰减; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A05: 非对称PSF伴随; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A06: 模糊/投影/衰减次序; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A07: 非均匀效率和非均匀灵敏度; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''A08: 有符号伴随及背景排除; independently frozen expected values for each stated output. Public contract: "The adjoint is with respect to the ordinary Euclidean dot products".'''
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
    '''B01: 无正则单步; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''B02: 非均匀灵敏度单步; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''B03: 背景与零观测; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''B04: 纯二次正则; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''B05: 不同步长缩放; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''B06: 有限步与最终最优解区别; independently frozen expected values for each stated output. Public contract: "Return the result of exactly K such mathematical steps.".'''
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
    '''C01: 二维各向同性组; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C02: 网格边界与间距; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C03: 非均匀TV权重; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C04: 严格融合的平坦块; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C05: 两个未融合像素区域; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C06: 二次与TV共同作用; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C07: 右近端与单步滞后区别; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''C08: 临界融合点基准图像; independently frozen expected values for each stated output. Public contract: "A component with no neighbour is zero.".'''
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
    '''D01: 光滑单像素初值响应; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D02: 衰减改变A和镜像度量; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D03: 背景方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D04: 数据方向含基准零计数; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D05: 临界融合增强TV方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D06: 临界融合减弱TV方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D07: 非零二维组曲率; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D08: 严格融合块内部方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D09: 临界二维组的可行方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D10: 多步链式依赖; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D11: 联合参数方向; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''D12: K=0方向语义; independently frozen expected values for each stated output. Public contract: "The requested object is this one-sided directional derivative.".'''
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
    '''E01: 非对称PSF与变衰减; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''E02: 混合正则与非均匀效率; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''E03: 较早一步临界融合影响后续; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''E04: 二维组边界与相邻边同时临界; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''E05: 网格轴互换与射线重排; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''E06: 联合方向的正齐次性及一般响应; independently frozen expected values for each stated output. Public contract: "The geometry, H, efficiency, w, tau and K stay fixed.".'''
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
    '''Public contract: "use only JSON-compatible finite real numbers, lists and dictionaries"; fixture A05.'''
    observations, budget = run_cases('A05')
    assert len(observations) == 1, "Every frozen request must be observed"
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert 'ray_matrix' in actual_0, "Missing ray_matrix"
    assert type(actual_0['ray_matrix']) is list, "ray_matrix must use JSON arrays"
    assert len(actual_0['ray_matrix']) == 4, "ray_matrix outer dimension from the frozen request"
    assert all(type(row) is list for row in actual_0['ray_matrix']), "ray_matrix rows must use JSON arrays"
    assert all(len(row) == 4 for row in actual_0['ray_matrix']), "ray_matrix row dimensions from the frozen request"
    ray_matrix_0 = tuple(item for row in actual_0['ray_matrix'] for item in row)
    assert all(type(item) in (int, float) for item in ray_matrix_0), "ray_matrix must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in ray_matrix_0), "ray_matrix must contain finite representable reals"
    assert 'system_matrix' in actual_0, "Missing system_matrix"
    assert type(actual_0['system_matrix']) is list, "system_matrix must use JSON arrays"
    assert len(actual_0['system_matrix']) == 4, "system_matrix outer dimension from the frozen request"
    assert all(type(row) is list for row in actual_0['system_matrix']), "system_matrix rows must use JSON arrays"
    assert all(len(row) == 4 for row in actual_0['system_matrix']), "system_matrix row dimensions from the frozen request"
    system_matrix_0 = tuple(item for row in actual_0['system_matrix'] for item in row)
    assert all(type(item) in (int, float) for item in system_matrix_0), "system_matrix must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in system_matrix_0), "system_matrix must contain finite representable reals"
    assert 'mean' in actual_0, "Missing mean"
    assert type(actual_0['mean']) is list, "mean must use JSON arrays"
    assert len(actual_0['mean']) == 4, "mean outer dimension from the frozen request"
    mean_0 = actual_0['mean']
    assert all(type(item) in (int, float) for item in mean_0), "mean must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in mean_0), "mean must contain finite representable reals"
    assert 'adjoint' in actual_0, "Missing adjoint"
    assert type(actual_0['adjoint']) is list, "adjoint must use JSON arrays"
    assert len(actual_0['adjoint']) == 4, "adjoint outer dimension from the frozen request"
    adjoint_0 = actual_0['adjoint']
    assert all(type(item) in (int, float) for item in adjoint_0), "adjoint must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in adjoint_0), "adjoint must contain finite representable reals"

def test_p01_forward_matrix_vectors_performance():
    '''Public contract: "a cumulative 10-second execution budget, including interpreter startup"; fixture A05.'''
    observations, budget = run_cases('A05', budget_only=True)
    assert len(observations) == 1, "Every frozen request must be observed"
    assert budget["calls"] == 1
    assert budget["elapsed"] <= 10.0
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert actual_0["returncode"] == 0
    assert actual_0["stdout_bytes"] <= 8388608

def test_f02_finite_history_objective_format():
    '''Public contract: "use only JSON-compatible finite real numbers, lists and dictionaries"; fixture B06.'''
    observations, budget = run_cases('B06')
    assert len(observations) == 1, "Every frozen request must be observed"
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert 'image' in actual_0, "Missing image"
    assert type(actual_0['image']) is list, "image must use JSON arrays"
    assert len(actual_0['image']) == 2, "image outer dimension from the frozen request"
    image_0 = actual_0['image']
    assert all(type(item) in (int, float) for item in image_0), "image must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in image_0), "image must contain finite representable reals"
    assert 'history' in actual_0, "Missing history"
    assert type(actual_0['history']) is list, "history must use JSON arrays"
    assert len(actual_0['history']) == 4, "history outer dimension from the frozen request"
    assert all(type(row) is list for row in actual_0['history']), "history rows must use JSON arrays"
    assert all(len(row) == 2 for row in actual_0['history']), "history row dimensions from the frozen request"
    history_0 = tuple(item for row in actual_0['history'] for item in row)
    assert all(type(item) in (int, float) for item in history_0), "history must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in history_0), "history must contain finite representable reals"
    assert 'objective' in actual_0, "Missing objective"
    assert type(actual_0['objective']) in (int, float), "objective must be a real scalar, not bool or string"
    assert finite_real(actual_0['objective']), "objective must be a finite representable real"

def test_p02_finite_history_objective_performance():
    '''Public contract: "a cumulative 10-second execution budget, including interpreter startup"; fixture B06.'''
    observations, budget = run_cases('B06', budget_only=True)
    assert len(observations) == 1, "Every frozen request must be observed"
    assert budget["calls"] == 1
    assert budget["elapsed"] <= 10.0
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert actual_0["returncode"] == 0
    assert actual_0["stdout_bytes"] <= 8388608

def test_f03_zero_step_direction_format():
    '''Public contract: "use only JSON-compatible finite real numbers, lists and dictionaries"; fixture D12.'''
    observations, budget = run_cases('D12')
    assert len(observations) == 1, "Every frozen request must be observed"
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert 'image' in actual_0, "Missing image"
    assert type(actual_0['image']) is list, "image must use JSON arrays"
    assert len(actual_0['image']) == 4, "image outer dimension from the frozen request"
    image_0 = actual_0['image']
    assert all(type(item) in (int, float) for item in image_0), "image must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in image_0), "image must contain finite representable reals"
    assert 'response' in actual_0, "Missing response"
    assert type(actual_0['response']) is list, "response must use JSON arrays"
    assert len(actual_0['response']) == 4, "response outer dimension from the frozen request"
    response_0 = actual_0['response']
    assert all(type(item) in (int, float) for item in response_0), "response must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in response_0), "response must contain finite representable reals"

def test_p03_zero_step_direction_performance():
    '''Public contract: "a cumulative 10-second execution budget, including interpreter startup"; fixture D12.'''
    observations, budget = run_cases('D12', budget_only=True)
    assert len(observations) == 1, "Every frozen request must be observed"
    assert budget["calls"] == 1
    assert budget["elapsed"] <= 10.0
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert actual_0["returncode"] == 0
    assert actual_0["stdout_bytes"] <= 8388608

def test_f04_critical_direction_format():
    '''Public contract: "use only JSON-compatible finite real numbers, lists and dictionaries"; fixture D09.'''
    observations, budget = run_cases('D09')
    assert len(observations) == 1, "Every frozen request must be observed"
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert 'image' in actual_0, "Missing image"
    assert type(actual_0['image']) is list, "image must use JSON arrays"
    assert len(actual_0['image']) == 4, "image outer dimension from the frozen request"
    image_0 = actual_0['image']
    assert all(type(item) in (int, float) for item in image_0), "image must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in image_0), "image must contain finite representable reals"
    assert 'response' in actual_0, "Missing response"
    assert type(actual_0['response']) is list, "response must use JSON arrays"
    assert len(actual_0['response']) == 4, "response outer dimension from the frozen request"
    response_0 = actual_0['response']
    assert all(type(item) in (int, float) for item in response_0), "response must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in response_0), "response must contain finite representable reals"

def test_p04_critical_direction_performance():
    '''Public contract: "a cumulative 10-second execution budget, including interpreter startup"; fixture D09.'''
    observations, budget = run_cases('D09', budget_only=True)
    assert len(observations) == 1, "Every frozen request must be observed"
    assert budget["calls"] == 1
    assert budget["elapsed"] <= 10.0
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert actual_0["returncode"] == 0
    assert actual_0["stdout_bytes"] <= 8388608

def test_f05_joint_directions_format():
    '''Public contract: "use only JSON-compatible finite real numbers, lists and dictionaries"; fixture E06.'''
    observations, budget = run_cases('E06')
    assert len(observations) == 2, "Every frozen request must be observed"
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert 'image' in actual_0, "Missing image"
    assert type(actual_0['image']) is list, "image must use JSON arrays"
    assert len(actual_0['image']) == 2, "image outer dimension from the frozen request"
    image_0 = actual_0['image']
    assert all(type(item) in (int, float) for item in image_0), "image must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in image_0), "image must contain finite representable reals"
    assert 'response' in actual_0, "Missing response"
    assert type(actual_0['response']) is list, "response must use JSON arrays"
    assert len(actual_0['response']) == 2, "response outer dimension from the frozen request"
    response_0 = actual_0['response']
    assert all(type(item) in (int, float) for item in response_0), "response must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in response_0), "response must contain finite representable reals"
    _, actual_1 = observations[1]
    assert type(actual_1) is dict, "Expected a completed request returning an object"
    assert 'image' in actual_1, "Missing image"
    assert type(actual_1['image']) is list, "image must use JSON arrays"
    assert len(actual_1['image']) == 2, "image outer dimension from the frozen request"
    image_1 = actual_1['image']
    assert all(type(item) in (int, float) for item in image_1), "image must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in image_1), "image must contain finite representable reals"
    assert 'response' in actual_1, "Missing response"
    assert type(actual_1['response']) is list, "response must use JSON arrays"
    assert len(actual_1['response']) == 2, "response outer dimension from the frozen request"
    response_1 = actual_1['response']
    assert all(type(item) in (int, float) for item in response_1), "response must contain real numbers, not booleans or strings"
    assert all(finite_real(item) for item in response_1), "response must contain finite representable reals"

def test_p05_joint_directions_performance():
    '''Public contract: "a cumulative 10-second execution budget, including interpreter startup"; fixture E06.'''
    observations, budget = run_cases('E06', budget_only=True)
    assert len(observations) == 2, "Every frozen request must be observed"
    assert budget["calls"] == 2
    assert budget["elapsed"] <= 10.0
    _, actual_0 = observations[0]
    assert type(actual_0) is dict, "Expected a completed request returning an object"
    assert actual_0["returncode"] == 0
    assert actual_0["stdout_bytes"] <= 8388608
    _, actual_1 = observations[1]
    assert type(actual_1) is dict, "Expected a completed request returning an object"
    assert actual_1["returncode"] == 0
    assert actual_1["stdout_bytes"] <= 8388608

