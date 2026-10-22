"""Map a single pytest item to the existing ratio-verifier wrapper contract."""
import os
import sys
import traceback


def main():
    # Initialization belongs to the trusted grader, not the candidate. Any
    # failure here must reach the outer infrastructure-error boundary.
    sys.path.insert(0, '/tests')
    import pytest
    from test_kernel import IDS

    if os.geteuid() != 0 or len(sys.argv) != 2 or sys.argv[1] not in IDS:
        return 2

    class Results:
        def __init__(self):
            self.reports = []
            self.invalid = False

        def pytest_runtest_logreport(self, report):
            self.reports.append(report)

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            yield
            if call.excinfo and not isinstance(
                call.excinfo.value,
                (AssertionError, pytest.fail.Exception, pytest.skip.Exception),
            ):
                self.invalid = True

    results = Results()
    result = pytest.main(
        ['-q', '-s', '--tb=short', '-p', 'no:cacheprovider',
         f'/tests/test_kernel.py::test_contract[{sys.argv[1]}]'],
        plugins=[results],
    )
    calls = [r for r in results.reports if r.when == 'call']
    if results.invalid or result not in (0, 1) or len(calls) != 1:
        return 2
    return 0 if calls[0].passed and all(r.passed for r in results.reports) else 1


if __name__ == '__main__':
    try:
        status = main()
    except BaseException:
        # Includes import errors, unexpected pytest errors, and interruption.
        # Do not allow an incidental Python exit code 1 to become a model FAIL.
        try:
            traceback.print_exc()
        except BaseException:
            pass
        status = 2
    raise SystemExit(status)
