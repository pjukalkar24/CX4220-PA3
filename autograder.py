"""Autograder Python script.

Test output for correctness and performance bounds.
"""
import subprocess
import os
import sys


def pa3_command(kind, nranks, testfile):
    assert kind in ['spgemm', 'apsp']
    return ['srun', '-n', str(nranks), './pa3', kind, testfile]


def run_pa3(kind, nranks, testfile):
    return subprocess.run(pa3_command(kind, nranks, testfile), capture_output=True,
                          text=True)


def parse_output(output):
    """Parse output from a run."""
    lines = [line.strip() for line in output.split('\n')]
    lines = [line[3:].strip() for line in lines if line.startswith('==>')]
    try:
        correctness = [line for line in lines if line.startswith('correctness_check')][0]
        correctness_output = correctness.split('=')[-1]
        time_taken = [line for line in lines if line.startswith('time_taken')][0]
        time_taken = float(time_taken.split('=')[-1][:-1])
        return correctness_output == 'ok', time_taken
    except (ValueError, IndexError):
        return False, None


def check_files(files):
    """Check that all of these files exist."""
    if not all(os.path.exists(f) for f in files):
        print('ERROR: Some of the files are missing and cannot run the autograder tests.', file=sys.stderr)
        print('Please make sure that you are running the autograder script from '
              'the root of the project and that all files have been downloaded.', file=sys.stderr)
        sys.exit(1)


def correctness_checks():
    """Run correctness checks on smaller test files."""
    spgemm_testfiles = [
        'testfiles/ibm32-spgemm.dat',
        'testfiles/simple-spgemm.dat',
    ]
    apsp_testfiles = [
        'testfiles/ibm32-apsp.dat',
        'testfiles/simple-apsp.dat',
    ]
    check_files(spgemm_testfiles)
    check_files(apsp_testfiles)
    print('=================== CORRECTNESS TESTS ===================')
    print()
    for kind, testfiles in [('spgemm', spgemm_testfiles), ('apsp', apsp_testfiles)]:
        print(f'TESTING {kind.upper()}:')
        print()
        all_correct = True
        for nranks in [1, 4, 9, 16, 25, 36]:
            print(f'==> Testing P={nranks}:')
            for testfile in testfiles:
                print(f'====> {testfile}: ', end='')
                cp = run_pa3(kind, nranks, testfile)
                correctness, _ = parse_output(cp.stdout)
                if not correctness:
                    all_correct = False
                    print(f'FAILED')
                else:
                    print(f'OK')
        print()
        print('*************************************')
        if all_correct:
            print(f'{kind.upper()} PASSED')
        else:
            print(f'{kind.upper()} FAILED')
            print()
            cmd = ' '.join(pa3_command(kind, 'P', 'TESTFILE'))
            print(f'Please try to run manually with a command similar to {cmd}')
        print('-------------------------------------')
        print()


def perf_checks():
    """Run the performance tests and check runtime bounds."""
    perf_testfiles = {
        'spgemm': ('perf-testfiles/GL7d14.dat', {16: [20.0, 30.0], 36: [10.0, 20.0]}, [20, 10]),
        'apsp': ('perf-testfiles/G12.dat', {16: [65.0, 80.0], 36: [30.0, 40.0]}, [15, 10]),
    }
    check_files([file for file, _, _ in perf_testfiles.values()])
    print('=================== PERFORMANCE TESTS ===================')
    print()
    for kind, (testfile, time_info, points) in perf_testfiles.items():
        print(f'TESTING {kind.upper()} WITH {testfile}:')
        print()
        all_correct = True
        for nranks, times in time_info.items():
            print(f'==> Testing P={nranks}:')
            cp = run_pa3(kind, nranks, testfile)
            correctness, current_time = parse_output(cp.stdout)
            if not correctness:
                print('====> FAILED correctness check!')
                all_correct = False
                continue
            passed_time = False
            for i, (time, p) in enumerate(zip(times, points)):
                passed_time = current_time < time
                if current_time < time and i == 0:
                    print(f'====> PASSED (runtime of {current_time}s)')
                    break
                elif current_time < time:
                    print(f'====> FAILED (runtime of {current_time}s)')
                    print(f'======> Passed for partial runtime of {current_time}s ({p} points)')
                    all_correct = False
                    break
            if not passed_time:
                print(f'====> FAILED (runtime of {current_time}s)')
            if not passed_time:
                all_correct = False
        print()
        print('*************************************')
        if all_correct:
            print(f'{kind.upper()} PASSED PERFORMANCE TEST')
        else:
            print(f'{kind.upper()} FAILED PERFORMANCE TEST')
            print()
            print('See handout for detailed breakdown of points and performance requirements')
            cmd = ' '.join(pa3_command(kind, '$P', '$TESTFILE'))
            print(f'Please try to run manually with a command similar to {cmd}')
        print('-------------------------------------')
        print()


if __name__ == '__main__':
    correctness_checks()
    perf_checks()
