"""Run numerical analysis from bundled inputs, without figures or manuscript builders."""
import argparse
import ast
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
INPUTS = ['country_names.csv', 'cru_crosswalk.csv', 'cru_monthly.parquet',
          'fao_prod.parquet', 'panel_dyad.parquet', 'policy_years_prohib_quota.csv',
          'policy_years_prohibition.csv']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check syntax, imports and required inputs without estimation.')
    parser.add_argument('--smoke', action='store_true', help='Run one existing sensitivity specification with two bootstrap draws; diagnostic only.')
    parser.add_argument('--output', type=Path, help='New or empty output directory; default: runs/<timestamp>.')
    parser.add_argument('--nproc', type=int, default=1, help='Bootstrap workers (default: 1; Windows requires 1).')
    args = parser.parse_args()
    if args.check and args.smoke:
        parser.error('--check and --smoke are mutually exclusive')
    if args.nproc < 1 or (sys.platform == 'win32' and args.nproc != 1):
        parser.error('--nproc must be positive; Windows requires --nproc 1')
    scripts = [ROOT / 'run_analysis.py', ROOT / 'code/rm2.py', ROOT / 'code/run_v2.py']
    missing = [str(ROOT / 'data' / name) for name in INPUTS if not (ROOT / 'data' / name).is_file()]
    missing += [str(path) for path in scripts if not path.is_file()]
    if missing:
        parser.error('Missing required files: ' + ', '.join(missing))
    if args.check:
        for path in scripts:
            ast.parse(path.read_text(encoding='utf-8-sig'))
        versions = {name: importlib.import_module(name).__version__
                    for name in ['numpy', 'pandas', 'scipy', 'pyarrow']}
        sys.path.insert(0, str(ROOT / 'code'))
        importlib.import_module('rm2')
        pd = importlib.import_module('pandas')
        shapes = {name: list((pd.read_parquet(ROOT / 'data' / name) if name.endswith('.parquet')
                             else pd.read_csv(ROOT / 'data' / name)).shape) for name in INPUTS}
        print(json.dumps({'syntax_and_imports': 'passed', 'input_shapes': shapes,
                          'python': sys.version.split()[0], 'versions': versions,
                          'experiment_run': False}, indent=2))
        return
    out = (args.output or ROOT / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output directory must be new or empty: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DATA_DIR=str(ROOT / 'data'), OUT_DIR=str(out / 'results'),
               NPROC=str(args.nproc), PYTHONDONTWRITEBYTECODE='1')
    env.pop('ONLY_ROB', None)

    def run(run_env):
        subprocess.run([sys.executable, '-B', str(ROOT / 'code/run_v2.py')],
                       env=run_env, cwd=ROOT, check=True)

    if args.smoke:
        print('Diagnostic only: clean_window_2y, two bootstrap draws, one worker.', flush=True)
        run(dict(env, B_MAIN='2', B_SUB='2', B_ROB='2', NPROC='1', ONLY_ROB='clean_window_2y'))
    else:
        run(env)
        # Preserve the original run_all.py order for the two additional sensitivities.
        run(dict(env, ONLY_ROB='clean_window_1y,clean_window_2y'))
    print('Outputs: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
