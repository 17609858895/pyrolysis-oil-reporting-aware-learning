"""Portable, sequential entry points for the code-only research release."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['setup', 'prepare', 'benchmark', 'ablation', 'analysis', 'all'])
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for name in ('data', 'cache', 'tables'):
        (ROOT / name).mkdir(exist_ok=True)
    if args.stage == 'setup':
        print('Place compound_records.xlsx in data/, then run prepare.')
        return
    if args.stage in ('prepare', 'all'):
        data = Path(os.environ.get('PYROLYSIS_DATA', ROOT / 'data/compound_records.xlsx'))
        if not data.is_file():
            parser.error('Download compound_records.xlsx from doi:10.23642/USN.30630182 into data/.')
        run('s00_resolve_pubchem.py')
        run('s01_build_dataset.py')
        run('s02_compound_descriptors.py')
        run('s03_build_grid.py')
        run('s03_build_grid.py', 3, 'grid_min3')
        run('s03_build_grid.py', 4, 'grid_pq25', .25, .75)
        run('s03_build_grid.py', 4, 'grid_pq50', .5, .5)
        run('s03_build_grid.py', 4, 'grid_broad', .1, .9, 'broad')
        # Sensitivity grids write the same oil table; restore the main definition last.
        run('s03_build_grid.py')
    if args.stage in ('benchmark', 'all'):
        run('s04_benchmark.py', '--methods', 'core', '--out', 'benchmark', '--save-pred', '--jobs', args.jobs)
        run('s04_benchmark.py', '--methods', 'stack', '--out', 'stack', '--save-pred', '--jobs', args.jobs)
        run('s04_benchmark.py', '--methods', 'scale', '--out', 'scale', '--save-pred', '--jobs', args.jobs)
        run('s10_final.py', '--mode', 'benchmark', '--jobs', args.jobs)
        run('s10_final.py', '--mode', 'stability', '--cfg', 'logistic-1.5', '--jobs', args.jobs)
        run('s09_paired_tests.py', 'RLCure', 'paired_tests_final.csv')
        run('s11_primary_tests.py')
    if args.stage in ('ablation', 'all'):
        specs = [
            ['--methods', 'RLCure-noP,RLCure-fixedL,RLAFT', '--out', 'abl_components'],
            ['--no-prior', '--methods', 'RLCure,Hurdle+P-cal,LightGBM-tuned', '--out', 'abl_noprior'],
            ['--drop-feats', 'structure', '--methods', 'RLCure,Hurdle+P-cal,LightGBM-tuned', '--out', 'abl_nostruct'],
            ['--grid', 'grid_min3', '--methods', 'RLCure,Hurdle+P-cal,LightGBM-tuned', '--out', 'abl_min3'],
            ['--grid', 'grid_pq25', '--methods', 'RLCure', '--out', 'abl_pq25'],
            ['--grid', 'grid_pq50', '--methods', 'RLCure', '--out', 'abl_pq50'],
            ['--grid', 'grid_broad', '--methods', 'RLCure,Hurdle+P-cal,LightGBM-tuned', '--out', 'abl_broad'],
            ['--family-out', '--methods', 'RLCure,Hurdle+P-cal,LightGBM-tuned', '--out', 'final_familyout'],
        ]
        for spec in specs:
            run('s10_final.py', *spec, '--jobs', args.jobs)
        run('s12_ablation_summary.py')
    if args.stage in ('analysis', 'all'):
        run('s05_within_study_causal.py')
        run('s07_counterfactual_fidelity.py', 'RLCure', '_final')
        run('s08_interpret.py', 'logistic', 1.5)
        run('s13_figure_data.py')
        for mode in ('stack', 'causal', 'ale'):
            run('s15_si_analyses.py', mode, args.jobs)

if __name__ == '__main__':
    main()
