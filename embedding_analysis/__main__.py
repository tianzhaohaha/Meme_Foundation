"""Run with: python -m embedding_analysis --data /path/subset_50000.zip."""
import argparse
import os
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent
PROJECT = MODULE.parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface'))
    if argv and argv[0] in ('extract', 'plot'):
        command = argv.pop(0)
        if command == 'extract':
            from .extract import main as execute
        else:
            from .plot import main as execute
        execute(argv)
        return
    if argv and argv[0] == 'run':
        argv.pop(0)
    from .extract import DEFAULT_VARIANTS, VARIANTS
    parser = argparse.ArgumentParser(description='Compare shared Binary source and V4/V5/V6 with PCA/t-SNE, without training.')
    parser.add_argument('--data', type=Path, default=PROJECT / 'data/subset_50000.zip')
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'training/runs_comparison')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument('--checkpoint', action='append', default=[], help='Repeat Binary=/path/best.pt, V4=/path/best.pt, etc.')
    parser.add_argument('--output', type=Path, default=MODULE / 'results')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--include-encoded', action='store_true')
    parser.add_argument('--trust-legacy-checkpoint', action='store_true')
    parser.add_argument('--normalization', choices=['l2', 'none'], default='l2')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42, help='Projection seed; does not retrain any checkpoint')
    args = parser.parse_args(argv)
    # Respect training environment caches; otherwise use the original project's defaults.
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve() / '.matplotlib'))
    features = args.output / 'features'
    extraction = ['--data', str(args.data), '--runs-root', str(args.runs_root),
        '--output', str(features), '--split', args.split, '--device', args.device,
        '--batch-size', str(args.batch_size), '--variants', *args.variants]
    for checkpoint in args.checkpoint:
        extraction.extend(['--checkpoint', checkpoint])
    if args.include_encoded:
        extraction.append('--include-encoded')
    if args.trust_legacy_checkpoint:
        extraction.append('--trust-legacy-checkpoint')
    from .extract import main as extract
    extract(extraction)
    names = [item.partition('=')[0] for item in args.checkpoint] if args.checkpoint else args.variants
    plotting = ['--features', *[str(features / f'{name}_{args.split}.npz') for name in names],
        '--output', str(args.output / 'plots'), '--normalization', args.normalization,
        '--perplexity', str(args.perplexity), '--iterations', str(args.iterations), '--seed', str(args.seed),
        '--spaces', 'fusion']
    if args.include_encoded:
        plotting.append('encoded')
    from .plot import main as plot
    plot(plotting)


if __name__ == '__main__':
    main()
