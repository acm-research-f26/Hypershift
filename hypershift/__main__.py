"""A single entry point for the organized historical research scripts."""
import argparse
from pathlib import Path
import sys
from .workspace import ROOT,catalog,clean,run_python,test_environment


def main(arguments=None):
    arguments=list(sys.argv[1:] if arguments is None else arguments)
    parser=argparse.ArgumentParser(description='Run the organized research sources with their original imports, paths and source hashes.')
    commands=parser.add_subparsers(dest='command',required=True)
    listing=commands.add_parser('list',help='List modules and their organized source paths')
    listing.add_argument('filter',nargs='?',default='')
    runner=commands.add_parser('run',help='Run a script by its original module name')
    runner.add_argument('module');runner.add_argument('arguments',nargs=argparse.REMAINDER)
    tests=commands.add_parser('test',help='Run unittest; with no arguments discover all tests')
    tests.add_argument('arguments',nargs=argparse.REMAINDER)
    python=commands.add_parser('python',help='Run Python with the historical workspace view')
    python.add_argument('arguments',nargs=argparse.REMAINDER)
    commands.add_parser('clean',help='Remove unchanged compatibility files left by an exited command')
    # Pass Python's own switches through instead of interpreting them as
    # launcher options. An optional -- separator is accepted as well.
    if arguments[:1]==['python'] and arguments[1:] not in [['-h'],['--help']]:
        rest=arguments[1:]
        if rest[:1]==['--']:rest=rest[1:]
        try:return run_python(rest)
        except (OSError,RuntimeError,ValueError) as exc:print('Hypershift:',exc,file=sys.stderr);return 2
    if arguments[:1]==['test'] and arguments[1:] not in [['-h'],['--help']]:
        rest=arguments[1:]
        if rest[:1]==['--']:rest=rest[1:]
        if not rest or all(value in ['-v','-q','-b','-f','--locals','--buffer','--failfast'] for value in rest):
            rest=['discover','-s',str(ROOT),'-p','test_*.py',*rest]
        try:return run_python(['-m','unittest',*rest],env=test_environment())
        except (OSError,RuntimeError,ValueError) as exc:print('Hypershift:',exc,file=sys.stderr);return 2
    options=parser.parse_args(arguments)
    try:
        if options.command=='list':
            for name,path in catalog().items():
                relative=path.relative_to(ROOT)
                if options.filter.lower() in str(relative).lower():print(f'{Path(name).stem:<45} {relative.as_posix()}')
            return 0
        if options.command=='clean':clean();return 0
        if options.command=='run':
            name=options.module.removesuffix('.py')+'.py'
            if Path(name).name!=name or name not in catalog():parser.error('Unknown research module. Use `python -m hypershift list`.')
            return run_python([str(ROOT/name),*options.arguments])
        if options.command=='test':
            args=options.arguments or ['discover','-s',str(ROOT),'-p','test_*.py']
            return run_python(['-m','unittest',*args])
        return run_python(options.arguments)
    except (OSError,RuntimeError,ValueError) as exc:
        print('Hypershift:',exc,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
