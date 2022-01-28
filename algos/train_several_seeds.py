"""
Script to train several seeds. Beware, it changes model name as appropriate.

Usage:
    train_several_seeds.py [--algos=ALGO]... [options] 

Options:
    -h --help              Show this screen.

    --use-TF               Use Teacher Forcing or not during training. Not
                           using it, would add a GRU cell at the GNN update step.
                           [default: False]

    --use-GRU              Force the usage of a GRU cell at the GNN update step.

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, max, mean, predinet}. [default: predinet]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --algos ALGO           Which algorithms to train {BFS, parallel_coloring}.
                           Repeatable parameter. [default: BFS]

    --epochs EP            Number of epochs to train. [default: 250]

    --no-patience          Do not utilise patience, train for max epochs. [default: False]

    --model-name MN        Name of the model when saving.
  
    --num-seeds NS         How many different seeds to train [default: 5]

    --starting-seed SS     Which seed index to start from [default: 0]

    --CUDA-START CS        From which cuda device to start [default: 0]

    --CUDA-MOD CM          How many machines to cycle [default: 1]

    --dry                  Dry run. Just print commands, nothing else. [default: False]
"""
import os
import schema
from docopt import docopt

args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--use-TF': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'mean', 'max', 'predinet']),
                        '--no-next-step-pool': bool,
                        '--use-GRU': bool,
                        '--no-patience': bool,
                        '--dry': bool,
                        '--num-seeds': schema.Use(int),
                        '--starting-seed': schema.Use(int),
                        '--CUDA-MOD': schema.Use(int),
                        '--CUDA-START': schema.Use(int),
                        '--model-name': schema.Or(None, schema.Use(str)),
                        '--epochs': schema.Use(int)})
args = schema.validate(args)

commands = []
for seed in range(args['--starting-seed'], args['--starting-seed']+args['--num-seeds']):
    machine = args['--CUDA-START'] + seed % args['--CUDA-MOD']
    command = f'CUDA_VISIBLE_DEVICES={machine} python -m algos.train --epochs {args["--epochs"]} --model-name {args["--model-name"]}_seed_{seed} --pooling {args["--pooling"]} '

    for algo in args['--algos']:
        command += f'--algos {algo} '

    for flag in [
            '--use-TF', '--use-GRU', '--no-patience',
            '--no-next-step-pool'
    ]:
        if args[flag]:
            command += flag + ' '
    command += f'--seed {seed} '
    command += '&'
    commands.append(command)

print(commands)
if args['--dry']:
    exit(0)

with open("runseeds.sh", 'w+') as f:
    for command in commands:
        print(command, file=f)
    pass

os.system('chmod +x runseeds.sh')
os.system('./runseeds.sh')
os.system('rm runseeds.sh')
