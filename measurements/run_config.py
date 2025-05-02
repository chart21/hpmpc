import argparse
import os
import time
from itertools import product
import subprocess

def parse_config(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            config[key] = list(map(str, value.split(',')))
    return config

def generate_combinations(config):
    keys, values = zip(*config.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))

def determine_n(protocol):
    if protocol == 4:
        return 2
    elif protocol < 7:
        return 3
    else:
        return 4

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

def run_command(command, log):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stdout:
        output = stdout.decode()
        print(output)
        log.write(output)
    if stderr:
        output = stderr.decode()
        print(output)
        log.write(output)

def main():
    parser = argparse.ArgumentParser(description='Run configurations')
    parser.add_argument('config', help='Configuration file or folder')
    parser.add_argument('-p', type=str, default='all', help='Party identifier')
    parser.add_argument('-a', type=str, default='127.0.0.1', help='IP address for a')
    parser.add_argument('-b', type=str, default='127.0.0.1', help='IP address for b')
    parser.add_argument('-c', type=str, default='127.0.0.1', help='IP address for c')
    parser.add_argument('-d', type=str, default='127.0.0.1', help='IP address for d')
    parser.add_argument('-g', type=int, default=0, help='Numbers of GPUs to use (0 for CPU only)')
    parser.add_argument('-f', type=str, default='', help='filename suffix')
    parser.add_argument('--override', nargs='*', help='Override config options (key=value)')
    parser.add_argument('-i', type=int, default=1, help='Number of iterations per run')

    args = parser.parse_args()

    # Parse override options
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            overrides[key] = value.split(',')

    config_files = []
    if os.path.isdir(args.config):
        config_files = [os.path.join(args.config, f) for f in os.listdir(args.config) if f.endswith('.conf')]
    else:
        config_files = [args.config]

    total_configs = len(config_files)
    config_count = 0

    for config_file in config_files:
        config_count += 1
        config = parse_config(config_file)

        # Apply overrides
        config.update(overrides)

        combinations = list(generate_combinations(config))
        total_runs = len(combinations)
        
        print("\n===================")
        print(f"====== Config {config_count}/{total_configs} ======")
        
        config_content = "\n".join([f"{k}={','.join(v)}" for k, v in config.items()])
        print(config_content)
        print("===================\n===================")

        timestamp = int(time.time())
        if args.f != '':
            log_file = f"measurements/logs/{os.path.basename(config_file).split('.')[0]}_{args.f}_{timestamp}.log"
            config_log_file = f"measurements/logs/{os.path.basename(config_file).split('.')[0]}_{args.f}_{timestamp}.log-config"
        else:
            log_file = f"measurements/logs/{os.path.basename(config_file).split('.')[0]}_{timestamp}.log"
            config_log_file = f"measurements/logs/{os.path.basename(config_file).split('.')[0]}_{timestamp}.log-config"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Save config details to the config log file
        with open(config_log_file, 'w') as config_log:
            config_log.write(config_content)
        
        with open(log_file, 'w') as log:
            run_count = 0
            for comb in combinations:
                run_count += 1
                for iteration in range(1, args.i + 1):
                    n_value = determine_n(int(comb['PROTOCOL']))
                    splitroles = int(comb['SPLITROLES']) if 'SPLITROLES' in comb else 0
                    print_progress_bar(run_count, total_runs, prefix='Progress', suffix=f'{run_count}/{total_runs}')
                    print(f"\n====== Run {run_count}/{total_runs} (Iteration {iteration}/{args.i}) ======")
                    print(f"Running: PARTY={args.p} " + ' '.join([f"{k}={v}" for k, v in comb.items()]))
                    print("===================")

                    make_command = f"make -j PARTY={args.p} " + ' '.join([f"{k}={v}" for k, v in comb.items()])
                    script_command = f"scripts/run.sh -n {n_value} -p {args.p} -a {args.a} -b {args.b} -c {args.c} -d {args.d} -g {args.g} -s {splitroles}"
                    log.write(f"\n====== Run {run_count}/{total_runs} (Iteration {iteration}/{args.i}) ======\n")
                    log.write(f"Running: PARTY={args.p} " + ' '.join([f"{k}={v}" for k, v in comb.items()]) + "\n")
                    log.write("===================\n")

                    log.write(f"Executing: {make_command}\n")
                    run_command(make_command, log)
                    log.write(f"Executing: {script_command}\n")
                    run_command(script_command, log)

                    print(f"==== Saved log file {run_count}/{total_runs} to {log_file} ====")
                    print(f"==== Saved config details to {config_log_file} ====")

if __name__ == '__main__':
    main()
