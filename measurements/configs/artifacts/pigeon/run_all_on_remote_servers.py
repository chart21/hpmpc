import json
import paramiko
import sys
import threading
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run experiments on remote servers.')
parser.add_argument('-g', type=str, help='Argument for -g')
parser.add_argument('-R', type=str, help='Argument for -R')
parser.add_argument('-O', type=str, help='Argument for -O')
parser.add_argument('-p', type=str, help='PID for the experiment', required=True)

# Parse the arguments
args = parser.parse_args()

# Load machine credentials from JSON file
with open('machines.json') as f:
    machines = json.load(f)

# Define commands to execute
base_commands = """
cd hpmpc
git pull
sudo chmod 777 measurements/logs
rm -rf measurements/logs/*
export MAX_BITWIDTH=$(lscpu | grep -i flags | grep -q "avx512" && echo 512 || (lscpu | grep -i flags | grep -q "avx2" && echo 256 || (lscpu | grep -i flags | grep -q "sse" && echo 128 || echo 64)))
export SUPPORTED_BITWIDTHS=$(echo 1 8 16 32 64 128 256 512 | awk -v max="$MAX_BITWIDTH" '{for(i=1;i<=NF;i++) if($i<=max) printf $i (i<NF && $i<max?",":"")}')
export ITERATIONS=1
echo "Running experiments with the following parameters: PID=$PID, IP0=$IP0, IP1=$IP1, IP2=$IP2, IP3=$IP3, ITERATIONS=$ITERATIONS, SUPPORTED_BITWIDTHS=$SUPPORTED_BITWIDTHS, MAX_BITWIDTH=$MAX_BITWIDTH" 
"""

# Construct experiment command with optional arguments
base_experiment_command = "./measurements/configs/artifacts/pigeon/run_all_experiments"
if args.O == "16Core_VMS":
    base_experiment_command += "_16Core_VMS"
base_experiment_command += ".sh "

experiment_command = base_experiment_command + "-a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -L $SUPPORTED_BITWIDTHS -D $MAX_BITWIDTH"

# Append -g and -R arguments if they are provided
if args.g:
    experiment_command += f" -g {args.g}"
if args.R:
    experiment_command += f" -R {args.R}"
# Load machine credentials from JSON file

parse_command = "python3 measurements/parse_logs.py measurements/logs"

print_lock = threading.Lock()

def execute_commands_on_remote(machine, pid):
    ip = machine['ip']
    username = machine['username']
    password = machine['password']
    hostname = machine['hostname']
    
    # Set up IP and PID environment variables
    ip_exports = "\n".join([f"export IP{i}={machines[i]['ip']}" for i in range(len(machines))])
    commands = f"{ip_exports}\nexport PID={pid}\n{base_commands}\n{experiment_command}\n{parse_command}"

    # Connect via SSH and execute commands
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    success = False
    
    try:
        client.connect(ip, username=username, password=password, timeout=30)
        stdin, stdout, stderr = client.exec_command(commands, get_pty=True)

        # Stream command output until completion
        for line in iter(stdout.readline, ""):
            with print_lock:
                print(f"[Machine {pid} - {hostname}] {line.strip()}")
        
        # Check exit status
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            with print_lock:
                print(f"[Machine {pid} - {hostname}] Experiments completed successfully")
            success = True
        else:
            with print_lock:
                print(f"[Machine {pid} - {hostname}] Experiments failed with exit status {exit_status}")
    finally:
        # Explicitly close all channels
        if hasattr(client, 'get_transport') and client.get_transport():
            if client.get_transport().is_active():
                client.get_transport().close()
        client.close()
    
    # Return success status to determine if logs should be fetched
    return success

def fetch_logs_with_timeout(machine, pid, timeout=30):
    """Fetch logs with explicit timeout and connection management"""
    ip = machine['ip']
    username = machine['username']
    password = machine['password']
    hostname = machine['hostname']
    local_log_dir = f"../../../logs/node_{pid}"
    remote_log_dir = "hpmpc/measurements/logs"
    os.makedirs(local_log_dir, exist_ok=True)
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username=username, password=password, timeout=timeout)
        sftp = client.open_sftp()
        
        for filename in sftp.listdir(remote_log_dir):
            remote_file = f"{remote_log_dir}/{filename}"
            local_file = f"{local_log_dir}/{filename}"
            sftp.get(remote_file, local_file)
            
        sftp.close()
    finally:
        # Explicitly close all channels
        if hasattr(client, 'get_transport') and client.get_transport():
            if client.get_transport().is_active():
                client.get_transport().close()
        client.close()

# Main execution
if __name__ == "__main__":
    # Check if script is launched with an argument for a specific machine
    if args.p != "all":
        machine_index = int(args.p)
        success = execute_commands_on_remote(machines[machine_index], machine_index)
        if success:
            fetch_logs_with_timeout(machines[machine_index], machine_index)
    else:
        # Run in multi-threaded mode for all machines
        threads = []
        results = [False] * len(machines)
        
        # Define a wrapper function to track success
        def run_and_track(machine, pid, result_index):
            results[result_index] = execute_commands_on_remote(machine, pid)
        
        # Start all threads
        for pid, machine in enumerate(machines):
            thread = threading.Thread(target=run_and_track, args=(machine, pid, pid))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Now fetch logs for successful executions
        for pid, success in enumerate(results):
            if success:
                fetch_logs_with_timeout(machines[pid], pid)
                
    print("Script execution completed. Logs are saved to measurements/logs")
    exit(0)
