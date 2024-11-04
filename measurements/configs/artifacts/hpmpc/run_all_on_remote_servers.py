import json
import paramiko
import sys
import threading

# Load machine credentials from JSON file
with open('machines.json') as f:
    machines = json.load(f)

# Define commands to execute
base_commands = """
cd hpmpc
sudo chmod 777 measurements/logs
rm -rf measurements/logs/*
export MAX_BITWIDTH=$(lscpu | grep -i flags | grep -q "avx512" && echo 512 || (lscpu | grep -i flags | grep -q "avx2" && echo 256 || (lscpu | grep -i flags | grep -q "sse" && echo 128 || echo 64)))
export SUPPORTED_BITWIDTHS=$(echo 1 8 16 32 64 128 256 512 | awk -v max="$MAX_BITWIDTH" '{for(i=1;i<=NF;i++) if($i<=max) printf $i (i<NF && $i<max?",":"")}')
export ITERATIONS=1
echo "Running experiments with the following parameters: PID=$PID, IP0=$IP0, IP1=$IP1, IP2=$IP2, IP3=$IP3, ITERATIONS=$ITERATIONS, SUPPORTED_BITWIDTHS=$SUPPORTED_BITWIDTHS, MAX_BITWIDTH=$MAX_BITWIDTH" 
"""

experiment_command = "sudo ./measurements/configs/artifacts/hpmpc/run_all_experiments.sh -a $IP0 -b $IP1 -c $IP2 -d $IP3 -p $PID -i $ITERATIONS -L $SUPPORTED_BITWIDTHS -D $MAX_BITWIDTH"
parse_command = "python3 measurements/parse_logs.py measurements/logs"

# Lock for synchronized printing
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
    try:
        client.connect(ip, username=username, password=password)
        stdin, stdout, stderr = client.exec_command(commands, get_pty=True)

        # Stream command output until completion
        for line in iter(stdout.readline, ""):
            with print_lock:
                print(f"[Machine {pid} - {hostname}] {line.strip()}")
        
        # Check exit status
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            with print_lock:
                print(f"[Machine {pid} - {hostname}] Commands completed successfully")
        else:
            with print_lock:
                print(f"[Machine {pid} - {hostname}] Command failed with exit status {exit_status}")
    finally:
        client.close()

# Main execution
if __name__ == "__main__":
    # Check if script is launched with an argument for a specific machine
    if len(sys.argv) > 1:
        machine_index = int(sys.argv[1])
        execute_commands_on_remote(machines[machine_index], machine_index)
    else:
        # Run in multi-threaded mode for all machines
        threads = []
        for pid, machine in enumerate(machines):
            thread = threading.Thread(target=execute_commands_on_remote, args=(machine, pid))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

