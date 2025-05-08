#!/bin/bash

# Parse -R and -g options
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -R|-g|-O)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Usage: $0 [-R value] [-g value] [-O value]"
      exit 1
      ;;
  esac
done

# Function to build command and quote it safely
build_cmd() {
  local machine_idx=$1
  local cmd=(python3 run_all_on_remote_servers.py -p "$machine_idx" "${EXTRA_ARGS[@]}")

  # Print command as a single string with proper quoting
  printf "%q " "${cmd[@]}"
}

# Kill existing tmux session if it exists
tmux kill-session -t grid 2>/dev/null

# Start a new tmux session named 'grid'
tmux new-session -d -s grid

# Split the window into a 2x2 grid and run the Python script for each machine
tmux send-keys "$(build_cmd 0)" C-m    # Machine 0
tmux split-window -h
tmux send-keys "$(build_cmd 1)" C-m    # Machine 1
tmux split-window -v
tmux send-keys "$(build_cmd 2)" C-m    # Machine 2
tmux select-pane -t 0
tmux split-window -v
tmux send-keys "$(build_cmd 3)" C-m    # Machine 3

# Adjust layout to ensure a 2x2 grid
tmux select-layout tiled

# Attach to the tmux session
tmux attach -t grid
