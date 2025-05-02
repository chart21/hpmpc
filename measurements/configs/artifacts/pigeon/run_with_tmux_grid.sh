#!/bin/bash

# Parse optional arguments, Reduced, GPU
while getopts "R:g:" opt; do
  case $opt in
    R) R_ARG="-R $OPTARG" ;;
    g) G_ARG="-g $OPTARG" ;;
    *) echo "Usage: $0 [-R value] [-g value]"; exit 1 ;;
  esac
done

# Kill existing tmux session if it exists
tmux kill-session -t grid 2>/dev/null

# Start a new detached tmux session
tmux new-session -d -s grid

# Base command with optional args
ARGS=("$R_ARG" "$G_ARG")
CMD="python3 run_all_on_remote_servers.py"

# Function to run the command on a given machine ID
run_on_machine() {
  tmux send-keys "$CMD $1 ${ARGS[*]}" C-m
}

# Launch scripts on 4 machines in a 2x2 tmux grid
run_on_machine 0
tmux split-window -h
run_on_machine 1
tmux split-window -v
run_on_machine 2
tmux select-pane -t 0
tmux split-window -v
run_on_machine 3

# Arrange panes and attach to session
tmux select-layout tiled
tmux attach -t grid
