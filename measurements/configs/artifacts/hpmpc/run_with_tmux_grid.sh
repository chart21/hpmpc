#!/bin/bash

while getopts "g:R:" opt; do
  case $opt in
    g) g_value="$OPTARG" ;;
    R) R_value="$OPTARG" ;;
    *) echo "Invalid option: -$opt" >&2; exit 1 ;;
  esac
done

arg_string=""
if [[ -n $g_value ]]; then
  arg_string+=" -g $g_value"
fi
if [[ -n $R_value ]]; then
  arg_string+=" -R $R_value"
fi

#if exists kill tmux session grid
tmux kill-session -t grid

# Start a new tmux session named 'grid'
tmux new-session -d -s grid

# Split the window into a 2x2 grid and run the Python script for each machine
tmux send-keys "python3 run_all_on_remote_servers.py ${arg_string} 0" C-m    # Machine 0
tmux split-window -h
tmux send-keys "python3 run_all_on_remote_servers.py ${arg_string} 1" C-m    # Machine 1
tmux split-window -v
tmux send-keys "python3 run_all_on_remote_servers.py ${arg_string} 2" C-m    # Machine 2
tmux select-pane -t 0
tmux split-window -v
tmux send-keys "python3 run_all_on_remote_servers.py ${arg_string} 3" C-m    # Machine 3

# Adjust layout to ensure a 2x2 grid
tmux select-layout tiled

# Attach to the tmux session
tmux attach -t grid

