#!/bin/bash
set -e

# Default values (can be passed via CLI or environment)
LATENCY_MS=${LATENCY_MS:-50}       # e.g. 100 for 100ms, -1 to disable
BANDWIDTH_MBIT=${BANDWIDTH_MBIT:-200}  # e.g. 100 for 100mbit, -1 to disable

# Collect all non-loopback, non-virtual interfaces
# IFACES=$(ip -o -4 link show | awk -F': ' '{print $2}' | grep -v -e lo -e vir -e docker)
IFACES="lo"

echo "Detected interfaces: $IFACES"
echo "Target latency: ${LATENCY_MS}ms | Target bandwidth: ${BANDWIDTH_MBIT}mbit"

for IFACE in $IFACES; do
  echo "Configuring interface: $IFACE"

  # Remove any existing qdisc
  tc qdisc del dev "$IFACE" root 2>/dev/null || true

  # -------------------------------
  # CASE 1: both latency and bandwidth shaping
  # -------------------------------
  if (( LATENCY_MS > 0 )) && (( BANDWIDTH_MBIT > 0 )); then
    echo " → Adding HTB + NETEM (latency + bandwidth)"
    tc qdisc add dev "$IFACE" root handle 1: htb default 10
    tc class add dev "$IFACE" parent 1: classid 1:10 htb rate ${BANDWIDTH_MBIT}mbit ceil ${BANDWIDTH_MBIT}mbit
    tc qdisc add dev "$IFACE" parent 1:10 handle 10: netem delay ${LATENCY_MS}ms

  # -------------------------------
  # CASE 2: only latency shaping
  # -------------------------------
  elif (( LATENCY_MS > 0 )) && (( BANDWIDTH_MBIT <= 0 )); then
    echo " → Adding NETEM only (latency)"
    tc qdisc add dev "$IFACE" root netem delay ${LATENCY_MS}ms

  # -------------------------------
  # CASE 3: only bandwidth shaping
  # -------------------------------
  elif (( LATENCY_MS <= 0 )) && (( BANDWIDTH_MBIT > 0 )); then
    echo " → Adding HTB only (bandwidth)"
    tc qdisc add dev "$IFACE" root handle 1: htb default 10
    tc class add dev "$IFACE" parent 1: classid 1:10 htb rate ${BANDWIDTH_MBIT}mbit ceil ${BANDWIDTH_MBIT}mbit

  # -------------------------------
  # CASE 4: both disabled
  # -------------------------------
  else
    echo " → Skipping shaping (latency=-1, bandwidth=-1)"
  fi
done

echo "✅ Network shaping applied."
