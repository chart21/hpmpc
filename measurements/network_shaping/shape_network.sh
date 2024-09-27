#!/bin/bash

# Function to parse JSON and extract data using jq
parse_json() {
  local json_file="${1}"
  local party_index="${2}"

  # Extract latencies and bandwidth data for the given party index
  latencies=$(jq -r ".latencies[${party_index}][]" "${json_file}")
  bandwidth=$(jq -r ".bandwidth[${party_index}][]" "${json_file}")

  if [[ -z "${latencies}" || -z "${bandwidth}" ]]; then
      echo "Error parsing latencies or bandwidth for party index ${party_index}"
      exit 1
  fi
}

# Function to initialize tc rules on a network interface
initialize_tc() {
  local iface="${1}"

  echo "Initializing tc on interface: ${iface}"

  # Initialize by clearing existing rules
  echo "Executing: tc qdisc del dev ${iface} root 2>/dev/null"
  tc qdisc del dev "${iface}" root 2>/dev/null

  # Add the root qdisc
  echo "Executing: tc qdisc add dev ${iface} root handle 1: htb default 30"
  tc qdisc add dev "${iface}" root handle 1: htb default 30
  if [ $? -eq 0 ]; then
    echo "Successfully added root qdisc to ${iface}"
  else
    echo "Error adding root qdisc to ${iface}"
  fi
}

# Function to apply tc rules to a specific IP on a network interface
apply_tc_ip_specific() {
  local iface="${1}"
  local ip="${2}"
  local latency="${3}"
  local bw="${4}"
  local classid="${5}"  # Unique class ID for each IP

  echo "Applying settings for IP ${ip} on interface ${iface}:"
  if [ "${latency}" -gt "0" ]; then
    echo "  Latency: ${latency}ms"
    # Add a netem qdisc to control latency
    echo "Executing: tc qdisc add dev ${iface} parent 1:${classid} handle ${classid}0: netem delay ${latency}ms"
    tc qdisc add dev "${iface}" parent 1:${classid} handle ${classid}0: netem delay "${latency}ms"
    if [ $? -eq 0 ]; then
      echo "  Successfully added latency control to ${iface} for ${ip}"
    else
      echo "  Error setting latency on ${iface} for ${ip}"
    fi
  else
    echo "  Skipping latency rule for IP ${ip}"
  fi

  if [ "${bw}" != "-1" ]; then
    echo "  Bandwidth: ${bw}mbit"
    # Add a class to control bandwidth for specific IP
    echo "Executing: tc class add dev ${iface} parent 1: classid 1:${classid} htb rate ${bw}mbit ceil ${bw}mbit"
    tc class add dev "${iface}" parent 1: classid 1:${classid} htb rate "${bw}mbit" ceil "${bw}mbit"
    if [ $? -eq 0 ]; then
      echo "  Successfully added bandwidth limit to ${iface} for ${ip}"
    else
      echo "  Error setting bandwidth on ${iface} for ${ip}"
    fi
  else
    echo "  Skipping bandwidth rule for IP ${ip}"
  fi

  # Filter by IP address (assuming IPv4 here)
  echo "Executing: tc filter add dev ${iface} protocol ip parent 1:0 prio 1 u32 match ip dst ${ip} flowid 1:${classid}"
  tc filter add dev "${iface}" protocol ip parent 1:0 prio 1 u32 match ip dst "${ip}" flowid 1:${classid}
  if [ $? -eq 0 ]; then
    echo "  Successfully added IP-specific filter for ${ip} on ${iface}"
  else
    echo "  Error adding filter for ${ip} on ${iface}"
  fi
}

# Default values for input arguments
PARTY=-1
JSON_FILE=""
LATENCY_DIVISOR=1
IPS=()
LATENCY_OVERRIDE=""
BANDWIDTH_OVERRIDE=""

# Parse input arguments
while getopts ":a:b:c:d:p:f:l:L:B:" opt; do
  case $opt in
    a) IPS[0]=$OPTARG ;;
    b) IPS[1]=$OPTARG ;;
    c) IPS[2]=$OPTARG ;;
    d) IPS[3]=$OPTARG ;;
    p) PARTY=$OPTARG ;;
    f) JSON_FILE=$OPTARG ;;
    l) LATENCY_DIVISOR=$OPTARG ;;
    L) LATENCY_OVERRIDE=$OPTARG ;;
    B) BANDWIDTH_OVERRIDE=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done


# Ensure the party index is valid
if [[ ${PARTY} -lt 0 ]]; then
  echo "Error: Please provide a valid party index."
  exit 1
fi

if [[ ${PARTY} = "all" ]]; then
    echo "Parties set to all, skipping network shaping"
    exit
fi

#Check if 127.0.0.1 is in the list of IPs and exit if it is
for ip in "${IPS[@]}"; do
  if [[ "${ip}" == "127.0.0.1" ]]; then
    echo "IP set to localhost, skipping network shaping"
    exit
    fi
done



# If neither L nor B is provided, read from the JSON file
if [[ -z "${LATENCY_OVERRIDE}" && -z "${BANDWIDTH_OVERRIDE}" ]]; then
  if [[ -z "${JSON_FILE}" ]]; then
    echo "Error: Please provide a valid JSON file."
    exit 1
  fi
  # Parse the JSON file and extract latencies and bandwidths for the given party
  parse_json "${JSON_FILE}" "${PARTY}"
fi

# Get network interfaces on the system
IFACES=$(ip -o -4 link show | awk -F': ' '{print $2}' | grep -v -e lo -e vir -e docker)

# Initialize tc, then apply latencies and bandwidths to each specified IP
for IFACE in ${IFACES}; do
  initialize_tc "${IFACE}" # Initialize tc root qdisc once for each interface
  
  config_index=0  # Index to track latency and bandwidth configuration
  for (( i = 0; i < 4; i++ )); do
    if [ "${i}" -eq "${PARTY}" ]; then
      echo "Skipping self configuration for index ${i}"
      continue
    fi

    IP=${IPS[i]}
    if [ -z "${IP}" ]; then
      echo "Skipping unspecified IP at index ${i}"
      continue
    fi

    if [[ -n "${LATENCY_OVERRIDE}" ]]; then
      RAW_LATENCY="${LATENCY_OVERRIDE}"
      if [[ "${RAW_LATENCY}" -lt 0 ]]; then
        LATENCY=-1
      else
        LATENCY=$(echo "scale=2; ${RAW_LATENCY} / ${LATENCY_DIVISOR}" | bc)
      fi
    else
      RAW_LATENCY=$(echo "${latencies}" | sed -n "$((config_index+1))p")
      LATENCY=$(echo "scale=2; ${RAW_LATENCY} / ${LATENCY_DIVISOR}" | bc)
    fi

    if [[ -n "${BANDWIDTH_OVERRIDE}" ]]; then
      BW="${BANDWIDTH_OVERRIDE}"
    else
      BW=$(echo "${bandwidth}" | sed -n "$((config_index+1))p")
    fi

    apply_tc_ip_specific "${IFACE}" "${IP}" "${LATENCY}" "${BW}" "$((10 + i))"
    config_index=$((config_index + 1))
  done
done

