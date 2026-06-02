import re
import csv
import os
import sys
import argparse
from collections import defaultdict

bool_functions = [1, 13, 14, 15, 25, 32, 42, 43, 44]  # List of function identifiers that are boolean-only functions

def parse_size(size_str):
    values = [s.strip().rstrip('MB') for s in size_str.split(',')]
    total = 0
    for value in values:
        if value:  # Check if the value is not empty
            if 'e' in value.lower():
                mantissa, exponent = value.lower().split('e')
                total += float(mantissa) * (10 ** int(exponent))
            else:
                total += float(value)
    return total

def parse_log_file(file_path, debug=False):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content into runs
    runs = re.split(r'====== Run \d+/\d+.*?======', content)
    runs = [run.strip() for run in runs if run.strip()]  # Remove empty runs
    
    parsed_data = []
    for run_index, run in enumerate(runs, 1):
        run_data = defaultdict(float)
        lines = run.split('\n')
        
        if debug:
            print(f"Parsing Run {run_index}")
        
        # Parse the "Running:" line
        running_line = next((line for line in lines if line.startswith('Running:')), None)
        if running_line:
            params = re.findall(r'(\w+)=(\w+)', running_line)
            run_data.update(params)

        # --- Triples Accumulator Init ---
        triples_accumulators = defaultdict(int)
        parties_seen = set()
        # -------------------------------------

        # Timings for each triple generation step (collect list of chronos to average)
        triple_step_times_occurrences = defaultdict(list)

        # NN_STATS (Aggregated)
        nn_stats_occurrences = defaultdict(list)

        # TRIPLE_STATS (Aggregated)
        triple_stats_aggregated_occurrences = defaultdict(list)
        # TRIPLE_STATS (Total)
        triple_stats_total_occurrences = []

        current_step = None

        # Initialize counters and lists for calculations
        triple_gen_total = 0.0
        pre_received = pre_sent = online_received = online_sent = 0
        pre_times = []
        online_times = []
        accuracies = []
        tests_passed = tests_total = 0
        key_exchange_time = 0.0
        key_exchange_sent_pre = 0.0
        key_exchange_received_pre = 0.0

        for line in lines:
            # 1. Track current triple generation step name
            step_match = re.search(r'P\d+(?:, PRE)?, PID\d+: Generating ([A-Za-z0-9_]+) (?:[Tt]riples|[Tt]uples)', line)
            if step_match:
                current_step = step_match.group(1).strip()

            # 2. Parse beaver triple generation times
            if 'beaver triple generation chrono:' in line:
                time_match = re.search(r'(?:([A-Z0-9_]+)\s+)?P\d+, PID\d+: Time measured to perform beaver triple generation chrono:\s*([\d.e+-]+)s', line)
                if time_match:
                    prefix = time_match.group(1)
                    val = float(time_match.group(2))
                    step_name = prefix
                    if step_name == "CONVOLUTION":
                        step_name = "CONV"
                    elif step_name == "BATCHNORM":
                        step_name = "BATCHNORM2D"
                    if not step_name:
                        step_name = current_step
                    if step_name:
                        triple_step_times_occurrences[step_name].append(val)

            # 3. Parse "Required" lines
            if 'Required' in line:
                req_match = re.search(r'P(\d+).*?:\s*(.*?)\s*Required.*:\s*(\d+)', line)
                if req_match:
                    party_id = req_match.group(1)
                    req_type = req_match.group(2).strip()
                    count = int(req_match.group(3))
                    
                    parties_seen.add(party_id)
                    triples_accumulators[req_type] += count

            # 4. Parse NN_STATS (Aggregated)
            if 'NN_STATS (Aggregated)' in line:
                nn_match = re.search(
                    r'P\d+:\s+--NN_STATS\s+\(Aggregated\)--\s+([A-Z0-9_]+)\s+MB\s+SENT:\s*([\d.]+)\s+MB\s+RECEIVED:\s*([\d.]+)\s+MB\s+SENT\s+PRE:\s*([\d.]+)\s+MB\s+RECEIVED\s+PRE:\s*([\d.]+)\s+ms\s+LIVE:\s*([\d.]+)\s+ms\s+PRE:\s*([\d.]+)',
                    line
                )
                if nn_match:
                    layer = nn_match.group(1).strip()
                    nn_stats_occurrences[layer].append({
                        'MB_SENT': float(nn_match.group(2)),
                        'MB_RECEIVED': float(nn_match.group(3)),
                        'MB_SENT_PRE': float(nn_match.group(4)),
                        'MB_RECEIVED_PRE': float(nn_match.group(5)),
                        'ms_LIVE': float(nn_match.group(6)),
                        'ms_PRE': float(nn_match.group(7)),
                    })

            # 5. Parse TRIPLE_STATS (Aggregated)
            if 'TRIPLE_STATS (Aggregated)' in line:
                t_match = re.search(
                    r'P\d+,\s+PID\d+:\s+--TRIPLE_STATS\s+\(Aggregated\)--\s+([A-Z0-9_]+)\s+MB\s+SENT\s+PRE:\s*([\d.e+-]+)\s+MB\s+RECEIVED\s+PRE:\s*([\d.e+-]+)\s+s\s+PRE:\s*([\d.e+-]+)',
                    line
                )
                if t_match:
                    category = t_match.group(1).strip()
                    triple_stats_aggregated_occurrences[category].append({
                        'MB_SENT_PRE': float(t_match.group(2)),
                        'MB_RECEIVED_PRE': float(t_match.group(3)),
                        's_PRE': float(t_match.group(4)),
                    })
                    
            # 6. Parse TRIPLE_STATS (Total)
            elif 'TRIPLE_STATS (Total)' in line:
                t_match = re.search(
                    r'P\d+,\s+PID\d+:\s+--TRIPLE_STATS\s+\(Total\)--\s+MB\s+SENT\s+PRE:\s*([\d.e+-]+)\s+MB\s+RECEIVED\s+PRE:\s*([\d.e+-]+)\s+s\s+PRE:\s*([\d.e+-]+)',
                    line
                )
                if t_match:
                    triple_stats_total_occurrences.append({
                        'MB_SENT_PRE': float(t_match.group(1)),
                        'MB_RECEIVED_PRE': float(t_match.group(2)),
                        's_PRE': float(t_match.group(3)),
                    })

            # --- Original parsing logic ---
            if 'data[MiB]:' in line:
                triple_match = re.search(r'data\[MiB\]:\s*([\d.e+-]+)', line)
                if triple_match:
                    triple_gen_total += float(triple_match.group(1))
            if 'Sending' in line or 'Receiving' in line:
                match = re.search(r'(Sending|Receiving).*?:(.*?)(?=\s*$)', line)
                if match:
                    sizes = parse_size(match.group(2))
                    if 'PRE' in line:
                        if 'Sending' in line:
                            pre_sent += sizes
                        else:
                            pre_received += sizes
                    elif 'ONLINE' in line:
                        if 'Sending' in line:
                            online_sent += sizes
                        else:
                            online_received += sizes
            elif 'Time measured to perform' in line:
                match = re.search(r'Time measure[d]? to perform ([\w\s]+?):\s*([\d.e+-]+)s', line)
                if match:
                    label = match.group(1).strip()
                    value = float(match.group(2))
                    if label == 'preprocessing chrono':
                        pre_times.append(value)
                    elif label == 'computation chrono':
                        online_times.append(value)
            
            # Parse accuracy
            accuracy_match = re.search(r'accuracy\((\d+) images\): ([\d.]+)%', line)
            if accuracy_match:
                accuracies.append(float(accuracy_match.group(2)))
            
            # Parse tests passed
            tests_match = re.search(r'Passed (\d+) out of (\d+) tests', line)
            if tests_match:
                tests_passed += int(tests_match.group(1))
                tests_total += int(tests_match.group(2))

            # Parse Key Exchange stats
            if 'Key exchange' in line or 'key exchange' in line:
                ke_time_match = re.search(r'P\d+,\s+PID\d+:\s+Key\s+exchange\s+s\s+PRE:\s*([\d.e+-]+)', line, re.IGNORECASE)
                if ke_time_match:
                    key_exchange_time += float(ke_time_match.group(1))
                ke_mb_match = re.search(r'P\d+,\s+PID\d+:\s+Key\s+exchange\s+MB\s+SENT\s+PRE:\s*([\d.e+-]+)\s+MB\s+RECEIVED\s+PRE:\s*([\d.e+-]+)', line, re.IGNORECASE)
                if ke_mb_match:
                    key_exchange_sent_pre += float(ke_mb_match.group(1))
                    key_exchange_received_pre += float(ke_mb_match.group(2))

        # --- Calculate Final Triple Stats ---
        num_parties = len(parties_seen) if len(parties_seen) > 0 else 1
        for t_type, t_count in triples_accumulators.items():
            val_raw = t_count / num_parties
            
            # Formulate backwards-compatible names and absolute counts with unit (#)
            if t_type.endswith(('Triples', 'Tuples', 'Multiplications')):
                col_name_raw = f"{t_type} Required (#)"
            else:
                col_name_raw = f"{t_type} Triples Required (#)"
            
            run_data[col_name_raw] = val_raw
        # -----------------------------------------

        # --- Add Triple Steps Gen Times (Averages) ---
        for step_name, occurrences in triple_step_times_occurrences.items():
            if occurrences:
                run_data[f"{step_name} Gen (s)"] = sum(occurrences) / len(occurrences)
            
        # --- Add NN Stats (Aggregated: sum MB numbers, average times) ---
        for layer, occurrences in nn_stats_occurrences.items():
            n = len(occurrences)
            if n > 0:
                run_data[f"NN_{layer}_SENT(MB)"] = sum(occ['MB_SENT'] for occ in occurrences)
                run_data[f"NN_{layer}_RECEIVED(MB)"] = sum(occ['MB_RECEIVED'] for occ in occurrences)
                run_data[f"NN_{layer}_SENT_PRE(MB)"] = sum(occ['MB_SENT_PRE'] for occ in occurrences)
                run_data[f"NN_{layer}_RECEIVED_PRE(MB)"] = sum(occ['MB_RECEIVED_PRE'] for occ in occurrences)
                run_data[f"NN_{layer}_LIVE(ms)"] = sum(occ['ms_LIVE'] for occ in occurrences) / n
                run_data[f"NN_{layer}_PRE(ms)"] = sum(occ['ms_PRE'] for occ in occurrences) / n

        # --- Add Triple Stats (Aggregated: sum MB numbers, average times) ---
        for category, occurrences in triple_stats_aggregated_occurrences.items():
            n = len(occurrences)
            if n > 0:
                run_data[f"TRIPLE_STATS_{category}_SENT_PRE(MB)"] = sum(occ['MB_SENT_PRE'] for occ in occurrences)
                run_data[f"TRIPLE_STATS_{category}_RECEIVED_PRE(MB)"] = sum(occ['MB_RECEIVED_PRE'] for occ in occurrences)
                run_data[f"TRIPLE_STATS_{category}_PRE(s)"] = sum(occ['s_PRE'] for occ in occurrences) / n

        # --- Add Triple Stats (Total: sum MB numbers, average times) ---
        total_triple_mb_sent_pre = 0.0
        total_triple_mb_received_pre = 0.0
        n_total = len(triple_stats_total_occurrences)
        if n_total > 0:
            total_triple_mb_sent_pre = sum(occ['MB_SENT_PRE'] for occ in triple_stats_total_occurrences)
            total_triple_mb_received_pre = sum(occ['MB_RECEIVED_PRE'] for occ in triple_stats_total_occurrences)
            run_data["TRIPLE_STATS_Total_SENT_PRE(MB)"] = total_triple_mb_sent_pre
            run_data["TRIPLE_STATS_Total_RECEIVED_PRE(MB)"] = total_triple_mb_received_pre
            run_data["TRIPLE_STATS_Total_PRE(s)"] = sum(occ['s_PRE'] for occ in triple_stats_total_occurrences) / n_total

        # Store individual key exchange metrics if they exist
        if key_exchange_time > 0 or key_exchange_sent_pre > 0 or key_exchange_received_pre > 0:
            run_data['KEY_EXCHANGE_PRE(s)'] = key_exchange_time
            run_data['KEY_EXCHANGE_SENT_PRE(MB)'] = key_exchange_sent_pre
            run_data['KEY_EXCHANGE_RECEIVED_PRE(MB)'] = key_exchange_received_pre

        # --- Introduce total pre MB (sum of pre MB, Aggregated triple MB, and Key Exchange MB) ---
        run_data['TOTAL_PRE_RECEIVED(MB)'] = pre_received + total_triple_mb_received_pre + key_exchange_received_pre
        run_data['TOTAL_PRE_SENT(MB)'] = pre_sent + total_triple_mb_sent_pre + key_exchange_sent_pre

        # --- Sums of Sent and Received ---
        run_data['PRE_SENT+RECV(MB)'] = pre_sent + pre_received
        run_data['TOTAL_PRE_SENT+RECV(MB)'] = run_data['TOTAL_PRE_SENT(MB)'] + run_data['TOTAL_PRE_RECEIVED(MB)']
        run_data['ONLINE_SENT+RECV(MB)'] = online_sent + online_received

        # Calculate statistics (including key exchange time in preprocessing)
        if key_exchange_time > 0:
            if pre_times:
                pre_times = [t + key_exchange_time for t in pre_times]
            else:
                pre_times = [key_exchange_time]

        pre_avg = sum(pre_times) / len(pre_times) if pre_times else 0
        pre_max = max(pre_times) if pre_times else 0
        online_avg = sum(online_times) / len(online_times) if online_times else 0
        online_max = max(online_times) if online_times else 0

        run_data.update({
            'TRPLE_GEN(MB)': triple_gen_total,
            'PRE_RECEIVED(MB)': pre_received,
            'PRE_SENT(MB)': pre_sent,
            'PRE_MAX(s)': pre_max,
            'PRE_AVG(s)': pre_avg,
            'ONLINE_RECEIVED(MB)': online_received,
            'ONLINE_SENT(MB)': online_sent,
            'ONLINE_MAX(s)': online_max,
            'ONLINE_AVG(s)': online_avg,
            'TP_PRE_AVG(Mbit/s)': (pre_received * 8) / pre_avg if pre_avg else 0,
            'TP_PRE_MAX(Mbit/s)': (pre_received * 8) / pre_max if pre_max else 0,
            'TP_ONLINE_AVG(Mbit/s)': (online_received * 8) / online_avg if online_avg else 0,
            'TP_ONLINE_MAX(Mbit/s)': (online_received * 8) / online_max if online_max else 0,
        })

        # Add accuracy if applicable
        if accuracies:
            run_data['ACCURACY(%)'] = sum(accuracies) / len(accuracies)

        # Add tests passed if applicable
        if tests_total > 0:
            run_data['TESTS_PASSED'] = f"{tests_passed}/{tests_total}"

        # Calculate Ops/s if applicable
        splitroles_factor = 1 
        if 'SPLITROLES' in run_data:
            splitroles_factor = 6 if run_data['SPLITROLES'] == '1' else splitroles_factor
            splitroles_factor = 24 if run_data['SPLITROLES'] == '2' else splitroles_factor
            splitroles_factor = 24 if run_data['SPLITROLES'] == '3' else splitroles_factor
        num_processes = 1
        if 'PROCESS_NUM' in run_data:
            num_processes = int(run_data['PROCESS_NUM'])
        if all(key in run_data for key in ['BITLENGTH', 'DATTYPE', 'NUM_INPUTS','FUNCTION_IDENTIFIER']):
            bitlength = float(run_data['BITLENGTH'])
            dattype = float(run_data['DATTYPE'])
            num_inputs = float(run_data['NUM_INPUTS'])
            function_identifier = int(run_data['FUNCTION_IDENTIFIER'])
            if function_identifier in bool_functions:
                run_data.update({
                    'TP_PRE_AVG(Ops/s)': (num_inputs * dattype * splitroles_factor * num_processes) / pre_avg if pre_avg else 0,
                    'TP_PRE_MAX(Ops/s)': (num_inputs * dattype * splitroles_factor * num_processes) / pre_max if pre_max else 0,
                    'TP_ONLINE_AVG(Ops/s)': (num_inputs * dattype * splitroles_factor * num_processes) / online_avg if online_avg else 0,
                    'TP_ONLINE_MAX(Ops/s)': (num_inputs * dattype * splitroles_factor * num_processes) / online_max if online_max else 0,
                })
            else:
                run_data.update({
                    'TP_PRE_AVG(Ops/s)': (num_inputs * (dattype / bitlength) * splitroles_factor * num_processes) / pre_avg if pre_avg else 0,
                    'TP_PRE_MAX(Ops/s)': (num_inputs * (dattype / bitlength) * splitroles_factor * num_processes) / pre_max if pre_max else 0,
                    'TP_ONLINE_AVG(Ops/s)': (num_inputs * (dattype / bitlength) * splitroles_factor * num_processes) / online_avg if online_avg else 0,
                    'TP_ONLINE_MAX(Ops/s)': (num_inputs * (dattype / bitlength) * splitroles_factor * num_processes) / online_max if online_max else 0,
                })

        if debug:
            print(f"Parsed data for Run {run_index}: {dict(run_data)}")
        parsed_data.append(run_data)

    if debug:
        print(f"Total parsed runs: {len(parsed_data)}")
    return parsed_data


def write_csv(parsed_data, output_file):
    if not parsed_data:
        print(f"No data to write to CSV: {output_file}")
        return
    
    # Define the fixed headers
    total_metrics = [
        'ACCURACY(%)', 'TESTS_PASSED',
        'PRE_RECEIVED(MB)', 'PRE_SENT(MB)', 'PRE_SENT+RECV(MB)',
        'KEY_EXCHANGE_RECEIVED_PRE(MB)', 'KEY_EXCHANGE_SENT_PRE(MB)',
        'TOTAL_PRE_RECEIVED(MB)', 'TOTAL_PRE_SENT(MB)', 'TOTAL_PRE_SENT+RECV(MB)',
        'ONLINE_RECEIVED(MB)', 'ONLINE_SENT(MB)', 'ONLINE_SENT+RECV(MB)',
        'KEY_EXCHANGE_PRE(s)',
        'PRE_AVG(s)', 'PRE_MAX(s)', 'ONLINE_AVG(s)', 'ONLINE_MAX(s)',
        'TP_PRE_AVG(Mbit/s)', 'TP_PRE_MAX(Mbit/s)', 'TP_ONLINE_AVG(Mbit/s)', 'TP_ONLINE_MAX(Mbit/s)',
        'TP_PRE_AVG(Ops/s)', 'TP_PRE_MAX(Ops/s)', 'TP_ONLINE_AVG(Ops/s)', 'TP_ONLINE_MAX(Ops/s)',
        'TRPLE_GEN(MB)'
    ]
    
    # Get all unique keys from all runs
    all_keys = set()
    for run in parsed_data:
        all_keys.update(run.keys())
    
    def get_column_category(k):
        # Category 4: Number of triples (ends with "Required (#)")
        if k.endswith('Required (#)'):
            return 4
        # Category 3: Triple and NN timings and comm
        if k.startswith(('NN_', 'TRIPLE_STATS_')) or k.endswith('Gen (s)'):
            return 3
        # Category 2: Total timings and comm
        if k in total_metrics:
            return 2
        # Category 1: Compile/input options (everything else)
        return 1

    def sort_key(k):
        cat = get_column_category(k)
        if cat == 2:
            try:
                sub_order = total_metrics.index(k)
            except ValueError:
                sub_order = len(total_metrics)
            return (cat, sub_order, k)
        else:
            return (cat, 0, k)

    # Sort all keys based on their category and sub-order
    fieldnames = sorted(list(all_keys), key=sort_key)
    
    # Filter out empty columns (Checking `is not None and != ""` to preserve 0s)
    fieldnames = [h for h in fieldnames if any(run.get(h) is not None and run.get(h) != "" for run in parsed_data)]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in parsed_data:
            formatted_run = {}
            for k in fieldnames:
                v = run.get(k)
                if v is not None and v != "":
                    if isinstance(v, (int, float)) and k != 'TESTS_PASSED':
                        if isinstance(v, int) or k.endswith('Required (#)'):
                            formatted_run[k] = str(int(round(v)))
                        else:
                            formatted_run[k] = f"{float(v):.6f}"
                    else:
                        formatted_run[k] = v
            writer.writerow(formatted_run)
    print(f"CSV file has been created: {output_file}")

def process_log_file(log_file_path, debug=False):
    parsed_data = parse_log_file(log_file_path, debug)
    output_csv_path = log_file_path.rsplit('.', 1)[0] + '.csv'
    write_csv(parsed_data, output_csv_path)

def main():
    parser = argparse.ArgumentParser(description="Process log files and generate CSV output.")
    parser.add_argument("path", help="Path to log file or directory containing log files")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        process_log_file(args.path, args.debug)
    elif os.path.isdir(args.path):
        for filename in os.listdir(args.path):
            if filename.endswith('.log'):
                file_path = os.path.join(args.path, filename)
                process_log_file(file_path, args.debug)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()

