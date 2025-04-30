import re
import csv
import os
import sys
import argparse
from collections import defaultdict
            
bool_functions=[1,13,14,15,25,32,42,43,44] # List of function identifiers that are boolean-only functions

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
            if debug:
                print(f"Parsed parameters: {dict(params)}")

        # Initialize counters and lists for calculations
        pre_received = pre_sent = online_received = online_sent = 0
        pre_times = []
        online_times = []
        accuracies = []
        tests_passed = tests_total = 0

        for line in lines:
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
                match = re.search(r'Time measured to perform (\w+).*?:\s*([\d.e+-]+)s', line)
                if match:
                    if match.group(1) == 'preprocessing':
                        pre_times.append(float(match.group(2)))
                    elif match.group(1) in ['computation', 'chrono']:
                        online_times.append(float(match.group(2)))
            
            # Parse accuracy
            accuracy_match = re.search(r'accuracy\((\d+) images\): ([\d.]+)%', line)
            if accuracy_match:
                accuracies.append(float(accuracy_match.group(2)))
            
            # Parse tests passed
            tests_match = re.search(r'Passed (\d+) out of (\d+) tests', line)
            if tests_match:
                tests_passed += int(tests_match.group(1))
                tests_total += int(tests_match.group(2))

        # Calculate statistics
        pre_avg = sum(pre_times) / len(pre_times) if pre_times else 0
        pre_max = max(pre_times) if pre_times else 0
        online_avg = sum(online_times) / len(online_times) if online_times else 0
        online_max = max(online_times) if online_times else 0

        run_data.update({
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
    
    # Define the fixed headers in the desired order
    fixed_headers = [
        'ACCURACY(%)', 'TESTS_PASSED',
        'PRE_RECEIVED(MB)', 'PRE_SENT(MB)', 'ONLINE_RECEIVED(MB)', 'ONLINE_SENT(MB)',
        'PRE_AVG(s)', 'PRE_MAX(s)', 'ONLINE_AVG(s)', 'ONLINE_MAX(s)',
        'TP_PRE_AVG(Mbit/s)', 'TP_PRE_MAX(Mbit/s)', 'TP_ONLINE_AVG(Mbit/s)', 'TP_ONLINE_MAX(Mbit/s)',
        'TP_PRE_AVG(Ops/s)', 'TP_PRE_MAX(Ops/s)', 'TP_ONLINE_AVG(Ops/s)', 'TP_ONLINE_MAX(Ops/s)'
    ]
    
    # Get all unique keys from all runs
    all_keys = set()
    for run in parsed_data:
        all_keys.update(run.keys())
    
    # Separate dynamic headers from fixed headers and sort them
    dynamic_headers = sorted([key for key in all_keys if key not in fixed_headers])
    
    # Filter out headers with no data
    dynamic_headers = [header for header in dynamic_headers if any(run.get(header) for run in parsed_data)]
    fixed_headers = [header for header in fixed_headers if any(run.get(header) for run in parsed_data)]
    
    # Combine headers in the desired order: sorted dynamic headers first, then fixed headers
    fieldnames = dynamic_headers + fixed_headers

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in parsed_data:
            # Convert scientific notation to normal numbers and filter out empty values
            formatted_run = {k: f"{float(v):.6f}" if isinstance(v, (int, float)) and k != 'TESTS_PASSED' else v 
                             for k, v in run.items() if v and k in fieldnames}
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
