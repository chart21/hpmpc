import re
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from decimal import Decimal, getcontext

# Set decimal precision
getcontext().prec = 8

def parse_data(file_content):
    records = []
    current_protocol = None
    current_data = []

    for line in file_content.splitlines():
        if line.startswith("Running:"):
            if current_data and current_protocol:
                records.append((current_protocol, current_data))
                current_data = []
            current_protocol = line
        else:
            current_data.append(line)
    
    if current_data and current_protocol:
        records.append((current_protocol, current_data))
    
    return records

def process_data(records):
    protocol_data = defaultdict(list)

    for protocol, data_lines in records:
        layer_data = defaultdict(lambda: {
            'MB_SENT': Decimal(0),
            'MB_SENT_PRE': Decimal(0),
            'MB_RECEIVED': Decimal(0),
            'MB_RECEIVED_PRE': Decimal(0),
            'ms_LIVE': [],
            'ms_PRE': []
        })
        
        total_data = {
            'MB_SENT': Decimal(0),
            'MB_SENT_PRE': Decimal(0),
            'MB_RECEIVED': Decimal(0),
            'MB_RECEIVED_PRE': Decimal(0),
            'ms_PREPROCESSING': [],
            'ms_COMPUTATION': [],
        }
        
        for line in data_lines:
            match_individual = re.match(
                r"P\d+: --NN_STATS \(Individual\)-- ID: (\d+) (\w+)\s+MB SENT:\s*(\d+\.\d+)\s+MB RECEIVED:\s*(\d+\.\d+)\s+"
                r"MB SENT PRE:\s*(\d+\.\d+)\s+MB RECEIVED PRE:\s*(\d+\.\d+)\s+ms LIVE:\s*(\d+\.\d+)\s+ms PRE:\s*(\d+\.\d+)", 
                line
            )
            match_individual_simple = re.match(
                r"P\d+: --NN_STATS \(Individual\)-- ID: (\d+) (\w+)\s+MB SENT:\s*(\d+\.\d+)\s+MB RECEIVED:\s*(\d+\.\d+)\s+"
                r"ms LIVE:\s*(\d+\.\d+)", 
                line
            )
            match_aggregated = re.match(
                r"P\d+: --NN_STATS \(Aggregated\)-- (\w+)\s+MB SENT:\s*(\d+\.\d+)\s+MB RECEIVED:\s*(\d+\.\d+)\s+"
                r"MB SENT PRE:\s*(\d+\.\d+)\s+MB RECEIVED PRE:\s*(\d+\.\d+)\s+ms LIVE:\s*(\d+\.\d+)\s+ms PRE:\s*(\d+\.\d+)", 
                line
            )
            match_aggregated_simple = re.match(
                r"P\d+: --NN_STATS \(Aggregated\)-- (\w+)\s+MB SENT:\s*(\d+\.\d+)\s+MB RECEIVED:\s*(\d+\.\d+)\s+"
                r"ms LIVE:\s*(\d+\.\d+)", 
                line
            )
            match_preprocessing = re.match(
                r"P\d+, PID\d+: Time measured to perform preprocessing chrono:\s*(\d+\.\d+)s", 
                line
            )
            match_computation = re.match(
                r"P\d+, PID\d+: Time measured to perform computation chrono:\s*(\d+\.\d+)s", 
                line
            )

            if match_individual:
                layer_id, layer_type, mb_sent, mb_received, mb_sent_pre, mb_received_pre, ms_live, ms_pre = match_individual.groups()
                key = (layer_type, layer_id)
                layer_data[key]['MB_SENT'] += Decimal(mb_sent)
                layer_data[key]['MB_SENT_PRE'] += Decimal(mb_sent_pre or 0)
                layer_data[key]['MB_RECEIVED'] += Decimal(mb_received)
                layer_data[key]['MB_RECEIVED_PRE'] += Decimal(mb_received_pre or 0)
                layer_data[key]['ms_LIVE'].append(Decimal(ms_live))
                if ms_pre:
                    layer_data[key]['ms_PRE'].append(Decimal(ms_pre))
            
            elif match_individual_simple:
                layer_id, layer_type, mb_sent, mb_received, ms_live = match_individual_simple.groups()
                key = (layer_type, layer_id)
                layer_data[key]['MB_SENT'] += Decimal(mb_sent)
                layer_data[key]['MB_RECEIVED'] += Decimal(mb_received)
                layer_data[key]['ms_LIVE'].append(Decimal(ms_live))

            elif match_aggregated:
                layer_type, mb_sent, mb_received, mb_sent_pre, mb_received_pre, ms_live, ms_pre = match_aggregated.groups()
                key = (layer_type, 'Aggregated')
                layer_data[key]['MB_SENT'] += Decimal(mb_sent)
                layer_data[key]['MB_SENT_PRE'] += Decimal(mb_sent_pre or 0)
                layer_data[key]['MB_RECEIVED'] += Decimal(mb_received)
                layer_data[key]['MB_RECEIVED_PRE'] += Decimal(mb_received_pre or 0)
                layer_data[key]['ms_LIVE'].append(Decimal(ms_live))
                if ms_pre:
                    layer_data[key]['ms_PRE'].append(Decimal(ms_pre))
                total_data['MB_SENT'] += Decimal(mb_sent)
                total_data['MB_SENT_PRE'] += Decimal(mb_sent_pre or 0)
                total_data['MB_RECEIVED'] += Decimal(mb_received)
                total_data['MB_RECEIVED_PRE'] += Decimal(mb_received_pre or 0)
            
            elif match_aggregated_simple:
                layer_type, mb_sent, mb_received, ms_live = match_aggregated_simple.groups()
                key = (layer_type, 'Aggregated')
                layer_data[key]['MB_SENT'] += Decimal(mb_sent)
                layer_data[key]['MB_RECEIVED'] += Decimal(mb_received)
                layer_data[key]['ms_LIVE'].append(Decimal(ms_live))
                total_data['MB_SENT'] += Decimal(mb_sent)
                total_data['MB_RECEIVED'] += Decimal(mb_received)
            
            elif match_preprocessing:
                chrono_time = Decimal(match_preprocessing.group(1)) * 1000  # Convert to ms
                total_data['ms_PREPROCESSING'].append(chrono_time)
            
            elif match_computation:
                chrono_time = Decimal(match_computation.group(1)) * 1000  # Convert to ms
                total_data['ms_COMPUTATION'].append(chrono_time)

        protocol_data[protocol].append((layer_data, total_data))
    
    return protocol_data

def calculate_statistics_within_configuration(layer_data, total_data):
    statistics = []

    for (layer_type, layer_id), values in layer_data.items():
        ms_live_avg = np.mean(values['ms_LIVE']) if values['ms_LIVE'] else Decimal(0)
        ms_live_max = np.max(values['ms_LIVE']) if values['ms_LIVE'] else Decimal(0)
        ms_live_stddev = np.std(values['ms_LIVE']) if len(values['ms_LIVE']) > 1 else Decimal(0)

        ms_pre_avg = np.mean(values['ms_PRE']) if values['ms_PRE'] else Decimal(0)
        ms_pre_max = np.max(values['ms_PRE']) if values['ms_PRE'] else Decimal(0)
        ms_pre_stddev = np.std(values['ms_PRE']) if len(values['ms_PRE']) > 1 else Decimal(0)

        statistics.append({
            'Layer_Type': layer_type,
            'Layer_ID': layer_id,
            'MB_SENT': values['MB_SENT'],
            'MB_SENT_PRE': values['MB_SENT_PRE'],
            'MB_RECEIVED': values['MB_RECEIVED'],
            'MB_RECEIVED_PRE': values['MB_RECEIVED_PRE'],
            'ms_LIVE Avg (mean)': ms_live_avg,
            'ms_LIVE Max (mean)': ms_live_max,
            'ms_LIVE Avg (stddev)': ms_live_stddev,
            'ms_LIVE Max (stddev)': ms_live_stddev,
            'ms_PRE Avg (mean)': ms_pre_avg,
            'ms_PRE Max (mean)': ms_pre_max,
            'ms_PRE Avg (stddev)': ms_pre_stddev,
            'ms_PRE Max (stddev)': ms_pre_stddev,
        })
    
    ms_preprocessing_avg = np.mean(total_data['ms_PREPROCESSING']) if total_data['ms_PREPROCESSING'] else Decimal(0)
    ms_preprocessing_max = np.max(total_data['ms_PREPROCESSING']) if total_data['ms_PREPROCESSING'] else Decimal(0)
    ms_computation_avg = np.mean(total_data['ms_COMPUTATION']) if total_data['ms_COMPUTATION'] else Decimal(0)
    ms_computation_max = np.max(total_data['ms_COMPUTATION']) if total_data['ms_COMPUTATION'] else Decimal(0)

    ms_live_avg_total = ms_computation_avg
    ms_live_max_total = ms_computation_max

    ms_live_stddev_preprocessing = np.std(total_data['ms_PREPROCESSING']) if len(total_data['ms_PREPROCESSING']) > 1 else Decimal(0)
    ms_live_stddev_computation = np.std(total_data['ms_COMPUTATION']) if len(total_data['ms_COMPUTATION']) > 1 else Decimal(0)
    ms_live_stddev_total = ms_live_stddev_preprocessing + ms_live_stddev_computation

    ms_pre_avg_total = ms_preprocessing_avg  # Total PRE is based on preprocessing times
    ms_pre_max_total = ms_preprocessing_max
    ms_pre_stddev_total = ms_live_stddev_preprocessing

    statistics.append({
        'Layer_Type': 'Total',
        'Layer_ID': 'Total',
        'MB_SENT': total_data['MB_SENT'],
        'MB_SENT_PRE': total_data['MB_SENT_PRE'],
        'MB_RECEIVED': total_data['MB_RECEIVED'],
        'MB_RECEIVED_PRE': total_data['MB_RECEIVED_PRE'],
        'ms_LIVE Avg (mean)': ms_live_avg_total,
        'ms_LIVE Max (mean)': ms_live_max_total,
        'ms_LIVE Avg (stddev)': ms_live_stddev_total,
        'ms_LIVE Max (stddev)': ms_live_stddev_total,
        'ms_PRE Avg (mean)': ms_pre_avg_total,
        'ms_PRE Max (mean)': ms_pre_max_total,
        'ms_PRE Avg (stddev)': ms_pre_stddev_total,
        'ms_PRE Max (stddev)': ms_pre_stddev_total,
    })

    return statistics

def aggregate_identical_configurations(statistics_by_protocol):
    aggregated_statistics = {}

    for protocol, statistics_list in statistics_by_protocol.items():
        combined_statistics = defaultdict(lambda: {
            'MB_SENT': Decimal(0),
            'MB_SENT_PRE': Decimal(0),
            'MB_RECEIVED': Decimal(0),
            'MB_RECEIVED_PRE': Decimal(0),
            'ms_LIVE_avg': [],
            'ms_LIVE_max': [],
            'ms_PRE_avg': [],
            'ms_PRE_max': []
        })

        for statistics in statistics_list:
            for entry in statistics:
                key = (entry['Layer_Type'], entry['Layer_ID'])
                combined_statistics[key]['MB_SENT'] += entry['MB_SENT']
                combined_statistics[key]['MB_SENT_PRE'] += entry['MB_SENT_PRE']
                combined_statistics[key]['MB_RECEIVED'] += entry['MB_RECEIVED']
                combined_statistics[key]['MB_RECEIVED_PRE'] += entry['MB_RECEIVED_PRE']
                combined_statistics[key]['ms_LIVE_avg'].append(entry['ms_LIVE Avg (mean)'])
                combined_statistics[key]['ms_LIVE_max'].append(entry['ms_LIVE Max (mean)'])
                combined_statistics[key]['ms_PRE_avg'].append(entry['ms_PRE Avg (mean)'])
                combined_statistics[key]['ms_PRE_max'].append(entry['ms_PRE Max (mean)'])

        aggregated_statistics[protocol] = []
        for (layer_type, layer_id), values in combined_statistics.items():
            ms_live_avg = np.mean(values['ms_LIVE_avg'])
            ms_live_stddev = np.std(values['ms_LIVE_avg']) if len(values['ms_LIVE_avg']) > 1 else Decimal(0)
            ms_live_max = np.mean(values['ms_LIVE_max'])
            ms_live_max_stddev = np.std(values['ms_LIVE_max']) if len(values['ms_LIVE_max']) > 1 else Decimal(0)

            ms_pre_avg = np.mean(values['ms_PRE_avg'])
            ms_pre_stddev = np.std(values['ms_PRE_avg']) if len(values['ms_PRE_avg']) > 1 else Decimal(0)
            ms_pre_max = np.mean(values['ms_PRE_max'])
            ms_pre_max_stddev = np.std(values['ms_PRE_max']) if len(values['ms_PRE_max']) > 1 else Decimal(0)

            mb_sent_avg = values['MB_SENT'] / Decimal(len(statistics_list))
            mb_sent_pre_avg = values['MB_SENT_PRE'] / Decimal(len(statistics_list))
            mb_received_avg = values['MB_RECEIVED'] / Decimal(len(statistics_list))
            mb_received_pre_avg = values['MB_RECEIVED_PRE'] / Decimal(len(statistics_list))

            gbit_s_live_avg = (mb_received_avg * Decimal(8)) / ms_live_avg if ms_live_avg else Decimal(0)
            gbit_s_live_stddev = gbit_s_live_avg * (ms_live_stddev / ms_live_avg) if ms_live_avg else Decimal(0)
            gbit_s_live_max = (mb_received_avg * Decimal(8)) / ms_live_max if ms_live_max else Decimal(0)
            gbit_s_live_max_stddev = gbit_s_live_max * (ms_live_max_stddev / ms_live_max) if ms_live_max else Decimal(0)

            gbit_s_pre_avg = (mb_received_pre_avg * Decimal(8)) / ms_pre_avg if ms_pre_avg else Decimal(0)
            gbit_s_pre_stddev = gbit_s_pre_avg * (ms_pre_stddev / ms_pre_avg) if ms_pre_avg else Decimal(0)
            gbit_s_pre_max = (mb_received_pre_avg * Decimal(8)) / ms_pre_max if ms_pre_max else Decimal(0)
            gbit_s_pre_max_stddev = gbit_s_pre_max * (ms_pre_max_stddev / ms_pre_max) if ms_pre_max else Decimal(0)

            aggregated_statistics[protocol].append({
                'Layer_Type': layer_type,
                'Layer_ID': layer_id,
                'MB_SENT': mb_sent_avg,
                'MB_SENT_PRE': mb_sent_pre_avg,
                'MB_RECEIVED': mb_received_avg,
                'MB_RECEIVED_PRE': mb_received_pre_avg,
                'ms_LIVE Avg (mean)': ms_live_avg,
                'ms_LIVE Max (mean)': ms_live_max,
                'ms_LIVE Avg (stddev)': ms_live_stddev,
                'ms_LIVE Max (stddev)': ms_live_max_stddev,
                'ms_PRE Avg (mean)': ms_pre_avg,
                'ms_PRE Max (mean)': ms_pre_max,
                'ms_PRE Avg (stddev)': ms_pre_stddev,
                'ms_PRE Max (stddev)': ms_pre_max_stddev,
                'Gbit/s LIVE Avg (mean)': gbit_s_live_avg,
                'Gbit/s LIVE Avg (stddev)': gbit_s_live_stddev,
                'Gbit/s LIVE Max (mean)': gbit_s_live_max,
                'Gbit/s LIVE Max (stddev)': gbit_s_live_max_stddev,
                'Gbit/s PRE Avg (mean)': gbit_s_pre_avg,
                'Gbit/s PRE Avg (stddev)': gbit_s_pre_stddev,
                'Gbit/s PRE Max (mean)': gbit_s_pre_max,
                'Gbit/s PRE Max (stddev)': gbit_s_pre_max_stddev,
            })

    return aggregated_statistics

def calculate_statistics(protocol_data):
    statistics_by_protocol = defaultdict(list)

    for protocol, data_list in protocol_data.items():
        for layer_data, total_data in data_list:
            statistics = calculate_statistics_within_configuration(layer_data, total_data)
            statistics_by_protocol[protocol].append(statistics)

    final_statistics = aggregate_identical_configurations(statistics_by_protocol)
    
    return final_statistics

def save_to_csv(final_data, input_filename):
    for protocol, data in final_data.items():
        if protocol:
            protocol_cleaned = protocol.replace(", ", "_").replace(" ", "_").replace(":", "")
            csv_filename = f"{input_filename}_{protocol_cleaned}.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False)
            print(f"Saved {len(data)} lines to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NN statistics log file.")
    parser.add_argument("input_file", type=str, help="Path to the input txt file.")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        file_content = f.read()

    input_filename = args.input_file.rsplit(".", 1)[0]

    records = parse_data(file_content)
    protocol_data = process_data(records)
    final_data = calculate_statistics(protocol_data)
    save_to_csv(final_data, input_filename)

