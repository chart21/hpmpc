#!/bin/bash

# Initialize associative arrays for sums and counts
declare -A sum_sent sum_received sum_live sum_pre sum_sent_pre sum_received_pre layertype total_comm total_runtime mbits comm_frac runtime_frac
declare -A count_sent count_received count_live count_pre count_sent_pre count_received_pre

declare -A aggr_sum_sent aggr_sum_received aggr_sum_live aggr_sum_pre aggr_sum_sent_pre aggr_sum_received_pre aggr_comm_frac aggr_runtime_frac aggr_mbits aggr_total_comm aggr_total_runtime
aggr_comm=0
aggr_runtime=0


# num_players=3
# dattype=256


init_arrays() {
    aggr_comm=0
    aggr_runtime=0
    for key in "${!sum_sent[@]}"; do
        unset sum_sent["$key"]
    done
    for key in "${!sum_received[@]}"; do
        unset sum_received["$key"]
    done
    for key in "${!sum_live[@]}"; do
        unset sum_live["$key"]
    done
    for key in "${!sum_pre[@]}"; do
        unset sum_pre["$key"]
    done
    for key in "${!sum_sent_pre[@]}"; do
        unset sum_sent_pre["$key"]
    done
    for key in "${!sum_received_pre[@]}"; do
        unset sum_received_pre["$key"]
    done
    for key in "${!layertype[@]}"; do
        unset layertype["$key"]
    done
    for key in "${!total_comm[@]}"; do
        unset total_comm["$key"]
    done
    for key in "${!total_runtime[@]}"; do
        unset total_runtime["$key"]
    done
    for key in "${!mbits[@]}"; do
        unset mbits["$key"]
    done
    for key in "${!comm_frac[@]}"; do
        unset comm_frac["$key"]
    done
    for key in "${!runtime_frac[@]}"; do
        unset runtime_frac["$key"]
    done
    for key in "${!count_sent[@]}"; do
        unset count_sent["$key"]
    done
    for key in "${!count_received[@]}"; do
        unset count_received["$key"]
    done
    for key in "${!count_live[@]}"; do
        unset count_live["$key"]
    done
    for key in "${!count_pre[@]}"; do
        unset count_pre["$key"]
    done
    for key in "${!count_sent_pre[@]}"; do
        unset count_sent_pre["$key"]
    done
    for key in "${!count_received_pre[@]}"; do
        unset count_received_pre["$key"]
    done
    for key in "${!aggr_sum_sent[@]}"; do
        unset aggr_sum_sent["$key"]
    done
    for key in "${!aggr_sum_received[@]}"; do
        unset aggr_sum_received["$key"]
    done
    for key in "${!aggr_sum_live[@]}"; do
        unset aggr_sum_live["$key"]
    done
    for key in "${!aggr_sum_pre[@]}"; do
        unset aggr_sum_pre["$key"]
    done
    for key in "${!aggr_sum_sent_pre[@]}"; do
        unset aggr_sum_sent_pre["$key"]
    done
    for key in "${!aggr_sum_received_pre[@]}"; do
        unset aggr_sum_received_pre["$key"]
    done
    for key in "${!aggr_comm_frac[@]}"; do
        unset aggr_comm_frac["$key"]
    done
    for key in "${!aggr_runtime_frac[@]}"; do
        unset aggr_runtime_frac["$key"]
    done
    for key in "${!aggr_mbits[@]}"; do
        unset aggr_mbits["$key"]
    done
    for key in "${!aggr_total_comm[@]}"; do
        unset aggr_total_comm["$key"]
    done
    for key in "${!aggr_total_runtime[@]}"; do
        unset aggr_total_runtime["$key"]
    done
        
}
    
# Function to process each line and update the associative arrays
process_line() {
    local line=$1

    # Regular expression to extract the relevant data
    if [[ $line =~ ID:\ ([0-9]+)\ ([A-Z0-9]+)\ +MB\ SENT:([0-9.]+)\ +MB\ RECEIVED:([0-9.]+)\ +MB\ SENT\ PRE:([0-9.]+)\ +MB\ RECEIVED\ PRE:\ ([0-9.]+)\ +ms\ LIVE:\ ([0-9.]+)\ +ms\ PRE:\ ([0-9.]+) ]]; then
        echo "Processing line: $line"
        local layer_id=${BASH_REMATCH[1]}
        echo "Layer ID: $layer_id"
        local layer_type=${BASH_REMATCH[2]}
        echo "Layer type: $layer_type"
        local mb_sent=${BASH_REMATCH[3]}
        echo "MB sent: $mb_sent"
        local mb_received=${BASH_REMATCH[4]}
        echo "MB received: $mb_received"
        local mb_sent_pre=${BASH_REMATCH[5]}
        echo "MB sent pre: $mb_sent_pre"
        local mb_received_pre=${BASH_REMATCH[6]}
        echo "MB received pre: $mb_received_pre"
        local ms_live=${BASH_REMATCH[7]}
        echo "MS live: $ms_live"
        local ms_pre=${BASH_REMATCH[8]}
        echo "MS pre: $ms_pre"
       
        # Update layer type
        layertype["$layer_id"]=$layer_type

        # Update sums
        if [ -z "${sum_sent["$layer_id"]}" ]; then
            sum_sent["$layer_id"]=0
        fi
        if [ -z "${sum_received["$layer_id"]}" ]; then
            sum_received["$layer_id"]=layer_type
        fi
        if [ -z "${sum_sent_pre["$layer_id"]}" ]; then
            sum_sent_pre["$layer_id"]=0
        fi
        if [ -z "${sum_received_pre["$layer_id"]}" ]; then
            sum_received_pre["$layer_id"]=layer_type
        fi
        if [ -z "${sum_live["$layer_id"]}" ]; then
            sum_live["$layer_id"]=0
        fi
        if [ -z "${sum_pre["$layer_id"]}" ]; then
            sum_pre["$layer_id"]=layer_type
        fi
        if [ -z "${total_comm["$layer_id"]}" ]; then
            total_comm["$layer_id"]=0
        fi
        if [ -z "${total_runtime["$layer_id"]}" ]; then
            total_runtime["$layer_id"]=0
        fi
        sum_sent["$layer_id"]=$(echo "scale=4; ${sum_sent["$layer_id"]} + $mb_sent" | bc)
        sum_received["$layer_id"]=$(echo "scale=4; ${sum_received["$layer_id"]} + $mb_received" | bc)
        sum_sent_pre["$layer_id"]=$(echo "scale=4; ${sum_sent_pre["$layer_id"]} + $mb_sent_pre" | bc)
        sum_received_pre["$layer_id"]=$(echo "scale=4; ${sum_received_pre["$layer_id"]} + $mb_received_pre" | bc)
        # total_comm["$layer_id"]=$(echo "scale=4; ${total_comm["$layer_id"]} + ($mb_sent + $mb_received + $mb_sent_pre + $mb_received_pre)" | bc)
        total_comm["$layer_id"]=$(echo "scale=4; ${total_comm["$layer_id"]} + ($mb_sent + $mb_sent_pre)" | bc) # don't count same data twice with recv and send
        sum_pre["$layer_id"]=$(echo "scale=4; ${sum_pre["$layer_id"]} + $ms_pre " | bc)
        sum_live["$layer_id"]=$(echo "scale=4; ${sum_live["$layer_id"]} + $ms_live " | bc)
        total_runtime["$layer_id"]=$(echo "scale=4; ${total_runtime["$layer_id"]} + ($ms_live + $ms_pre) " | bc)
        






        

        # Update counts
        if [ -z "${count_sent["$layer_id"]}" ]; then
            count_sent["$layer_id"]=0
        fi
        if [ -z "${count_received["$layer_id"]}" ]; then
            count_received["$layer_id"]=0
        fi
        if [ -z "${count_sent_pre["$layer_id"]}" ]; then
            count_sent_pre["$layer_id"]=0
        fi
        if [ -z "${count_received_pre["$layer_id"]}" ]; then
            count_received_pre["$layer_id"]=0
        fi
        if [ -z "${count_live["$layer_id"]}" ]; then
            count_live["$layer_id"]=0
        fi
        if [ -z "${count_pre["$layer_id"]}" ]; then
            count_pre["$layer_id"]=0
        fi
        count_sent["$layer_id"]=$((${count_sent["$layer_id"]} + 1))
        count_received["$layer_id"]=$((${count_received["$layer_id"]} + 1))
        count_sent_pre["$layer_id"]=$((${count_sent_pre["$layer_id"]} + 1))
        count_received_pre["$layer_id"]=$((${count_received_pre["$layer_id"]} + 1))
        count_live["$layer_id"]=$((${count_live["$layer_id"]} + 1))
        count_pre["$layer_id"]=$((${count_pre["$layer_id"]} + 1))
    elif [[ $line =~ ID:\ ([0-9]+)\ ([A-Z0-9]+)\ +MB\ SENT:([0-9.]+)\ +MB\ RECEIVED:([0-9.]+)\ +ms\ LIVE:\ ([0-9.]+) ]]; then
        echo "Processing line: $line"
        local layer_id=${BASH_REMATCH[1]}
        echo "Layer ID: $layer_id"
        local layer_type=${BASH_REMATCH[2]}
        echo "Layer type: $layer_type"
        local mb_sent=${BASH_REMATCH[3]}
        echo "MB sent: $mb_sent"
        local mb_received=${BASH_REMATCH[4]}
        echo "MB received: $mb_received"
        local ms_live=${BASH_REMATCH[5]}
        echo "MS live: $ms_live"

        # Update layer type
        layertype["$layer_id"]=$layer_type

        # Update sums
        if [ -z "${sum_sent["$layer_id"]}" ]; then
            sum_sent["$layer_id"]=0
        fi
        if [ -z "${sum_received["$layer_id"]}" ]; then
            sum_received["$layer_id"]=layer_type
        fi
        if [ -z "${sum_live["$layer_id"]}" ]; then
            sum_live["$layer_id"]=0
        fi
        if [ -z "${total_comm["$layer_id"]}" ]; then
            total_comm["$layer_id"]=0
        fi
        if [ -z "${total_runtime["$layer_id"]}" ]; then
            total_runtime["$layer_id"]=0
        fi

        sum_sent["$layer_id"]=$(echo "scale=4; ${sum_sent["$layer_id"]} + $mb_sent" | bc)
        sum_received["$layer_id"]=$(echo "scale=4; ${sum_received["$layer_id"]} + $mb_received" | bc)
        # total_comm["$layer_id"]=$(echo "scale=4; ${total_comm["$layer_id"]} + ($mb_sent + $mb_received)" | bc)
        total_comm["$layer_id"]=$(echo "scale=4; ${total_comm["$layer_id"]} + $mb_sent" | bc) # don't count same data twice with recv and send
        sum_live["$layer_id"]=$(echo "scale=4; ${sum_live["$layer_id"]} + $ms_live" | bc)
        total_runtime["$layer_id"]=$(echo "scale=4; ${total_runtime["$layer_id"]} + $ms_live " | bc)

       
        

        # Update counts
        if [ -z "${count_sent["$layer_id"]}" ]; then
            count_sent["$layer_id"]=0
        fi
        if [ -z "${count_received["$layer_id"]}" ]; then
            count_received["$layer_id"]=0
        fi
        if [ -z "${count_live["$layer_id"]}" ]; then
            count_live["$layer_id"]=0
        fi
        count_sent["$layer_id"]=$((${count_sent["$layer_id"]} + 1))
        count_received["$layer_id"]=$((${count_received["$layer_id"]} + 1))
        count_live["$layer_id"]=$((${count_live["$layer_id"]} + 1))
    fi


}

nn_parse()
{

# Prepare the CSV file and delete if it already exists
# output_file="aggregated_results.csv"
output_file=$2
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

init_arrays

# Read the file line by line
while IFS= read -r line; do
    process_line "$line"
# done < "testresults.txt"
done < $1


for key in "${!sum_sent[@]}"; do
    aggr_comm=$(echo "scale=4; ${total_comm["$key"]} + $aggr_comm" | bc)
    total_runtime["$key"]=$(echo "scale=4; ${total_runtime["$key"]} / ${count_live["$key"]}" | bc) #all layers were evaluated in parallel
    aggr_runtime=$(echo "scale=4; ${total_runtime["$key"]} + $aggr_runtime" | bc)
done


# Calculate averages and write to the CSV file
for key in "${!sum_sent[@]}"; do
    llayer_type=${layertype["$key"]}
    # avg_sent=$(echo "scale=4; ${sum_sent["$key"]} / ${count_sent["$key"]}" | bc)
    avg_sent=$(echo "scale=4; ${sum_sent["$key"]}" | bc)
    # avg_received=$(echo "scale=4; ${sum_received["$key"]} / ${count_received["$key"]}" | bc)
    avg_received=$(echo "scale=4; ${sum_received["$key"]}" | bc)
    if [ -z "${sum_sent_pre["$key"]}" ]; then
    avg_sent_pre="N/A"
    else
    # avg_sent_pre=$(echo "scale=4; ${sum_sent_pre["$key"]} / ${count_sent_pre["$key"]}" | bc)
    avg_sent_pre=$(echo "scale=4; ${sum_sent_pre["$key"]}" | bc)
    fi
    if [ -z "${sum_received_pre["$key"]}" ]; then
    avg_received_pre="N/A"
    else
    # avg_received_pre=$(echo "scale=4; ${sum_received_pre["$key"]} / ${count_received_pre["$key"]}" | bc)
    avg_received_pre=$(echo "scale=4; ${sum_received_pre["$key"]}" | bc)
    fi
    avg_live=$(echo "scale=4; ${sum_live["$key"]} / ${count_live["$key"]}" | bc)
    if [ -z "${sum_pre["$key"]}" ]; then
    avg_pre="N/A"
    else
    avg_pre=$(echo "scale=4; ${sum_pre["$key"]} / ${count_pre["$key"]}" | bc)
    fi
    layer_id=$(echo "$key" | cut -d',' -f1)
    mbits["$key"]=$(echo "scale=4; 8000 * ${total_comm["$key"]} / ${total_runtime["$key"]}" | bc)
    echo "total_comm: ${total_comm["$key"]}"
    echo "total_runtime: ${total_runtime["$key"]}"
    echo "mbits: ${mbits["$key"]}"
    runtime_frac["$key"]=$(echo "scale=4; ${total_runtime["$key"]} / $aggr_runtime" | bc)
    comm_frac["$key"]=$(echo "scale=4; ${total_comm["$key"]} / $aggr_comm" | bc)
    echo "aggr_comm: $aggr_comm"
    echo "aggr_runtime: $aggr_runtime"
    echo "comm_frac: ${comm_frac["$key"]}"
    echo "runtime_frac: ${runtime_frac["$key"]}"
    echo "$layer_id,$llayer_type,$avg_sent_pre,$avg_sent,$avg_received_pre,$avg_received,$avg_pre,$avg_live,${mbits["$key"]},${comm_frac["$key"]},${runtime_frac["$key"]}" >> "$output_file"
done





#sort output_file rows by LayerID, make sure 2 is before 10
sort -t, -k1,1n -k2,2 $output_file -o $output_file

# add header to output_file
#"LayerID,LayerType,MB Sent Pre,MB Sent,MB Received Pre,MB Received,MS Pre,MS Live"
sed -i '1iLayerID,LayerType,MB Sent Pre,MB Sent,MB Received Pre,MB Received,MS Pre,MS Live, Mbit/s, comm_frac,runtime_frac' $output_file

#aggregate results based on layer type
for key in "${!sum_sent[@]}"; do
    llayer_type=${layertype["$key"]}
    if [ -z "${aggr_sum_sent["$llayer_type"]}" ]; then
            aggr_sum_sent["$llayer_type"]=0
        fi
    if [ -z "${aggr_sum_received["$llayer_type"]}" ]; then
            aggr_sum_received["$llayer_type"]=0
        fi
    if [ -z "${aggr_sum_live["$llayer_type"]}" ]; then
            aggr_sum_live["$llayer_type"]=0
        fi
    if [ -z "${aggr_sum_pre["$llayer_type"]}" ]; then
            aggr_sum_pre["$llayer_type"]=0
        fi
    if [ -z "${aggr_sum_sent_pre["$llayer_type"]}" ]; then
            aggr_sum_sent_pre["$llayer_type"]=0
        fi
    if [ -z "${aggr_sum_received_pre["$llayer_type"]}" ]; then
            aggr_sum_received_pre["$llayer_type"]=0
        fi
    if [ -z "${aggr_comm_frac["$llayer_type"]}" ]; then
            aggr_comm_frac["$llayer_type"]=0
        fi
    if [ -z "${aggr_runtime_frac["$llayer_type"]}" ]; then
            aggr_runtime_frac["$llayer_type"]=0
        fi
    if [ -z "${aggr_total_comm["$llayer_type"]}" ]; then
            aggr_total_comm["$llayer_type"]=0
        fi
    if [ -z "${aggr_total_runtime["$llayer_type"]}" ]; then
            aggr_total_runtime["$llayer_type"]=0
        fi

    # percentage_comm["key"]=$(echo "scale=4; ${precentage_comm["$key"]} + $mb_sent" + $mb_received + $mb_sent_pre + $mb_received_pre | bc)
    # percentage_runtime["key"]=$(echo "scale=4; ${precentage_runtime["$key"]} + $ms_live + $ms_pre" | bc)
    # mbits["key"]=$(echo "scale=4; ${mbits["$key"]} + ($mb_sent" + $mb_received + $mb_sent_pre + $mb_received_pre) / ($ms_live + $ms_pre)" | bc)
    aggr_sum_sent["$llayer_type"]=$(echo "scale=4; ${aggr_sum_sent["$llayer_type"]} + ${sum_sent["$key"]}" | bc)
    aggr_sum_received["$llayer_type"]=$(echo "scale=4; ${aggr_sum_received["$llayer_type"]} + ${sum_received["$key"]}" | bc)
    aggr_sum_live["$llayer_type"]=$(echo "scale=4; ${aggr_sum_live["$llayer_type"]} + ${sum_live["$key"]} / ${count_live["$key"]}" | bc)

    if [ -z "${sum_pre["$key"]}" ]; then
    aggr_sum_pre["$llayer_type"]="N/A"
    else
    aggr_sum_pre["$llayer_type"]=$(echo "scale=4; ${aggr_sum_pre["$llayer_type"]} + ${sum_pre["$key"]} / ${count_pre["$key"]}" | bc)
    fi
    if [ -z "${sum_sent_pre["$key"]}" ]; then
    aggr_sum_sent_pre["$llayer_type"]="N/A"
    else
    aggr_sum_sent_pre["$llayer_type"]=$(echo "scale=4; ${aggr_sum_sent_pre["$llayer_type"]} + ${sum_sent_pre["$key"]}" | bc)
    fi
    if [ -z "${sum_received_pre["$key"]}" ]; then
    aggr_sum_received_pre["$llayer_type"]="N/A"
    else
    aggr_sum_received_pre["$llayer_type"]=$(echo "scale=4; ${aggr_sum_received_pre["$llayer_type"]} + ${sum_received_pre["$key"]}" | bc)
    fi
    if [ -z "${comm_frac["$key"]}" ]; then
        aggr_comm_frac["$llayer_type"]="N/A"
    else
        # aggr_comm_frac["$llayer_type"]=$(echo "scale=4; ${aggr_comm_frac["$llayer_type"]} + ${comm_frac["$key"]}" | bc)
        aggr_comm_frac["$llayer_type"]=$(echo "scale=4; ${aggr_comm_frac["$llayer_type"]} + ${total_comm["$key"]}" | bc)
    fi
    if [ -z "${runtime_frac["$key"]}" ]; then
        aggr_runtime_frac["$llayer_type"]="N/A"
    else
        # aggr_runtime_frac["$llayer_type"]=$(echo "scale=4; ${aggr_runtime_frac["$llayer_type"]} + ${runtime_frac["$key"]}" | bc)
        aggr_runtime_frac["$llayer_type"]=$(echo "scale=4; ${aggr_runtime_frac["$llayer_type"]} + ${total_runtime["$key"]}" | bc)
    fi

    aggr_total_comm["$llayer_type"]=$(echo "scale=4; ${aggr_total_comm["$llayer_type"]} + ${total_comm["$key"]}" | bc)
    aggr_total_runtime["$llayer_type"]=$(echo "scale=4; ${aggr_total_runtime["$llayer_type"]} + ${total_runtime["$key"]}" | bc)

    


done

total_sum_sent=0
total_sum_received=0
total_sum_live=0
total_sum_pre=0
total_sum_sent_pre=0
total_sum_received_pre=0
total_mbits=0
total_total_comm=0
total_total_runtime=0


for key in "${!aggr_sum_sent[@]}"; do
    mbits["$key"]=$(echo "scale=4; 8000 * ${aggr_total_comm["$key"]} / ${aggr_total_runtime["$key"]}" | bc)
    aggr_runtime_frac["$key"]=$(echo "scale=4; ${aggr_runtime_frac["$key"]} / $aggr_runtime" | bc)
    aggr_comm_frac["$key"]=$(echo "scale=4; ${aggr_comm_frac["$key"]} / $aggr_comm" | bc)
    # aggrruntime_frac["$key"]=$(echo "scale=4; ${aggr_total_runtime["$key"]} / $aggr_runtime" | bc)
    # aggrcomm_frac["$key"]=$(echo "scale=4; ${aggr_total_comm["$key"]} / $aggr_comm" | bc)
    # mbits["$key"]=8*$mbits["$key"]
    echo "key: $key"
    echo "aggr_total_comm: ${aggr_total_comm["$key"]}"
    echo "aggr_total_runtime: ${aggr_total_runtime["$key"]}"
    echo "aggr_mbits: ${mbits["$key"]}"
    echo "AGR",$key,${aggr_sum_sent_pre["$key"]},${aggr_sum_sent["$key"]},${aggr_sum_received_pre["$key"]},${aggr_sum_received["$key"]},${aggr_sum_pre["$key"]},${aggr_sum_live["$key"]},${mbits["$key"]},${aggr_comm_frac["$key"]},${aggr_runtime_frac["$key"]} >> "$output_file"

    total_sum_sent=$(echo "scale=4; ${aggr_sum_sent["$key"]} + $total_sum_sent" | bc)
    total_sum_received=$(echo "scale=4; ${aggr_sum_received["$key"]} + $total_sum_received" | bc)
    total_sum_live=$(echo "scale=4; ${aggr_sum_live["$key"]} + $total_sum_live" | bc)
    total_total_comm=$(echo "scale=4; ${aggr_total_comm["$key"]} + $total_total_comm" | bc)
    total_total_runtime=$(echo "scale=4; ${aggr_total_runtime["$key"]} + $total_total_runtime" | bc)
    if [ "${aggr_sum_pre["$key"]}" == "N/A" ]; then
        total_sum_pre="N/A"
    else
    total_sum_pre=$(echo "scale=4; ${aggr_sum_pre["$key"]} + $total_sum_pre" | bc)
    echo "total_sum_pre: $total_sum_pre"
    echo "aggr_sum_pre: ${aggr_sum_pre["$key"]}"
    fi
    if [ "${aggr_sum_sent_pre["$key"]}" == "N/A" ]; then
        total_sum_sent_pre="N/A"
    else
    total_sum_sent_pre=$(echo "scale=4; ${aggr_sum_sent_pre["$key"]} + $total_sum_sent_pre" | bc)
    fi
    if [ "${aggr_sum_received_pre["$key"]}" == "N/A" ]; then
        total_sum_received_pre="N/A"
    else
    total_sum_received_pre=$(echo "scale=4; ${aggr_sum_received_pre["$key"]} + $total_sum_received_pre" | bc)
    fi
    
done
    total_mbits=$(echo "scale=4; 8000 * $aggr_comm / $aggr_runtime" | bc)
    echo "TOT","Total",$total_sum_sent_pre,$total_sum_sent,$total_sum_received_pre,$total_sum_received,$total_sum_pre,$total_sum_live,$total_mbits,"1","1" >> "$output_file"
    

echo "Aggregation complete. Results stored in $output_file"
}


#Usage
#check if bash source is 0
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
if [ $# -ne 2 ]; then
    echo "Save the output an NN inference to a file and run this script on it."
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
else
    nn_parse $1 $2
fi
fi


