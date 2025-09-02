#!/bin/bash

# ==============================================================================
# SCRIPT FUNCTIONALITY:
# 1. Reads a log file containing multiple blocks of statistical data.
# 2. Aggregates the cycle counts for each statistical item (e.g., "Global Total")
#    across all blocks in the file.
# 3. After processing the entire file, it performs two main calculations:
#    a. [Total Accumulated Time]: Converts the total aggregated cycle counts
#       directly into time in seconds.
#    b. [Average Time per Block]: Divides the total cycle counts by a specified
#       number of blocks to get an average, then converts that average into
#       time in seconds.
# 4. Generates a formatted final report containing the results of both calculations.
# ==============================================================================

# --- Configuration Section ---
# Total number of data blocks to use for the averaging calculation.
NUM_BLOCKS=100
# CPU frequency in GHz.
# PLEASE REPLACE "xxx" WITH THE ACTUAL FREQUENCY.
FREQUENCY_GHZ="xxx"

# --- Main Script Logic ---

# Check if the user has provided the required input and output filenames.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file_with_all_data> <output_file_for_report>"
    echo "Example: ./compute_total_and_avg_time.sh all_stats.log final_report.txt"
    exit 1
fi

# Assign command-line arguments to more readable variable names.
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if the input file exists.
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Aggregating data from '$INPUT_FILE'..."
echo "Calculating totals and averages based on $NUM_BLOCKS data blocks..."

# Use awk to perform all aggregation and calculation tasks.
# The awk associative array `sums` is used to store the accumulated cycle count for each metric by name.
awk -v num_blocks="$NUM_BLOCKS" -v freq_ghz="$FREQUENCY_GHZ" '
# For each line, check for a metric keyword. If a keyword is found,
# add the last field of the line ($NF, which is the numeric value)
# to the corresponding item in the `sums` array.
/Global Total:/      { sums["total"]      += $NF }
/Global precompute:/ { sums["precompute"] += $NF }
/Global cale_time:/  { sums["calc_time"]  += $NF } # Assuming "cale_time" is a typo for "calc_time"
/Global sort time:/  { sums["sort"]       += $NF }
/Global reduce time:/ { sums["reduce"]     += $NF }

# The END block is executed once after the entire file has been read.
END {
    # --- Preparation: Define calculation constants ---
    frequency_hz = freq_ghz * 1e9;

    # --- Part 1: Calculate [Total Accumulated Time] ---
    # Formula: Total Time (s) = Total Cycles / Frequency (Hz)
    total_time_total      = sums["total"] / frequency_hz;
    total_time_precompute = sums["precompute"] / frequency_hz;
    total_time_calc       = sums["calc_time"] / frequency_hz;
    total_time_sort       = sums["sort"] / frequency_hz;
    total_time_reduce     = sums["reduce"] / frequency_hz;

    # --- Part 2: Calculate [Average Time per Block] ---
    # Formula: Average Time (s) = (Total Cycles / Number of Blocks) / Frequency (Hz)
    avg_time_total      = (sums["total"] / num_blocks) / frequency_hz;
    avg_time_precompute = (sums["precompute"] / num_blocks) / frequency_hz;
    avg_time_calc       = (sums["calc_time"] / num_blocks) / frequency_hz;
    avg_time_sort       = (sums["sort"] / num_blocks) / frequency_hz;
    avg_time_reduce     = (sums["reduce"] / num_blocks) / frequency_hz;

    # Note: The original script multiplies the final times by 2. This is preserved here.
    # This might be specific to the use case (e.g., accounting for 2 nodes).
    multiplier = 2;

    # --- Print the [Total Accumulated Time] Report ---
    print "### Total Accumulated Time (Unit: seconds) ###";
    print "=================================================";
    printf "Global Total: %.9f\n", total_time_total * multiplier;
    printf "|  Global precompute: %.9f\n", total_time_precompute * multiplier;
    printf "|  Global calc_time:  %.9f\n", total_time_calc * multiplier;
    printf "|  Global sort time:  %.9f\n", total_time_sort * multiplier;
    printf "|  Global reduce time:%.9f\n", total_time_reduce * multiplier;
    print "-------------------------------------------------";

    # Print a blank line for separation.
    print "";

    # --- Print the [Average Time per Block] Report ---
    print "### Average Time per Block (Unit: seconds) ###";
    print "=================================================";
    printf "Global Total: %.9f\n", avg_time_total * multiplier;
    printf "|  Global precompute: %.9f\n", avg_time_precompute * multiplier;
    printf "|  Global calc_time:  %.9f\n", avg_time_calc * multiplier;
    printf "|  Global sort time:  %.9f\n", avg_time_sort * multiplier;
    printf "|  Global reduce time:%.9f\n", avg_time_reduce * multiplier;
    print "-------------------------------------------------";

    # --- Print Footer Information ---
    printf "\n(Calculations based on %d data blocks and a %.2fGHz frequency)\n", num_blocks * multiplier, freq_ghz;

}' "$INPUT_FILE" > "$OUTPUT_FILE"

# Check if the output file was generated successfully.
if [ -f "$OUTPUT_FILE" ]; then
    echo "----------------------------------------"
    echo "Calculation complete!"
    echo "Final report has been written to: $OUTPUT_FILE"
    echo ""
    echo "--- Final Report Preview ---"
    cat "$OUTPUT_FILE"
else
    echo "Error: Failed to generate the report file."
    exit 1
fi