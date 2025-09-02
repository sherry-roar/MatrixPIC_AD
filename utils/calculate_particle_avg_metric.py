#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script provides a utility to parse a file and calculate the average value 
of a specific metric. It is designed to read text files where the data of 
interest is on lines beginning with a specific prefix, such as 'avg_metric:'.
"""

import re

def calculate_avg_metric(file_path):
    """
    Calculates the average of numbers found on lines starting with 'avg_metric:'.

    This function opens a file, iterates through each line, and uses a regular
    expression to find lines that match the specified format. It collects all
    the numeric values from these lines and computes their average.

    :param file_path: str, The path to the input file.
    :return: float, The calculated average, or None if no matching lines are found
             or the file does not exist.
    """
    values = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Use regex to match lines starting with 'avg_metric:' followed by a number.
                # - r'avg_metric:' : Matches the literal text.
                # - \s* : Matches zero or more whitespace characters.
                # - (\d+)          : Captures one or more digits as a group.
                match = re.match(r'avg_metric:\s*(\d+)', line.strip())
                if match:
                    # Extract the captured number (group 1) and convert it to an integer.
                    values.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    
    # Check if any values were found to avoid a division-by-zero error.
    if not values:
        print("Warning: No lines matching the 'avg_metric:' format were found.")
        return None
    
    # Calculate the average using built-in sum() and len() functions for efficiency.
    average = sum(values) / len(values)
    print(f"Number of values found: {len(values)}")
    return average

# --- Example Usage ---
# Replace this with the actual path to your data file.
file_path = "sme.LWFA.444.timer.static.out" 
print(f"Processing file: {file_path}")

result = calculate_avg_metric(file_path)

if result is not None:
    # Print the final result formatted to two decimal places.
    print(f"The calculated average metric is: {result:.2f}")