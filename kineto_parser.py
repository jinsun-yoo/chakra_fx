import re
import csv
import argparse

# Define the regex pattern
pattern = r'"kernel", "name": "(.+)", .+\n.+"dur": ([\d\.]+)'

# Read the content of the file
with open('/my_workspace/data/nanogpt_TP2FSDP4_Kineto/postexec_trace/trace_kineto_rank0.json', 'r') as file:
    content = file.read()

# Find all matches using the regex pattern
matches = re.findall(pattern, content)

# Write the matches to a CSV file
with open('asdf.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Kernel Name', 'Duration'])  # Write the header row
    writer.writerows(matches)  # Write the matched data

print(f"Data has been written to 'asdf.csv'.")

def parse_args():
    # Create parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        "--job",
        type=str,
        default="nanogpt",
        required=False,
        help="""Either 'simple' or 'nanogpt' or 'resnet18'. Chooses which model to work on.""",
    )
    args = parser.parse_args()
    return args

def parse_step_time():
    import os
import re
import sys

def iterate_files(directory, pattern):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        sys.exit(1)

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if regex.match(filename):
            print(f"Found: {filename}")

# Example usage:
directory = "/path/to/directory"
pattern = r"^.*\.txt$"  # Regular expression to match .txt files

iterate_files(directory, pattern)

if __name__ == "__main__":
    args = parse_args()

    job = args.job
    if job == "compute":
        parse_compute_time()
    if job == "steptime":
        parse_step_time()