# Read the {A File} and fill the missing values with the {B File} string
# and write the result to the file train_mean.json

# User Command as follows: python normal.py --append {A File} --fill {B File} --output {Output File}

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--append", help="The file to be appended")
parser.add_argument("--fill", help="The file to fill the missing values")
parser.add_argument("--output", help="The output file")

args = parser.parse_args()

# Read the file 'Append'
with open(args.append, encoding="UTF-8") as f:
    append = json.load(f)

# Read the file 'Fill'
with open(args.fill, encoding="UTF-8") as f:
    fill = json.load(f)

# Copy 'Append' to 'Output'
output = append

# Change all Answers to relevant index
for idx in range(len(append)):
    for pdx in range(4):
        if append[idx]["paragraphs"][pdx] == append[idx]["relevant"]:
            output[idx]["relevant"] = pdx

# Fill the missing values
for idx in range(len(append)):
    for pdx in range(4):
        num = output[idx]["paragraphs"][pdx]
        output[idx]["paragraphs"][pdx] = fill[num]

# Change relevant index to relevant answer
for idx in range(len(append)):
        num =  output[idx]["relevant"]
        output[idx]["relevant"] =output[idx]["paragraphs"][num]

# Change Answer to list
for idx in range(len(append)):
    output[idx]["answer"]["text"] = [output[idx]["answer"]["text"]]
    output[idx]["answer"]["start"] = [output[idx]["answer"]["start"]]

# Change the column name answer to answer_start
for idx in range(len(append)):
    output[idx]["answer"]["answer_start"] = output[idx]["answer"]["start"]
    del output[idx]["answer"]["start"]

# Write the result to the file 'Output'
with open(args.output, "w", encoding="UTF-8") as f:
    json.dump(output, f, sort_keys=True, indent=4, ensure_ascii=False)
