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

# Write the result to the file 'Output'
with open(args.output, "w", encoding="UTF-8") as f:
    json.dump(output, f, sort_keys=True, indent=4, ensure_ascii=False)
