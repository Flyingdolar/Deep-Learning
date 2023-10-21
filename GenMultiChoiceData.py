# Generate the data for the task of Multiple Choice Model Training
# Input: 2 files, 'Origin' and 'Context'
# Output: 1 file, 'Output'
#
# User Command as follows: python GenMultiChoiceData.py --origin {Origin File} --context {Context File} --output {Output File}
# For example: python GenMultiChoiceData.py --origin data/train.json --context data/context.json --output data/trainMC.json

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--origin", help="The file to be appended")
parser.add_argument("--context", help="The file to fill the missing values")
parser.add_argument("--output", help="The output file")

args = parser.parse_args()

# Open All Files that needed
with open(args.origin, encoding="UTF-8") as f:
    origin = json.load(f)

with open(args.context, encoding="UTF-8") as f:
    context = json.load(f)

output = []
# Generate the data for the task of Extractive Question Answering Model Training
for idx in range(len(origin)):
    output.append(
        {
            "id": origin[idx]["id"],
            "sent1": origin[idx]["question"],
            "sent2": origin[idx]["paragraphs"],
            "label": origin[idx]["relevant"],
        }
    )

# Relabel `label` from context id to sent2 id
for idx in range(len(output)):
    for sdx in range(len(output[idx]["sent2"])):
        if output[idx]["sent2"][sdx] == output[idx]["label"]:
            output[idx]["label"] = sdx
            break

# Fill the missing values in the column 'sent2'
for idx in range(len(output)):
    for sdx in range(len(output[idx]["sent2"])):
        output[idx]["sent2"][sdx] = context[output[idx]["sent2"][sdx]]


# Write the result to the file 'Output'
with open(args.output, "w", encoding="UTF-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
