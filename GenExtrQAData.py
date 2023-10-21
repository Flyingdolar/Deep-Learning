# Generate the data for the task of Extractive Question Answering Model Training
# Input: 2 files, 'Origin' and 'Context'
# Output: 1 file, 'Output'
#
# User Command as follows: python GenExtrQAData.py --origin {Origin File} --context {Context File} --output {Output File}
# For example: python GenExtrQAData.py --origin data/train.json --context data/context.json --output data/trainQA.json

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
            "question": origin[idx]["question"],
            "context": origin[idx]["relevant"],
            "answer": {
                "answer_start": [origin[idx]["answer"]["start"]],
                "text": [origin[idx]["answer"]["text"]],
            },
        }
    )

# Fill the missing values in the column 'context'
for idx in range(len(output)):
    output[idx]["context"] = context[output[idx]["context"]]

# Write the result to the file 'Output'
with open(args.output, "w", encoding="UTF-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
