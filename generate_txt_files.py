import csv
from collections import defaultdict
import os

# Read the CSV and group words by document
documents = defaultdict(list)

with open('UntN-Nasu texts Word-level.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_id = row['id_text']
        word = row['Text']
        documents[doc_id].append(word)

# Create a .txt file for each document
output_dir = 'texts'
os.makedirs(output_dir, exist_ok=True)

for doc_id, words in documents.items():
    # Convert "UntN TZ 1" to "UntN_TZ_1.txt" (handle slashes and other special chars)
    filename = doc_id.replace(' ', '_').replace('/', '_') + '.txt'
    filepath = os.path.join(output_dir, filename)

    # Write words separated by spaces (one document per file)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(' '.join(words))

    print(f"Created: {filename} ({len(words)} words)")

print(f"\nTotal documents: {len(documents)}")
