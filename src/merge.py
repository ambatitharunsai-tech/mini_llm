import pandas as pd
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # mini_llm/ — one level up from src/

def merge_datasets(new_file):
    base = os.path.join(_ROOT, "data", "chat_dataset.csv")

    # ── Load existing dataset ─────────────────────────────────────────────────
    old = pd.read_csv(base, dtype=str, low_memory=False).fillna("")
    old = old[old['input'].str.strip().str.lower() != 'input']
    # Keep only input/output — drop example column if present
    old = old[['input', 'output']]
    print(f"Existing rows: {len(old)}")

    # ── Load new data — accept multiple column naming conventions ─────────────
    new = pd.read_csv(new_file, dtype=str, low_memory=False).fillna("")
    # Normalise column names to lowercase
    new.columns = [c.lower().strip() for c in new.columns]
    new = new.rename(columns={
        'instruction': 'input',  'question': 'input',  'prompt':      'input',
        'response':    'output', 'answer':   'output', 'completion':  'output',
        'review':      'input',  'text':     'input',  'sentence':    'input',
    })

    # Check we have both required columns
    if 'input' not in new.columns or 'output' not in new.columns:
        print(f"❌ Could not find input/output columns.")
        print(f"   Available columns: {list(new.columns)}")
        print(f"   Please rename your columns to: input, output")
        return

    new = new[new['input'].str.strip().str.lower() != 'input']
    new = new[['input', 'output']]
    # Drop rows where either column is empty
    new = new[new['input'].str.strip().ne("") & new['output'].str.strip().ne("")]
    print(f"New rows to add: {len(new)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    combined = pd.concat([old, new], ignore_index=True)
    combined['_score'] = combined['output'].str.len()
    combined = combined.sort_values('_score', ascending=False)
    combined = combined.drop_duplicates(subset='input', keep='first')
    combined = combined.drop(columns=['_score']).reset_index(drop=True)
    combined = combined[['input', 'output']]
    print(f"Total after merge: {len(combined)}")

    combined.to_csv(base, index=False)
    print(f"✅ Saved to {base}")


if __name__ == "__main__":
    new_file = input("Enter path to your new CSV file: ").strip()
    new_file = new_file.strip("'\"")
    if os.path.exists(new_file):
        merge_datasets(new_file)
    else:
        print(f"❌ File not found: {new_file}")
        print(f"   Check the path and try again.")


if __name__ == "__main__":
    new_file = input("Enter path to your new CSV file: ").strip()
    if os.path.exists(new_file):
        merge_datasets(new_file)
    else:
        print(f"File not found: {new_file}")

## How to Use
##
## Your new CSV can use any of these column names:
##
##   input / output         (default Hema format)
##   instruction / response (Alpaca / LLaMA format)
##   question / answer      (QA format)
##   prompt / completion    (OpenAI format)
##
## Example CSV (instruction/response):
##   instruction,response
##   What is Python?,"Python is a high-level programming language..."
##   How does recursion work?,"Recursion is when a function calls itself..."
##
## Run:
##   python merge.py
##   > Enter path to your new CSV file: data/my_new_data.csv