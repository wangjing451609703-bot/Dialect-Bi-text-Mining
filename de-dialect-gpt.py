import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = <YOUR API KEY>
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

sentences = []
de_path = "/content/drive/MyDrive/de.txt" # de corpus path
with open(de_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.append(line)

len(sentences)

def build_prompt_simple(text):
    return f"Translate the following Standard German sentence into natural, fluent Swiss German.Try to aim for diverse translations.Only output the translation.\n\nGerman: {text}\nSwissGerman:"

MODEL_NAME = "gpt-4o"

def translate_batch(text_batch):
    results = []
    for text in text_batch:
        prompt = build_prompt_simple(text)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        answer = completion.choices[0].message.content.strip()
        results.append(answer)
    return results

batch_size = 50
all_gsw = [] # change to nds/gsw/bar
sentence = sentences[26000:31000] # source sentences from Tatoeba(eng-de)
#test_batch = sentences[:100]

for i in tqdm(range(0, len(sentence), batch_size)):
    batch = sentence[i:i+batch_size]
    batch_out = translate_batch(batch)
    all_gsw.extend(batch_out)

len(all_gsw)

df = pd.DataFrame({
    "de": sentence,
    "nds": all_gsw # change to nds/gsw/bar
      })
df.to_csv("/content/drive/MyDrive/de-gsw-gpt4o-2.csv", index=False, encoding="utf-8-sig")
