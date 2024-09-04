
import pandas as pd

def prepare_dataset():
    # Read the text as csv file
    df1 = pd.read_csv(r"data\ara_eng.txt",delimiter="\t",names=["english","arabic"])

    # Read the text file and remove empty lines
    with open(r'data\translation_data.txt', 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    # Initialize lists to store English and Arabic sentences
    english_sentences = []
    arabic_sentences = []

    # Process the lines to extract English and Arabic sentences
    for i in range(0, len(lines), 2):
        english_sentence = lines[i].strip()
        # Check if there is a corresponding Arabic sentence available
        if i + 1 < len(lines):
            arabic_sentence = lines[i + 1].strip()
        else:
            arabic_sentence = ''
        english_sentences.append(english_sentence)
        arabic_sentences.append(arabic_sentence)

    # Create a DataFrame
    df2 = pd.DataFrame({
        'english': english_sentences,
        'arabic': arabic_sentences,
    })

    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv(r"data\merge_df.csv", index=False)