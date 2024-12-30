import os
import re
from tqdm import tqdm
import sentencepiece as spm

def process_text_files(folders, output_file_name="../../aggregated_result.txt"):
    """
    Aggregate text from all .txt files in the provided folders and save it to a single file.
    """
    aggregated_content = ""

    total_files = sum(len(files) for folder_name in folders for _, _, files in os.walk(folder_name) if files)

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for folder_name in folders:
            for root, _, files in os.walk(folder_name):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                processed_line = re.sub(r"^[\d-]+\s*", "", line)
                                aggregated_content += processed_line
                        pbar.update(1)

    with open(output_file_name, "w", encoding="utf-8") as output_file:
        output_file.write(aggregated_content)

    print(f"Aggregated content saved to {output_file_name}")

def remove_punctuation(text):
    """
    Remove all punctuation from the input text using regex.
    """
    return re.sub(r"[^\w\s]", "", text)

def train_sentencepiece_model(input_file="../../aggregated_result.txt", 
                              output_dir="../../sentencepiece_model", 
                              preprocessed_file="../../preprocessed_unigrams.txt", 
                              vocab_size=1000, 
                              batch_size=10):
    """
    Preprocess the input file and train a SentencePiece model.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Preprocessing unigram file...")
    with open(input_file, "r", encoding="utf-8") as infile, open(preprocessed_file, "w", encoding="utf-8") as outfile:
        batch = []
        for line in infile:
            line = line.lower()
            word = remove_punctuation(line.strip())
            if word:
                batch.append(word)
            if len(batch) >= batch_size:
                outfile.write(" ".join(batch) + "\n")
                batch = []
        if batch:
            outfile.write(" ".join(batch) + "\n")

    print(f"[INFO] Preprocessed file saved to {preprocessed_file}")

    model_prefix = os.path.join(output_dir, "librispeech_unigram_model")
    spm.SentencePieceTrainer.train(
        input=preprocessed_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        pad_id=1,
        shuffle_input_sentence=True,
        normalization_rule_name="identity"
    )

    print(f"[INFO] SentencePiece model saved to {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    # Example usage
    folders = ['../../data/datasets/librispeech/train-clean-100', '../../data/datasets/librispeech/train-clean-360', '../../data/datasets/librispeech/train-clean-500']
    aggregated_file = "../../aggregated_result.txt"

    process_text_files(folders, output_file_name=aggregated_file)
    train_sentencepiece_model(input_file=aggregated_file)
