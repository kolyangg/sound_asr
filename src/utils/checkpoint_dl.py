import os
import gdown

def download_checkpoint():

    
    gdrive_file_id = "1L8eeDb1R_g29w02Ysl3kgx1SulpgMtLI"
    output_path = "saved_big_bpe/ds2_lm/model_best.pth"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct the Google Drive URL
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    # Download the file using gdown
    gdown.download(url, output_path, quiet=False)

    print(f"Checkpoint downloaded and saved to: {output_path}")



def download_tokenizer():

    
    gdrive_file_id = "1xY3sgbFo-QD_gp3jocc-GT9kv3AKoT6V"
    output_path =  "sentencepiece_model/librispeech_unigram_model1000_new2.model"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct the Google Drive URL
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    # Download the file using gdown
    gdown.download(url, output_path, quiet=False)

    print(f"BPE model downloaded and saved to: {output_path}")



    gdrive_file_id = "1kx-HbdiUrmyScaqOcW1PwDLDoGLkMgWn"
    output_path =  "sentencepiece_model/librispeech_unigram_model1000_new2.vocab"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct the Google Drive URL
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    # Download the file using gdown
    gdown.download(url, output_path, quiet=False)

    print(f"BPE vocab downloaded and saved to: {output_path}")



if __name__ == "__main__":
    download_checkpoint()
    download_tokenizer()
