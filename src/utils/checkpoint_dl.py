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

if __name__ == "__main__":
    download_checkpoint()
