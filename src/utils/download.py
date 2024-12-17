import os
from torchaudio.utils import download_asset

# convert 3-gram to lowercase
import os

def LM_lowercase_convert(uppercase_lm_path, lowercase_lm_path):
    if not os.path.exists(lowercase_lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lowercase_lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')

uppercase_lm_path = '3-gram.arpa'
lowercase_lm_path = '3-gram_lc.arpa'
print('Converting 3-gram LM file to lowercase...')
LM_lowercase_convert(uppercase_lm_path, lowercase_lm_path)

uppercase_lm_path = '4-gram.arpa'
lowercase_lm_path = '4-gram_lc.arpa'
print('Converting 4-gram LM file to lowercase...')
LM_lowercase_convert(uppercase_lm_path, lowercase_lm_path)


# Create noise directory
SAMPLE_NOISE_PATH = "noise_samples"
os.makedirs(SAMPLE_NOISE_PATH, exist_ok=True)

# Download and save the VOiCES noise
NOISE_FILENAME = "voices_noise.wav"
noise_path = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")

# Copy the downloaded noise to your noise directory
import shutil
shutil.copy(noise_path, os.path.join(SAMPLE_NOISE_PATH, NOISE_FILENAME))
print("Noise sample has been copied to", os.path.join(SAMPLE_NOISE_PATH, NOISE_FILENAME))


print('All done!')