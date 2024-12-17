# download LM model 4-gram.arpa
echo "Downloading LM model 4-gram.arpa"
wget https://www.openslr.org/resources/11/4-gram.arpa.gz
gunzip 4-gram.arpa.gz

# download LM model 3-gram.arpa
echo "Downloading LM model 3-gram.arpa"
wget https://www.openslr.org/resources/11/3-gram.arpa.gz
gunzip 3-gram.arpa.gz

# download unigram vocab
echo "Downloading unigram vocab"
wget http://www.openslr.org/resources/11/librispeech-vocab.txt

# convert models to lowercase
echo "Converting models to lowercase and downloading noise data"
python3 src/utils/download.py

# convert arpa models to binary
echo "Converting arpa models to binary"
./src/utils/LM_build.sh 4-gram.arpa 4-gram.bin
./src/utils/LM_build.sh 3-gram.arpa 3-gram.bin

echo "Done"
