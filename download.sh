# This script downloads the necessary data for the project to run

mkdir -p data
cd data
wget -nc https://github.com/mahnazkoupaee/WikiHow-Dataset/raw/master/all_train.txt
wget -nc https://github.com/mahnazkoupaee/WikiHow-Dataset/raw/master/all_test.txt
wget -nc https://github.com/mahnazkoupaee/WikiHow-Dataset/raw/master/all_val.txt

# Download dataset from google drive
FILEID="1VEJFS0JkU_d9rMXFA-xdZyArI0MD8UyR"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O wikihowAll.csv
rm -rf /tmp/cookies.txt

# Preprocess data
python preprocess_data.py