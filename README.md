# CJDClassification
learning to automatically classify Chinese judgment documents according to the industry involved in the factual content.

# Running order
extract_data.py  --> to extract x and y from original txts.

sentences2words.py  -->  to cut words in sentences

build_vocab.py  -->  to build vocabs

data_loader.py  -->  to transfer original texts to ids. for example: 我爱中国 -> character ids: 9, 89, 344, 1244

# Training and Test
train.py  -->  to train/test character level CNNs for CJDClassification/THUCnews/IMDb

train_bert.py  --> to train/test CWSB-CNN for CJDClassification/THUCnews/IMDb 
(Please note that this requires the use of bert for token level and sentence level encoding, see project:[`fine-tuning BERT`](https://github.com/realcactus/bert)

*   **[`legal_data`](https://github.com/realcactus/CJDClassification/blob/master/data/legal_domain/legal_domain.zip)**

*   **[`SubIMDb`](https://github.com/realcactus/legal_clas/blob/master/data/SubIMDb.zip)** (Need to manually divide the file into train.txt, val.txt and test.txt)

*   **[`SubTHUCNews`](https://github.com/realcactus/legal_clas/blob/master/data/SubTHUCNews.zip)** (Need to manually divide the file into train.txt, val.txt and test.txt)
