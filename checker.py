import glob

import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

stemmer = nltk.stem.porter.PorterStemmer()

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def ngramize(text, n=3):
    tokens = stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
    ngrams = []
    for i in range(0, len(tokens)-n):
        ngrams.append("".join(tokens[i:i + n]))
    # print(ngrams)
    return np.copy(ngrams)


tfvect = TfidfVectorizer(analyzer=ngramize, min_df=0, stop_words='english', sublinear_tf=True)

input_directory = str(input("Specify input directory:"))
files = sorted(glob.glob(input_directory + "/*.txt"))
corpus = [open(file, encoding="utf8").read() for file in files]
# print(len(corpus))

tfidf_matrix = tfvect.fit_transform(corpus)

similarity_matrix = (tfidf_matrix*tfidf_matrix.T).A
# print(similarity_matrix)

for i in range(0, len(corpus)):
    for j in range(i+1, len(corpus)):
        if similarity_matrix[i,j] > 0.6:
            print(f"Document {files[i]} and {files[j]} is plagarised.")
