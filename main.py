
from __future__ import division
import codecs
import re
#"codecs" is for reading the text files, "re" (regular expretions) and "collections" for working with tokens, "nltk" (natural language toolkit)
import copy
import collections

import numpy as np
import pandas as pd

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

#bunu axtarib tapmag lazimdufrom __future__ import division



import  matplotlib
#matplotlib inline
nltk.download('stopwords')
# Stop words areremove  common words like ‘the’, ‘and’, ‘I’, and etc

from nltk.corpus import stopwords

with codecs.open("C:\\Users\\rufat\PycharmProjects\\TextminingwithPhyton/AliandNino.txt", "r", encoding="utf-8") as f:
    text_AN = f.read()
with codecs.open("C:\\Users\\rufat\PycharmProjects\\TextminingwithPhyton/LeylaandMajnun.txt", "r", encoding="utf-8") as f:
    text_LM = f.read()
esw = stopwords.words('english')
esw.append("would")
word_pattern = re.compile("^\w+$")


def get_text_counter(text):
    tokens = WordPunctTokenizer().tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    return collections.Counter(tokens), len(tokens)

def make_df(counter, size):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq / size
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq, rel_freq]).T, index=index, columns=["Absolute frequency", "Relative frequency"])
    df.index.name = "Most common words"
    return df

an_counter, an_size = get_text_counter(text_AN)



print(make_df(an_counter.most_common(15), an_size))


lm_counter, lm = get_text_counter(text_LM)
print(make_df(lm_counter.most_common(15), lm))

all_counter = lm_counter + an_counter
all_df = make_df(lm_counter.most_common(1000), 1)
most_common_words = all_df.index.values
#print(most_common_words)

#Create a data frame with the differences in word frequency
df_data = []
for word in most_common_words:
    an_c = an_counter.get(word, 0) / an_size
    lm_c = lm_counter.get(word, 0) / lm
    d = abs(an_c - lm_c)
    df_data.append([an_c, lm_c, d])

diff_df = pd.DataFrame(data=df_data, index=most_common_words,
                           columns=["AN relative frequency", "LM relative frequency",
                                    "Differences in relative frequency"])
diff_df.index.name = "Most common words"
diff_df.sort_values("Differences in relative frequency", ascending=False, inplace=True)


print(diff_df.head(20))