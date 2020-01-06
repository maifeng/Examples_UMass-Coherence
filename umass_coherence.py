from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents
from multiprocessing import Pool
from functools import partial
import math
import numpy as np


# use the newsgroup data as corpus
df = pd.read_json(
    "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
)
documents = df.content.tolist()
documents = preprocess_documents(documents)

# fit an LDA model, n_topic = 5
news_dictionary = Dictionary(documents)
news_dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=5000, keep_tokens=None)
corpus = [news_dictionary.doc2bow(text) for text in documents]
lda = gensim.models.LdaModel(corpus, num_topics=5, id2word=news_dictionary)

lda.show_topics()

# convert gensim corpus to a sparse document-term matrix for coherence measure
corpus_dense = gensim.matutils.corpus2csc(corpus, num_terms=len(news_dictionary.keys()))
corpus_dense = corpus_dense.astype(int)
corpus_dense = corpus_dense.transpose()
print(corpus_dense.shape)

# implements the UMass coherence in Mimno et al. 2011 - Optimizing Semantic Coherence in Topic Models
def cooccur_df_ws(w1, w2, corpus_dense, w2ids):
    """
    Returns the co-document frequency of two words
    """
    w1_id, w2_id = w2ids.token2id.get(w1), w2ids.token2id.get(w2)
    co_freq_array = (corpus_dense[:, [w1_id, w2_id]] > 0).sum(axis=1).A1
    return np.count_nonzero(co_freq_array == 2)


def word_lst_coherence(word_list, corpus_dense, w2ids):
    """
    Given a sequence of words, calculate the overall UMASS-coherence (eq 1 in the paper)
    """
    C = 0
    for i, w_rare in enumerate(word_list[1:]):
        for j, w_common in enumerate(word_list):
            # m = rare word, l = common word in the Mimno paper
            if i >= j:  # make sure the index of w_common is ahead of w_rare
                D_m_l = cooccur_df_ws(w_rare, w_common, corpus_dense, w2ids)
                D_l = w2ids.dfs[w2ids.token2id.get(w_common)]
                C = C + math.log((D_m_l + 1) / D_l)
    return C


def topic_model_coherence(model, corpus_dense, w2ids, n_top_words=20, n_core=4):
    """
    Calculate the average coherence of all the topics in a fitted LDA model
    """
    top_words_under_topics = []  # a list of list of top words under each topic
    for i in range(lda.num_topics):
        top_words_under_topics.append(
            [x[0] for x in model.show_topic(i, topn=n_top_words)]
        )

    with Pool(n_core) as pool:
        topic_coherences = pool.map(
            partial(word_lst_coherence, corpus_dense=corpus_dense, w2ids=w2ids),
            top_words_under_topics,
        )
    print(
        "topic coherence for a {}-topic model is {}".format(
            model.num_topics, np.mean(topic_coherences)
        )
    )
    return np.mean(topic_coherences)


topic_model_coherence(
    model=lda, corpus_dense=corpus_dense, w2ids=news_dictionary, n_top_words=20
)

