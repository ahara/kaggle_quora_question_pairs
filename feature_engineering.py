"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""
import pandas as pd
import numpy as np
import gensim
from collections import Counter
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize

from data_enrichment import make_enriched_dataset


stop_words = set(stopwords.words('english'))


def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def sent2vec(s, model):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if w not in stop_words and w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    vec = v / np.sqrt((v ** 2).sum())

    return np.zeros((300, )) if np.isnan(vec).all() else vec


def get_metrics(q1, q2, model):
    v1 = sent2vec(q1, model)
    v2 = sent2vec(q2, model)
    q1v = np.nan_to_num(v1)
    q2v = np.nan_to_num(v2)

    metrics = dict()
    metrics['cosine_distance'] = cosine(q1v, q2v)
    metrics['cityblock_distance'] = cityblock(q1v, q2v)
    metrics['jaccard_distance'] = jaccard(q1v, q2v)
    metrics['canberra_distance'] = canberra(q1v, q2v)
    metrics['euclidean_distance'] = euclidean(q1v, q2v)
    metrics['minkowski_distance'] = minkowski(q1v, q2v, 3)
    metrics['braycurtis_distance'] = braycurtis(q1v, q2v)

    metrics['skew_q1vec'] = skew(q1v)
    metrics['skew_q2vec'] = skew(q2v)
    metrics['kur_q1vec'] = kurtosis(q1v)
    metrics['kur_q2vec'] = kurtosis(q2v)

    return pd.Series(metrics)


#for fname in ['train', 'test']:
for fname in ['test']:
    dataset = pd.read_csv('../data/%s.csv' % fname)

    for batch_id in xrange(6):
        print 'Batch id: %d' % batch_id
        batch_size = 5e6
        batch_start = batch_id * batch_size
        batch_end = (batch_id + 1) * batch_size

        if dataset.shape[0] < batch_start:
            break

        batch_end = min(batch_end, dataset.shape[0])
        data = dataset.loc[batch_start:batch_end, :].copy()

        print 'Make basic features'
        data.loc[:, 'len_q1'] = data.question1.apply(lambda x: len(str(x)))
        data.loc[:, 'len_q2'] = data.question2.apply(lambda x: len(str(x)))
        data.loc[:, 'diff_len'] = data.len_q1 - data.len_q2

        data.loc[:, 'len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data.loc[:, 'len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data.loc[:, 'diff_len_char'] = data.len_char_q1 - data.len_char_q2

        data.loc[:, 'len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
        data.loc[:, 'len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
        data.loc[:, 'diff_len_word'] = data.len_word_q1 - data.len_word_q2

        data.loc[:, 'avg_word_len1'] = data.len_char_q1 / data.len_word_q1
        data.loc[:, 'avg_word_len2'] = data.len_char_q2 / data.len_word_q2
        data.loc[:, 'diff_avg_word'] = data.avg_word_len1 - data.avg_word_len2

        data.loc[:, 'exactly_same'] = (data.question1 == data.question2).astype(int)
        data.loc[:, 'duplicated'] = data.duplicated(['question1', 'question2']).astype(int)

        data.loc[:, 'stop_words_q1'] = data.question1.apply(lambda x: len(set(str(x).split()).intersection(stop_words)))
        data.loc[:, 'stop_words_q2'] = data.question2.apply(lambda x: len(set(str(x).split()).intersection(stop_words)))
        data.loc[:, 'stops1_ratio'] = data.stop_words_q1 / data.len_word_q1.astype(float)
        data.loc[:, 'stops2_ratio'] = data.stop_words_q2 / data.len_word_q2.astype(float)
        data.loc[:, 'diff_stops_r'] = data.stops1_ratio - data.stops2_ratio

        # If a word appears only once, we ignore it completely (likely a typo)
        # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
        def get_weight(count, eps=10000, min_count=2):
            return 0 if count < min_count else 1 / (count + eps)

        train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(str)
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}

        def weighted_word_match(row):
            q1 = set(str(row['question1']).lower().split())
            q1words = q1.difference(stop_words)
            q2 = set(str(row['question2']).lower().split())
            q2words = q2.difference(stop_words)
            shared_words = q1words.intersection(q2words)
            shared_weights = [weights.get(w, 0) for w in shared_words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            return np.sum(shared_weights) / np.sum(total_weights)

        data.loc[:, 'weighted_word_match'] = data.apply(weighted_word_match, axis=1)
        data.loc[:, 'common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
        data.loc[:, 'fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data.loc[:, 'fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

        data.loc[:, 'tfidf_word_match'] = data.common_words / (data.len_word_q1 + data.len_word_q2)

        print 'Word2vec'
        model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        data.loc[:, 'wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], model), axis=1)

        print 'Sentence2vec'
        vector_features = data.apply(lambda x: get_metrics(x.question1, x.question2, model), axis=1)
        data = pd.concat([data, vector_features], axis=1)

        print 'Normalized word2vec'
        norm_model = model
        model = None
        norm_model.init_sims(replace=True)
        data.loc[:, 'norm_wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], norm_model), axis=1)

        norm_model = None

        print 'Save results'
        #data.to_csv('../data/train_features_%d_%d.csv' % (batch_start, batch_end), index=False)
        data.to_csv('../data/%s_features.csv' % fname, index=False)
