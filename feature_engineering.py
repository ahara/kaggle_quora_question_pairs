# -*- coding: utf-8 -*-
"""
Detecting duplicate quora questions
feature engineering
@author: Adam Harasimowicz
"""
import difflib
import gensim
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from fuzzywuzzy import fuzz
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.feature_extraction.text import TfidfVectorizer

import magic_feature_1
import magic_feature_2


stop_words = set(stopwords.words('english'))


def wmd(s1, s2, model):
    s1 = unicode(s1).lower().split()
    s2 = unicode(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def sent2vec(s, model):
    words = unicode(s).lower()#.decode('utf-8')
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


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(unicode(st1).lower(), unicode(st2).lower())

    return seq.ratio()


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in unicode(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in unicode(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def weighted_word_match(row):
    q1 = set(unicode(row['question1']).lower().split())
    q1words = q1.difference(stop_words)
    q2 = set(unicode(row['question2']).lower().split())
    q2words = q2.difference(stop_words)
    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    return np.sum(shared_weights) / np.sum(total_weights)


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


train = pd.read_csv('../data/train.csv', encoding='utf-8')
test = pd.read_csv('../data/test.csv', encoding='utf-8')

magic_1 = magic_feature_1.magic_feature_1(train, test)
magic_2 = magic_feature_2.magic_feature_2(train, test)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(unicode)
tfidf.fit_transform(tfidf_txt)
del train, test, tfidf_txt


for fname in ['train', 'test']:
    if 'reverse' in fname:
        dataset = pd.read_csv('../data/%s.csv' % fname.replace('_reverse', ''), encoding='utf-8')
        dataset.rename(index=str, columns={'qid1': 'qid2', 'qid2': 'qid1', 'question1': 'question2',
                                           'question2': 'question1'})
    else:
        dataset = pd.read_csv('../data/%s.csv' % fname, encoding='utf-8')

    for batch_id in xrange(6):
        print 'Batch id: %d' % batch_id
        batch_size = 5e6
        batch_start = batch_id * batch_size
        batch_end = (batch_id + 1) * batch_size

        if dataset.shape[0] < batch_start:
            break

        batch_end = min(batch_end, dataset.shape[0])
        data = dataset.loc[batch_start:batch_end, :].copy()

        print 'Magic feature 1'
        data.loc[:, 'm1_q1_freq'] = magic_1[fname].loc[batch_start:batch_end, 'm1_q1_freq']
        data.loc[:, 'm1_q2_freq'] = magic_1[fname].loc[batch_start:batch_end, 'm1_q2_freq']

        print 'Magic feature 2'
        data.loc[:, 'm2_q1_q2_intersect'] = magic_2[fname].loc[batch_start:batch_end, 'm2_q1_q2_intersect']
        data.loc[:, 'm2_q1_q2_intersect'] = magic_2[fname].loc[batch_start:batch_end, 'm2_q1_q2_intersect']

        print 'Make basic features'
        data.loc[:, 'len_q1'] = data.question1.apply(lambda x: len(unicode(x)))
        data.loc[:, 'len_q2'] = data.question2.apply(lambda x: len(unicode(x)))
        data.loc[:, 'diff_len'] = data.len_q1 - data.len_q2

        data.loc[:, 'len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(unicode(x).replace(' ', '')))))
        data.loc[:, 'len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(unicode(x).replace(' ', '')))))
        data.loc[:, 'diff_len_char'] = data.len_char_q1 - data.len_char_q2

        data.loc[:, 'len_word_q1'] = data.question1.apply(lambda x: len(unicode(x).split()))
        data.loc[:, 'len_word_q2'] = data.question2.apply(lambda x: len(unicode(x).split()))
        data.loc[:, 'diff_len_word'] = data.len_word_q1 - data.len_word_q2

        data.loc[:, 'avg_word_len1'] = data.len_char_q1 / data.len_word_q1
        data.loc[:, 'avg_word_len2'] = data.len_char_q2 / data.len_word_q2
        data.loc[:, 'diff_avg_word'] = data.avg_word_len1 - data.avg_word_len2

        data.loc[:, 'exactly_same'] = (data.question1 == data.question2).astype(int)
        data.loc[:, 'duplicated'] = data.duplicated(['question1', 'question2']).astype(int)

        data.loc[:, 'stop_words_q1'] = data.question1.apply(lambda x: len(set(unicode(x).split()).intersection(stop_words)))
        data.loc[:, 'stop_words_q2'] = data.question2.apply(lambda x: len(set(unicode(x).split()).intersection(stop_words)))
        data.loc[:, 'stops1_ratio'] = data.stop_words_q1 / data.len_word_q1.astype(float)
        data.loc[:, 'stops2_ratio'] = data.stop_words_q2 / data.len_word_q2.astype(float)
        data.loc[:, 'diff_stops_r'] = data.stops1_ratio - data.stops2_ratio

        train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(unicode)
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}

        data.loc[:, 'weighted_word_match'] = data.apply(weighted_word_match, axis=1)
        data.loc[:, 'common_words'] = data.apply(lambda x: len(set(unicode(x['question1']).lower().split()).intersection(set(unicode(x['question2']).lower().split()))), axis=1)
        data.loc[:, 'fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(unicode(x['question1']), unicode(x['question2'])), axis=1)
        data.loc[:, 'fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(unicode(x['question1']), unicode(x['question2'])), axis=1)

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

        print 'Question word features'
        # What
        data['q1_count_word_what'] = data.question1.apply(lambda x: unicode(x).lower().count('what'))
        data['q2_count_word_what'] = data.question2.apply(lambda x: unicode(x).lower().count('what'))
        data['diff_count_word_what'] = data['q1_count_word_what'] - data['q2_count_word_what']
        # How
        data['q1_count_word_how'] = data.question1.apply(lambda x: unicode(x).lower().count('how'))
        data['q2_count_word_how'] = data.question2.apply(lambda x: unicode(x).lower().count('how'))
        data['diff_count_word_how'] = data['q1_count_word_how'] - data['q2_count_word_how']
        # Why
        data['q1_count_word_why'] = data.question1.apply(lambda x: unicode(x).lower().count('why'))
        data['q2_count_word_why'] = data.question2.apply(lambda x: unicode(x).lower().count('why'))
        data['diff_count_word_why'] = data['q1_count_word_why'] - data['q2_count_word_why']
        # How many
        data['q1_count_word_how_many'] = data.question1.apply(lambda x: unicode(x).lower().count('how many'))
        data['q2_count_word_how_many'] = data.question2.apply(lambda x: unicode(x).lower().count('how many'))
        data['diff_count_word_how_many'] = data['q1_count_word_how_many'] - data['q2_count_word_how_many']
        # How much
        data['q1_count_word_how_much'] = data.question1.apply(lambda x: unicode(x).lower().count('how much'))
        data['q2_count_word_how_much'] = data.question2.apply(lambda x: unicode(x).lower().count('how much'))
        data['diff_count_word_how_much'] = data['q1_count_word_how_much'] - data['q2_count_word_how_much']
        # If
        data['q1_count_word_if'] = data.question1.apply(lambda x: unicode(x).lower().count('if'))
        data['q2_count_word_if'] = data.question2.apply(lambda x: unicode(x).lower().count('if'))
        data['diff_count_word_if'] = data['q1_count_word_if'] - data['q2_count_word_if']
        # When
        data['q1_count_word_when'] = data.question1.apply(lambda x: unicode(x).lower().count('when'))
        data['q2_count_word_when'] = data.question2.apply(lambda x: unicode(x).lower().count('when'))
        data['diff_count_word_when'] = data['q1_count_word_when'] - data['q2_count_word_when']
        # [math]
        data['q1_count_symbol_math'] = data.question1.apply(lambda x: unicode(x).lower().count('[math]'))
        data['q2_count_symbol_math'] = data.question2.apply(lambda x: unicode(x).lower().count('[math]'))
        data['diff_count_symbol_math'] = data['q1_count_symbol_math'] - data['q2_count_symbol_math']
        # ? (question mark)
        data['q1_count_symbol_question_mark'] = data.question1.apply(lambda x: unicode(x).lower().count('?'))
        data['q2_count_symbol_question_mark'] = data.question2.apply(lambda x: unicode(x).lower().count('?'))
        data['diff_count_symbol_question_mark'] = data['q1_count_symbol_question_mark'] - data['q2_count_symbol_question_mark']
        # . (dot)
        data['q1_count_symbol_dot'] = data.question1.apply(lambda x: unicode(x).lower().count('.'))
        data['q2_count_symbol_dot'] = data.question2.apply(lambda x: unicode(x).lower().count('.'))
        data['diff_count_symbol_dot'] = data['q1_count_symbol_dot'] - data['q2_count_symbol_dot']
        # Upper count
        data['q1_count_symbol_upper_letter'] = data.question1.apply(lambda x: sum(1 for c in unicode(x) if c.isupper()))
        data['q2_count_symbol_upper_letter'] = data.question2.apply(lambda x: sum(1 for c in unicode(x) if c.isupper()))
        data['diff_count_symbol_upper_letter'] = data['q1_count_symbol_upper_letter'] - data['q2_count_symbol_upper_letter']
        # Is first upper
        data['q1_is_first_upper_letter'] = data.question1.apply(lambda x: unicode(x)[0].isupper() if len(unicode(x)) else -1)
        data['q2_is_first_upper_letter'] = data.question2.apply(lambda x: unicode(x)[0].isupper() if len(unicode(x)) else -1)
        data['diff_is_first_upper_letter'] = data['q1_is_first_upper_letter'] - data['q2_is_first_upper_letter']
        # Digit count
        data['q1_count_symbol_digit'] = data.question1.apply(lambda x: sum(1 for c in unicode(x) if c.isdigit()))
        data['q2_count_symbol_digit'] = data.question2.apply(lambda x: sum(1 for c in unicode(x) if c.isdigit()))
        data['diff_count_symbol_digit'] = data['q1_count_symbol_digit'] - data['q2_count_symbol_digit']

        # Noun features
        print('nouns...')
        data['question1_nouns'] = data.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(unicode(x).lower())) if t[:1] in ['N']])
        data['question2_nouns'] = data.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(unicode(x).lower())) if t[:1] in ['N']])
        data['z_noun_match'] = data.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
        data['question1_nouns_len'] = data.question1_nouns.apply(lambda x: len(x))
        data['question2_nouns_len'] = data.question2_nouns.apply(lambda x: len(x))
        data = data.drop(['question1_nouns', 'question2_nouns'], axis=1)
        print('lengths...')
        data['z_len1'] = data.question1.map(lambda x: len(unicode(x)))
        data['z_len2'] = data.question2.map(lambda x: len(unicode(x)))
        data['z_word_len1'] = data.question1.map(lambda x: len(unicode(x).split()))
        data['z_word_len2'] = data.question2.map(lambda x: len(unicode(x).split()))
        print('difflib...')
        data['z_match_ratio'] = data.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
        print('word match...')
        data['z_word_match'] = data.apply(word_match_share, axis=1, raw=True)
        print('tfidf...')
        data['z_tfidf_sum1'] = data.question1.map(lambda x: np.sum(tfidf.transform([unicode(x)]).data))
        data['z_tfidf_sum2'] = data.question2.map(lambda x: np.sum(tfidf.transform([unicode(x)]).data))
        data['z_tfidf_mean1'] = data.question1.map(lambda x: np.mean(tfidf.transform([unicode(x)]).data))
        data['z_tfidf_mean2'] = data.question2.map(lambda x: np.mean(tfidf.transform([unicode(x)]).data))
        data['z_tfidf_len1'] = data.question1.map(lambda x: len(tfidf.transform([unicode(x)]).data))
        data['z_tfidf_len2'] = data.question2.map(lambda x: len(tfidf.transform([unicode(x)]).data))
        data.fillna(0.0)

        print 'Fill missing values'
        data.loc[:, 'cosine_distance'].fillna(1, inplace=True)
        data.update(data.loc[:, ('jaccard_distance', 'braycurtis_distance')].fillna(0))
        data.update(data.loc[:, ('z_noun_match', 'question1_nouns_len', 'question2_nouns_len',
                                 'z_len1', 'z_len2', 'z_word_len1', 'z_word_len2', 'z_match_ratio',
                                 'z_word_match', 'z_tfidf_sum1', 'z_tfidf_sum2', 'z_tfidf_mean1',
                                 'z_tfidf_mean2', 'z_tfidf_len1', 'z_tfidf_len2')].fillna(0))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        print 'Save results'
        #data.to_csv('../data/train_features_%d_%d.csv' % (batch_start, batch_end), index=False)
        data.to_csv('../data/%s_features_v3.csv' % fname, index=False, encoding='utf-8')
