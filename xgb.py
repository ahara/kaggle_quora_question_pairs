# -*- coding: utf-8 -*-
import difflib
import nltk
import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


MISSING = -99999

stop_words = set(stopwords.words('english'))


def load_data(file_name):
    data = pd.read_csv('../data/%s' % file_name, encoding='utf-8')
    data.loc[:, 'cosine_distance'].fillna(1, inplace=True)
    data.update(data.loc[:, ('jaccard_distance', 'braycurtis_distance')].fillna(0))

    return data


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(unicode(st1).lower(), unicode(st2).lower())

    return seq.ratio()


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


def get_train_validation_sets(testset=False, tfidf_obj=None):
    if not testset:
        data = load_data('train_features.csv')
        test = pd.read_csv('../data/test.csv', encoding='utf-8')
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(data['question1'].tolist() + data['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(unicode)
        tfidf.fit_transform(tfidf_txt)
        del test, tfidf_txt
    else:
        data = load_data('test_features.csv')
        tfidf = tfidf_obj

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

    if testset:
        print 'Prepare test set'
        test_id = data['test_id']
        x_test = data.drop(['test_id', 'question1', 'question2'], axis=1)

        x_test.fillna(MISSING, inplace=True)

        return x_test, test_id
    else:
        print 'Prepare train and validation sets'
        y_train = data['is_duplicate']
        x_train = data.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)

        x_train.fillna(MISSING, inplace=True)

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

        return x_train, x_valid, y_train, y_valid, tfidf


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid, tfidf_obj = get_train_validation_sets()

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.05  # 0.05
    params['max_depth'] = 16  # 11
    params['subsample'] = 0.8
    params['max_delta_step'] = 1
    params['min_child_weight'] = 2
    params['scale_pos_weight'] = 0.165 / (1 - 0.165)

    x_train2 = pd.concat([x_train, x_valid])
    y_train2 = pd.concat([y_train, y_valid])

    d_train = xgb.DMatrix(x_train2, label=y_train2, missing=MISSING)
    d_valid = xgb.DMatrix(x_valid, label=y_valid, missing=MISSING)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print 'Start training'

    bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=10)

    del d_train, d_valid

    x_test, test_id = get_train_validation_sets(testset=True, tfidf_obj=tfidf_obj)

    preds = bst.predict(xgb.DMatrix(x_test.fillna(MISSING), missing=MISSING))

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = preds
    sub.to_csv('xgb_preds.csv', index=False)

    import pdb
    pdb.set_trace()
