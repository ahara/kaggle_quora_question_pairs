import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

RS = 12357
ROUNDS = 315
MISSING = -99999

print("Started")
np.random.seed(RS)
input_folder = '../data/'


def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

    x_train.fillna(MISSING, inplace=True)
    x_val.fillna(MISSING, inplace=True)

    xg_train = xgb.DMatrix(x_train, label=y_train, missing=MISSING)
    xg_val = xgb.DMatrix(x_val, label=y_val, missing=MISSING)

    watchlist = [(xg_train, 'train'), (xg_val, 'eval')]

    return xgb.train(params, xg_train, ROUNDS, watchlist)


def predict_xgb(clr, x_test):
    return clr.predict(xgb.DMatrix(x_test.fillna(MISSING), missing=MISSING))


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


def main():
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.05  # 0.05
    params['max_depth'] = 18  # 16
    params['silent'] = 1
    params['seed'] = RS

    df_train = pd.read_csv(input_folder + 'train.csv')
    df_test = pd.read_csv(input_folder + 'test.csv')
    print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

    print("Features processing, be patient...")

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    stops = set(stopwords.words("english"))

    def word_shares(row):
        q1 = set(str(row['question1']).lower().split())
        q1words = q1.difference(stops)
        if len(q1words) == 0:
            return '0:0:0:0:0'

        q2 = set(str(row['question2']).lower().split())
        q2words = q2.difference(stops)
        if len(q2words) == 0:
            return '0:0:0:0:0'

        q1stops = q1.intersection(stops)
        q2stops = q2.intersection(stops)

        shared_words = q1words.intersection(q2words)
        shared_weights = [weights.get(w, 0) for w in shared_words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
        R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
        R31 = len(q1stops) / len(q1words) #stops in q1
        R32 = len(q2stops) / len(q2words) #stops in q2
        return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)

    df = pd.concat([df_train, df_test])
    df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

    x = pd.DataFrame()

    x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))  # Done
    x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))  # Done

    x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))  # Done
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))  # Done
    x['diff_len'] = x['len_q1'] - x['len_q2']  # Done

    x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))  # Done
    x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))  # Done
    x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']  # Done

    x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))  # Done
    x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))  # Done
    x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']  # Done

    x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
    x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

    x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
    x['duplicated'] = df.duplicated(['question1', 'question2']).astype(int)

    # What
    x['q1_count_word_what'] = df.question1.apply(lambda x: str(x).lower().count('what'))
    x['q2_count_word_what'] = df.question2.apply(lambda x: str(x).lower().count('what'))
    x['diff_count_word_what'] = x['q1_count_word_what'] - x['q2_count_word_what']
    # How
    x['q1_count_word_how'] = df.question1.apply(lambda x: str(x).lower().count('how'))
    x['q2_count_word_how'] = df.question2.apply(lambda x: str(x).lower().count('how'))
    x['diff_count_word_how'] = x['q1_count_word_how'] - x['q2_count_word_how']
    # Why
    x['q1_count_word_why'] = df.question1.apply(lambda x: str(x).lower().count('why'))
    x['q2_count_word_why'] = df.question2.apply(lambda x: str(x).lower().count('why'))
    x['diff_count_word_why'] = x['q1_count_word_why'] - x['q2_count_word_why']
    # How many
    x['q1_count_word_how_many'] = df.question1.apply(lambda x: str(x).lower().count('how many'))
    x['q2_count_word_how_many'] = df.question2.apply(lambda x: str(x).lower().count('how many'))
    x['diff_count_word_how_many'] = x['q1_count_word_how_many'] - x['q2_count_word_how_many']
    # How much
    x['q1_count_word_how_much'] = df.question1.apply(lambda x: str(x).lower().count('how much'))
    x['q2_count_word_how_much'] = df.question2.apply(lambda x: str(x).lower().count('how much'))
    x['diff_count_word_how_much'] = x['q1_count_word_how_much'] - x['q2_count_word_how_much']
    # If
    x['q1_count_word_if'] = df.question1.apply(lambda x: str(x).lower().count('if'))
    x['q2_count_word_if'] = df.question2.apply(lambda x: str(x).lower().count('if'))
    x['diff_count_word_if'] = x['q1_count_word_if'] - x['q2_count_word_if']
    # When
    x['q1_count_word_when'] = df.question1.apply(lambda x: str(x).lower().count('when'))
    x['q2_count_word_when'] = df.question2.apply(lambda x: str(x).lower().count('when'))
    x['diff_count_word_when'] = x['q1_count_word_when'] - x['q2_count_word_when']
    # [math]
    x['q1_count_symbol_math'] = df.question1.apply(lambda x: str(x).lower().count('[math]'))
    x['q2_count_symbol_math'] = df.question2.apply(lambda x: str(x).lower().count('[math]'))
    x['diff_count_symbol_math'] = x['q1_count_symbol_math'] - x['q2_count_symbol_math']
    # ? (question mark)
    x['q1_count_symbol_question_mark'] = df.question1.apply(lambda x: str(x).lower().count('?'))
    x['q2_count_symbol_question_mark'] = df.question2.apply(lambda x: str(x).lower().count('?'))
    x['diff_count_symbol_question_mark'] = x['q1_count_symbol_question_mark'] - x['q2_count_symbol_question_mark']
    # . (dot)
    x['q1_count_symbol_dot'] = df.question1.apply(lambda x: str(x).lower().count('.'))
    x['q2_count_symbol_dot'] = df.question2.apply(lambda x: str(x).lower().count('.'))
    x['diff_count_symbol_dot'] = x['q1_count_symbol_dot'] - x['q2_count_symbol_dot']
    # Upper count
    x['q1_count_symbol_upper_letter'] = df.question1.apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    x['q2_count_symbol_upper_letter'] = df.question2.apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    x['diff_count_symbol_upper_letter'] = x['q1_count_symbol_upper_letter'] - x['q2_count_symbol_upper_letter']
    # Is first upper
    x['q1_is_first_upper_letter'] = df.question1.apply(lambda x: str(x)[0].isupper() if len(str(x)) else -1)
    x['q2_is_first_upper_letter'] = df.question2.apply(lambda x: str(x)[0].isupper() if len(str(x)) else -1)
    x['diff_is_first_upper_letter'] = x['q1_is_first_upper_letter'] - x['q2_is_first_upper_letter']
    # Digit count
    x['q1_count_symbol_digit'] = df.question1.apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
    x['q2_count_symbol_digit'] = df.question2.apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
    x['diff_count_symbol_digit'] = x['q1_count_symbol_digit'] - x['q2_count_symbol_digit']

    #... YOUR FEATURES HERE ...

    feature_names = list(x.columns.values)
    create_feature_map(feature_names)
    print("Features: {}".format(feature_names))

    x_train = x[:df_train.shape[0]]
    x_test = x[df_train.shape[0]:]
    y_train = df_train['is_duplicate'].values
    del x, df_train, df, train_qs, counts, words

    if False:  # Now we oversample the negative class - on your own risk of overfitting!
        pos_train = x_train[y_train == 1]
        neg_train = x_train[y_train == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / float(len(pos_train) + len(neg_train))))
        p = 0.165
        scale = ((len(pos_train) / float(len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = pd.concat([neg_train, neg_train])
            scale -= 1
        neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        x_train = pd.concat([pos_train, neg_train])
        y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train

    print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
    clr = train_xgb(x_train, y_train, params)

    import pdb
    pdb.set_trace()

    preds = predict_xgb(clr, x_test)

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds
    sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)

    print("Features importances...")
    importance = clr.get_fscore(fmap='xgb.fmap')
    print(importance)
    print(importance.items())
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print(ft)

    ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 30))
    plt.gcf().savefig('features_importance.png')


main()
print("Done.")
