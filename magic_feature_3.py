"""
Implementation:

"""
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


def magic_feature_3_with_load():
    train = pd.read_csv('../data/train.csv', encoding='utf-8')
    train_idx, valid_idx = train_test_split(range(train.shape[0]), test_size=0.2, random_state=4242)
    valid = train.iloc[valid_idx]
    train = train.iloc[train_idx]
    test = pd.read_csv('../data/test.csv', encoding='utf-8')
    return magic_feature_3(train, valid, test)


def magic_feature_3(train_orig, valid_orig, test_orig):
    q_dict = defaultdict(set)
    for i in range(train_orig.shape[0]):
        if train_orig.is_duplicate.iloc[i] == 1:
            q_dict[train_orig.question1.iloc[i]].add(train_orig.question2.iloc[i])
            q_dict[train_orig.question2.iloc[i]].add(train_orig.question1.iloc[i])

    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    train_orig['m3_q1_q2_pos_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
    valid_orig['m3_q1_q2_pos_intersect'] = valid_orig.apply(q1_q2_intersect, axis=1, raw=True)
    test_orig['m3_q1_q2_pos_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

    return dict(train=train_orig[['m3_q1_q2_pos_intersect']],
                valid=valid_orig[['m3_q1_q2_pos_intersect']],
                test=test_orig[['m3_q1_q2_pos_intersect']])