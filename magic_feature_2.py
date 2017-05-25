"""
Implementation:
https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain/notebook
"""
import pandas as pd
from collections import defaultdict


def magic_feature_2(train_orig, test_orig):
    ques = pd.concat([train_orig[['question1', 'question2']], \
        test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')

    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    train_orig['m2_q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
    test_orig['m2_q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

    return dict(train=train_orig[['m2_q1_q2_intersect']], test=test_orig[['m2_q1_q2_intersect']])