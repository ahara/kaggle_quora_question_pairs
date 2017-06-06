"""
Implementation based on:
https://www.kaggle.com/c/quora-question-pairs/discussion/33371
"""
import networkx as nx
import pandas as pd


def magic_feature_3_with_load():
    train = pd.read_csv('../data/train.csv', encoding='utf-8')
    test = pd.read_csv('../data/test.csv', encoding='utf-8')
    return magic_feature_3(train, test)


def magic_feature_3(train_orig, test_orig):
    cores_dict = pd.read_csv("../data/question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]

    def gen_qid1_max_kcore(x):
        return cores_dict[hash(x)]

    def gen_qid2_max_kcore(x):
        return cores_dict[hash(x)]

    #def gen_max_kcore(row):
    #    return max(row["qid1_max_kcore"], row["qid2_max_kcore"])

    train_orig.loc[:, "m3_qid1_max_kcore"] = train_orig.loc[:, 'question1'].apply(gen_qid1_max_kcore)
    test_orig.loc[:, "m3_qid1_max_kcore"] = test_orig.loc[:, 'question1'].apply(gen_qid1_max_kcore)
    train_orig.loc[:, "m3_qid2_max_kcore"] = train_orig.loc[:, 'question2'].apply(gen_qid2_max_kcore)
    test_orig.loc[:, "m3_qid2_max_kcore"] = test_orig.loc[:, 'question2'].apply(gen_qid2_max_kcore)
    #df_train["max_kcore"] = df_train.apply(gen_max_kcore, axis=1)
    #df_test["max_kcore"] = df_test.apply(gen_max_kcore, axis=1)

    return dict(train=train_orig.loc[:, ['m3_qid1_max_kcore', 'm3_qid2_max_kcore']],
                test=test_orig.loc[:, ['m3_qid1_max_kcore', 'm3_qid2_max_kcore']])


def create_qid_dict(train_orig):
    df_id1 = train_orig.loc[:, ["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
    df_id2 = train_orig.loc[:, ["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

    df_id1.columns = ["qid", "question"]
    df_id2.columns = ["qid", "question"]

    print(df_id1.shape, df_id2.shape)

    df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
    print(df_id1.shape, df_id2.shape, df_id.shape)

    dict_questions = df_id.set_index('question').to_dict()
    return dict_questions["qid"]


def get_id(question, dict_questions):
    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id = len(dict_questions[question]) + 1
        dict_questions[question] = new_id
        return new_id, new_id


def run_kcore():
    df_train = pd.read_csv("../data/train.csv", encoding='utf-8')
    df_test = pd.read_csv("../data/test.csv", encoding='utf-8')
    df_train.loc[:, 'qid1'] = df_train.loc[:, 'question1'].apply(hash)
    df_train.loc[:, 'qid2'] = df_train.loc[:, 'question2'].apply(hash)
    df_test.loc[:, 'qid1'] = df_test.loc[:, 'question1'].apply(hash)
    df_test.loc[:, 'qid2'] = df_test.loc[:, 'question2'].apply(hash)
    df_all = pd.concat([df_train.loc[:, ["qid1", "qid2"]],
                        df_test.loc[:, ["qid1", "qid2"]]], axis=0).reset_index(drop='index')
    print("df_all.shape:", df_all.shape)  # df_all.shape: (2750086, 2)
    df = df_all
    g = nx.Graph()
    g.add_nodes_from(df.qid1)
    edges = list(df.loc[:, ['qid1', 'qid2']].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())

    print(len(set(df.qid1)), g.number_of_nodes())  # 4789604
    print(len(df), g.number_of_edges())  # 2743365 (after self-edges)

    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
    print("df_output.shape:", df_output.shape)

    NB_CORES = 20

    for k in range(2, NB_CORES + 1):
        fieldname = "kcore{}".format(k)
        print("fieldname = ", fieldname)
        ck = nx.k_core(g, k=k).nodes()
        print("len(ck) = ", len(ck))
        df_output[fieldname] = 0
        df_output.ix[df_output.qid.isin(ck), fieldname] = k

    df_output.to_csv("../data/question_kcores.csv", index=None)


def run_kcore_max():
    df_cores = pd.read_csv("../data/question_kcores.csv", index_col="qid")
    df_cores.index.names = ["qid"]
    df_cores.loc[:, 'max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)
    df_cores.loc[:, ['max_kcore']].to_csv("../data/question_max_kcores.csv")  # with index


if __name__ == '__main__':
    run_kcore()
    run_kcore_max()
