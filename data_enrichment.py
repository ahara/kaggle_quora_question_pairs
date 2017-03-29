__author__ = 'Adam Harasimowicz'

import pandas as pd
from collections import defaultdict


class Concept:
    def __init__(self, similar=None, different=None):
        self.similar = set([]) if similar is None else similar
        self.different = set([]) if different is None else different

    def __add__(self, other):
        """
        Merge two concepts into one
        """
        c = Concept(self.similar.union(other.similar), self.different.union(other.different))

        return c

    def __str__(self):
        return 'similar: %s\t different: %s' % (str(self.similar), str(self.different))

    def add_similar(self, qid):
        self.similar.add(qid)

    def get_similar(self):
        return self.similar

    def add_different(self, qid):
        if isinstance(qid, int):
            self.different.add(qid)
        else:
            for q in qid:
                self.different.add(q)

    def get_different(self):
        return self.different

    def get_conflicts(self):
        return self.similar.intersection(self.different)

    def has_conflicts(self):
        return len(self.get_conflicts()) > 0


class ConceptGraph:
    def __init__(self):
        self.graph = defaultdict(Concept)
        self.questions = dict()

    def build_graph(self, data):
        for i in xrange(data.shape[0]):
            if i % 10000 == 0:
                print '%.3f%%' % ((i + 1.) / data.shape[0] * 100)
            row = data.iloc[i]
            qid1 = row['qid1']
            qid2 = row['qid2']
            # Fill question dictionary
            self.questions[qid1] = row['question1']
            self.questions[qid2] = row['question2']
            # Update concepts
            if row['is_duplicate'] == 0:
                c1 = self.graph[qid1]
                c2 = self.graph[qid2]
                # Add itself
                c1.add_similar(qid1)
                c2.add_similar(qid2)
                # Update question 1
                c1.add_different(qid2)
                c1.add_different(c2.get_similar())
                self._spread_concept(c1)
                # Update question 2
                c2.add_different(qid1)
                c2.add_different(c1.get_similar())
                self._spread_concept(c2)
            else:
                # Merge concepts
                c = self.graph[qid1] + self.graph[qid2]
                c.add_similar(qid1)
                c.add_similar(qid2)
                # Spread updated concept across similar questions
                self._spread_concept(c)
                # Spread similar questions across different
                #print c.get_different(), c.get_similar(), c.has_conflicts()
                for q in c.get_different():
                    if self.graph[q] is not c:
                        self.graph[q].add_different(c.get_similar())

    def _spread_concept(self, c):
        for qid in c.get_similar():
            self.graph[qid] = c

    def __str__(self):
        s = ''
        for k, v in self.graph.items():
            s += '%d = %s\n' % (k, str(v))

        return s

    def get_enriched_data(self):
        enriched_data = []
        for qid1, c in self.graph.items():
            # Duplicated questions
            for qid2 in c.get_similar():
                enriched_data.append((qid1, qid2, self.questions[qid1], self.questions[qid2], 1))
            # Not duplicated questions
            for qid2 in c.get_different():
                enriched_data.append((qid1, qid2, self.questions[qid1], self.questions[qid2], 0))

        return pd.DataFrame.from_records(enriched_data,
                                         columns=['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])


def make_enriched_dataset(data_source):
    data = pd.read_csv(data_source) if isinstance(data_source, str) else data_source
    cg = ConceptGraph()
    cg.build_graph(data)

    return cg.get_enriched_data()


if __name__ == '__main__':
    #data = pd.read_csv('../data/train.csv')
    data = pd.read_csv('../data/concept_graph_unit_test.csv')
    cg = ConceptGraph()
    cg.build_graph(data.iloc[:20])

    print data.iloc[:20]
    print str(cg)
    print cg.get_enriched_data()

    #enriched_data = make_enriched_dataset('../data/train.csv')
    #print enriched_data
    #print enriched_data.shape
