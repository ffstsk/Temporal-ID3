import pandas as pd

data = pd.DataFrame({'class': None, 'fever': set(), 'head': set()})
data.astype(object)
data.loc[0] = ['C1', {(3,6)}, {(4,9)}]
data.loc[1] = ['C1', {(2,5)}, {(3,6)}]
data.loc[2] = ['C2', {(1,5)}, {(6,8)}]
data.loc[3] = ['C2', {}, {(5,7)}]

print(data)

# get all the unique values for a column in a dataset
def uniqueValues(dataset, column):
    return set([dataset.iloc[row][column] for row in range(0, dataset.shape[0])])

# print(uniqueValues(data, 'class'))

def countClassElements(dataset):
    counts = dict()
    for row in range(0, dataset.shape[0]):
        label = dataset.iloc[row][0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# print(countClassElements(data))

class Question:
    # 'column'  : propositional letter
    # 'relation': HS interval temporal relation
    # 'interval': considered world (i.e., an interval)
    def __init__(self, column, relation, interval):
        self.column = column
        self.relation = relation
        self.interval = interval

    def match(self, example):
        matched = False
        intervals = example[self.column]
        for interval in intervals:
            if interval[0] > self.interval[0]:
                matched = True
                break
        return matched

    def __repr__(self):
        return "The timeline on inteval [%s, %s] satisfies <%s>%s ?" % (
            str(self.interval[0]), str(self.interval[1]), self.relation, self.column)

q = Question('fever', 'L', (0,1))
print(q)
print(q.match(data.iloc[3]))

class Leaf:
    def __init__(self, dataset):
        self.predictions = countClassElements(dataset)

class Node:
    def __init__(self, leftBranch, rightBranch, anchor, question):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.anchor = anchor
        self.question = question


