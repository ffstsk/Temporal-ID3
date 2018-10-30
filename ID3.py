import pandas as pd
import numpy as np

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

class Operator:
    def __init__(self, operator):
        self.operator = operator

    # verifying if 'currentInterval' is in relation 'operator' with 'checkInterval'
    # for instance, if we want to check if [0, 1] R_L [2, 3],
    # then currentInterval = [0, 1], checkInterval = [2, 3], and operator = 'L'
    def check(self, currentInterval, checkInterval):
        # L: y < x'
        if self.operator == 'L':
            if currentInterval[1] < checkInterval[0]:
                return True
            else:
                return False
        # L_: y' < x
        elif self.operator == 'L_':
            if checkInterval[1] < currentInterval[0]:
                return True
            else:
                return False
        # A: y = x'
        elif self.operator == 'A':
            if currentInterval[1] == checkInterval[0]:
                return True
            else: return False
        # A_: x = y'
        elif self.operator == 'A_':
            if currentInterval[0] == checkInterval[1]:
                return True
            else:
                return False
        # O: x < x' < y < y'
        elif self.operator == 'O':
            if (currentInterval[0] < checkInterval[0] and
                checkInterval[0] < currentInterval[1] and
                currentInterval[1] < checkInterval[1]):
                return True
            else:
                return False
        # O_: x' < x < y' < y
        elif self.operator == 'O_':
            if (checkInterval[0] < currentInterval[0] and
                currentInterval[0] < checkInterval[1] and
                checkInterval[1] < currentInterval[1]):
                return True
            else:
                return False
        # E: x < x' and y = y'
        elif self.operator == 'E':
            if (currentInterval[0] < checkInterval[0] and
                currentInterval[1] == checkInterval[1]):
                return True
            else:
                return False
        # E_: x' < x and y = y'
        elif self.operator == 'E_':
            if (checkInterval[0] < currentInterval[0] and
                currentInterval[1] == checkInterval[1]):
                return True
            else:
                return False
        # D:  x < x' and y' < x
        elif self.operator == 'D':
            if (currentInterval[0] < checkInterval[0] and
                checkInterval[1] < currentInterval[1]):
                return True
            else:
                return False
        # D_: x' < x and y < y'
        elif self.operator == 'D_':
            if (checkInterval[0] < currentInterval[0] and
                currentInterval[1] < checkInterval[1]):
                return True
            else:
                return False
        # B: x = x' and y' < y
        elif self.operator == 'B':
            if (currentInterval[0] == checkInterval[0] and
                checkInterval[1] < currentInterval[1]):
                return True
            else:
                return False
        # B_: x = x' and y < y'
        elif self.operator == 'B_':
            if (currentInterval[0] == checkInterval[0] and
                currentInterval[1] < checkInterval[1]):
                return True
            else:
                False

    def __repr__(self):
        return str(self.operator)


class Question:
    # 'column'  : propositional letter
    # 'operator': HS interval temporal relation
    # 'interval': considered world (i.e., an interval)
    # 'value'   : for categorical or numerical attributes
    def __init__(self, column=None, operator=None, interval=None, value=None):
        self.column = column
        self.operator = operator
        self.interval = interval
        self.value = value

    def match(self, example):
        intervals = example[self.column]
        for interval in intervals:
            if self.operator.check(self.interval, interval):
                return True
        # TODO if we omit this control we could have 'None' as returned value
        return False

    def __repr__(self):
        # TODO treat the cases for categorial and numerical attributes
        if isinstance(self.interval, tuple) and self.interval != None:
            return "The timeline on interval [%s, %s] satisfies <%s>%s ?" % (
                str(self.interval[0]), str(self.interval[1]), self.operator, self.column)

# op = Operator('B_')
# q = Question('fever', op, (3, 8))
# print(q)
# print(q.match(data.iloc[0]))

def partition(dataset, question):
    trueRows = pd.DataFrame({'class': None, 'fever': set(), 'head': set()})
    falseRows = pd.DataFrame({'class': None, 'fever': set(), 'head': set()})
    trueRows.astype(object)
    trueRows.astype(object)

    for row in range(0, dataset.shape[0]):
        if question.match(dataset.loc[row]):
            if trueRows.index.empty:
                trueRows.at[0] = dataset.loc[row]
            else:
                trueRows.at[trueRows.index[-1] + 1] = dataset.loc[row]
        else:
            if falseRows.index.empty:
                falseRows.at[0] = dataset.loc[row]
            else:
                falseRows.at[falseRows.index[-1] + 1] = dataset.loc[row]
    return trueRows, falseRows

op = Operator('L')
q = Question('fever', op, (-1,0))
print(q)
trueRows, falseRows = partition(data, q)
print('true rows:')
print(trueRows)
print('false rows:')
print(falseRows)
print('\n\n')

def entropy(dataset):
    counts = countClassElements(dataset)
    information = 0
    for label in counts:
        probability = counts[label] / dataset.shape[0]
        information += - probability * np.log2(probability)
    return information

def temporalInformation(dataset, q):
    trueRows, falseRows = partition(dataset, q)
    return trueRows.shape[0]/dataset.shape[0] * entropy(trueRows) + \
           falseRows.shape[0]/dataset.shape[0] * entropy(falseRows)

print(q)
print('entropy of data:', entropy(data))
print('entropy of trueRows:', entropy(trueRows))
print('entropy of falseRows:', entropy(falseRows))
print('temporal information of data:', temporalInformation(data, q))
print('information gain data:', entropy(data) - temporalInformation(data, q))
print('\n')

op = Operator('O')
q2 = Question('head', op, (2, 5))
trueRows2, falseRows2 = partition(trueRows, q2)
print('trueRows2:')
print(trueRows2)
print('falseRows2:')
print(falseRows2)
print(q2)
print('entropy of trueRows:', entropy(trueRows))
print('entropy of trueRows2:', entropy(trueRows2))
print('entropy of falseRows2:', entropy(falseRows2))
print('temporal information of trueRows:', temporalInformation(trueRows, q2))
print('information gain trueRows:', entropy(trueRows) - temporalInformation(trueRows, q2))

class Leaf:
    def __init__(self, dataset):
        self.predictions = countClassElements(dataset)

class Node:
    def __init__(self, leftBranch, rightBranch, anchor, question):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.anchor = anchor
        self.question = question
