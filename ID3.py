import pandas as pd
import numpy as np

#intervalRelations = ['L', 'L_', 'A', 'A_', 'O', 'O_', 'E', 'E_', 'D', 'D_', 'B', 'B_']
intervalRelations = ['L', 'O']
data = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
data.astype(object)
data.loc[0] = ['C1', {(3,6)}, {(4,9)}, (-1, 0)]
data.loc[1] = ['C1', {(2,5)}, {(3,6)}, (-1, 0)]
data.loc[2] = ['C2', {(1,5)}, {(6,8)}, (-1, 0)]
data.loc[3] = ['C2', {}, {(5,7)}, (-1, 0)]

print(data)
print('\n\n\n')

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
    def __init__(self, column=None, operator=None):
        self.literal = column
        self.operator = operator

    def match(self, currentInterval, example):
        intervals = example[self.literal]
        for interval in intervals:
            if self.operator.check(currentInterval, interval):
                return True, interval
        # TODO if we omit this control we could have 'None' as returned value, think about it
        return False, None

    def __repr__(self):
        # TODO treat the cases for categorial and numerical attributes
        if self.operator != None:
            return "The timeline satisfies <%s>%s ?" % (
                self.operator, self.literal)

# op = Operator('B_')
# q = Question('fever', op, (3, 8))
# print(q)
# print(q.match(data.iloc[0]))

def partition(dataset, question):
    trueRows = pd.DataFrame({'class': None,'fever': set(), 'head': set(), 'interval': None})
    falseRows = pd.DataFrame({'class': None, 'fever': set(), 'head': set(), 'interval': None})
    trueRows.astype(object)
    trueRows.astype(object)

    # for each row in the current fragment of the training set
    for row in range(0, dataset.shape[0]):
        # make the question, if the answer is positive then there exist an interval ('existsInterval' is True)
        # and the interval is 'interval'
        existsInterval, interval = question.match(dataset.iloc[row]['interval'], dataset.loc[row])
        if existsInterval:
            # dataset.loc[row]['interval'] = interval
            if trueRows.index.empty:
                observation = dataset.loc[row].copy()
                observation['interval'] = interval
                trueRows.at[0] = observation
            else:
                observation = dataset.loc[row].copy()
                observation['interval'] = interval
                trueRows.at[trueRows.index[-1] + 1] = observation
        else:
            if falseRows.index.empty:
                falseRows.at[0] = dataset.loc[row]
            else:
                falseRows.at[falseRows.index[-1] + 1] = dataset.loc[row]
    return trueRows, falseRows

'''op = Operator('L')
q = Question('fever', op)
print(q)
trueRows, falseRows = partition(data, q)
print('true rows:', trueRows.shape[0])
print(trueRows)
print('false rows:', falseRows.shape[0])
print(falseRows)
print('\n\n') '''

def entropy(dataset):
    counts = countClassElements(dataset)
    information = 0
    for label in counts:
        probability = counts[label] / dataset.shape[0]
        information += - probability * np.log2(probability)
    return round(information, 4)

def temporalInformation(dataset, q):
    trueRows, falseRows = partition(dataset, q)
    # TODO necessary?
    if len(trueRows) == 0 or len(falseRows) == 0:
        return 0
    #print('>>> trueRows:', trueRows.shape[0])
    #print(trueRows)
    #print('>>> falseRows:', falseRows.shape[0])
    #print(falseRows)
    return round(trueRows.shape[0]/dataset.shape[0] * entropy(trueRows) +
                 falseRows.shape[0]/dataset.shape[0] * entropy(falseRows), 4)

'''op = Operator('L')
q = Question('fever', op)
print(q)
information = entropy(data)
tempInfo = temporalInformation(data, q)
print('\n')
print('entropy of data:', information)
print('temporal information of data:', tempInfo)
print('information gain data:', round(information - tempInfo, 4))
print('\n\n\n')

op = Operator('O')
q2 = Question('head', op)
print(q2)
trueRows = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
trueRows.astype(object)
trueRows.loc[0] = ['C1', {(3,6)}, {(4,9)}, (3, 6)]
trueRows.loc[1] = ['C1', {(2,5)}, {(3,6)}, (2, 5)]
trueRows.loc[2] = ['C2', {(1,5)}, {(6,8)}, (1, 5)]
information = entropy(trueRows)
tempInfo = temporalInformation(trueRows, q2)
print('entropy of trueRows:', information)
print('temporal information of trueRows:', tempInfo)
print('information gain trueRows:', information - tempInfo) '''

def findBestSplit(dataset):
    bestGain = 0
    bestQuestion = None

    for column in ['fever', 'head']:
        for relation in intervalRelations:
            question = Question(column, Operator(relation))
            #trueRows, falseRows = partition(dataset, question)
            '''print('###################')
            print('>>> <%s>%s' % (relation, column))
            print('>>> data before:')
            print(dataset)'''
            gain = temporalInformation(dataset, question)
            '''print('>>> data after:')
            print(dataset)
            print('###################\n\n\n')'''
            if gain >= bestGain:
                bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

''' bestGain, bestQuestion = findBestSplit(data)
print('best gain:', bestGain)
print('best question:', bestQuestion) '''

class Leaf:
    def __init__(self, dataset):
        self.predictions = countClassElements(dataset)

class Node:
    def __init__(self, leftBranch, rightBranch, anchored, question):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.anchored = anchored
        self.question = question

def buildTree(dataset, anchored=0):
    gain, question = findBestSplit(dataset)

    # TODO: better stop criteria
    #if gain == 0:
    if len(uniqueValues(dataset, 'class')) == 1:
        return Leaf(dataset)

    trueRows, falseRows = partition(dataset, question)

    leftBranch = buildTree(trueRows, 1)
    rightBranch = buildTree(falseRows, 0)

    return Node(leftBranch, rightBranch, anchored, question)

def printTree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    printTree(node.leftBranch, spacing + "\t")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    printTree(node.rightBranch, spacing + "\t")

temporalTree = buildTree(data, 0)
printTree(temporalTree)