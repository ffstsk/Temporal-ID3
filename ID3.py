import pandas as pd
import numpy as np
import operator # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

# TODO: file configurazione
# TODO: file output come weka
# TODO: file output con l'albero
# TODO: gain < k, purity > k', |TS| < k'' (percentuale del dataset originale, prima domanda), l'ordine è da k'', k', k


# stop criteria
# the priority are like the followings:
#   1) stopPOD: percent of the original dataset
#   2) stopPurity: purity degree
#   3) stopGain: gain less than
stopPOD = 0.25
stopPurity = 0.7
stopGain = 0.05

intervalRelations = ['L', 'L_', 'A', 'A_', 'O', 'O_', 'E', 'E_', 'D', 'D_', 'B', 'B_']
#intervalRelations = ['L', 'O']
data = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
data.astype(object)
data.loc[0] = ['C1', {(3, 6)}, {(4, 9)}, (-1, 0)]
data.loc[1] = ['C1', {(2, 5)}, {(3, 6)}, (-1, 0)]
data.loc[2] = ['C2', {(1, 5)}, {(6, 8)}, (-1, 0)]
data.loc[3] = ['C2', {}, {(5, 7)}, (-1, 0)]
#data.loc[4] = ['C3', {(1, 2), (5, 10)}, {(5, 7)}, (-1, 0)]
#data.loc[5] = ['C3', {(3, 8), (6, 9)}, {(1, 2), (4, 7), (5, 8)}, (-1, 0)]

print(data)
print(data.shape)
print(len(data))

# the percentual representing the fraction of the original dataset to use as stop criterium
stopPOD = int(stopPOD * data.shape[0])
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

def classWithMostElements(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0]

classElements = countClassElements(data)
print(classWithMostElements(classElements))

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
    def __init__(self, column=None, operator=None, interval=None):
        self.literal = column
        self.operator = operator
        self.interval = interval

    def match(self, example):
        if self.interval == None:
            currentInterval = example['interval']
        else:
            currentInterval = self.interval
        intervals = example[self.literal]
        for interval in intervals:
            if self.operator.check(currentInterval, interval):
                return True, interval
        # TODO if we omit this control we could have 'None' as returned value, think about it
        return False, None

    def __repr__(self):
        # TODO treat the cases for categorial and numerical attributes
        if isinstance(self.operator, Operator):
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
        existsInterval, interval = question.match(dataset.loc[row])
        if existsInterval:
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
        probability = counts[label] / len(dataset)
        information += - probability * np.log2(probability)
    return round(information, 4)

def temporalInformationGain(dataset, q):
    trueRows, falseRows = partition(dataset, q)
    # if the true rows or the false ones are 0, then it is necessary to return 0 as gain
    # otherwise a 'ZeroDivisionError' will raise
    if len(trueRows) == 0 or len(falseRows) == 0:
        return 0
    ''' print('>>>>> trueRows:', trueRows.shape[0])
    print(trueRows)
    print('>>>>> entropy trueRows:', entropy(trueRows))
    print('>>>>> falseRows:', falseRows.shape[0])
    print(falseRows)
    print('>>>>> entropy falseRows:', entropy(falseRows)) '''

    return round(entropy(dataset) -
                 len(trueRows)/len(dataset) * entropy(trueRows) +
                 len(falseRows)/len(dataset) * entropy(falseRows), 4)

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
            # trueRows, falseRows = partition(dataset, question)
            # print('###################')
            # print('>>> Splitting with <%s>%s' % (relation, column), end='')
            # print('>>> data:')
            # print(dataset)
            gain = temporalInformationGain(dataset, question)
            # print('\t will result a', gain, 'gain')
            # print('>>> data after:')
            # print(dataset)
            # print('###################\n\n\n')
            if gain >= bestGain:
                bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

def findBestInterval(dataset):
    return None

''' bestGain, bestQuestion = findBestSplit(data)
print('best gain:', bestGain)
print('best question:', bestQuestion) '''

class Leaf:
    def __init__(self, dataset):
        self.predictions = countClassElements(dataset)

class Node:
    def __init__(self, leftBranch=None, rightBranch=None, anchored=None, question=None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.anchored = anchored
        self.question = question

def buildTree(dataset, anchored=0):
    # TODO: find best interval for an non-archored node
    interval = None
    if anchored == 0:
        interval = findBestInterval(dataset)

    gain, question = findBestSplit(dataset)

    #if gain == 0 or len(uniqueValues(dataset, 'class')) == 1:
    #    return Leaf(dataset)
    #print('length:', len(dataset))

    # stop criteria
    # 1) if the dimension of the current dataset is less than stopPOD, then stop
    if len(dataset) <= stopPOD:
        return Leaf(dataset)
    else:
        # count the elements for each class
        classElements = countClassElements(dataset)
        # the class with the highest value
        cls = classWithMostElements(classElements)
        # 2) if stopPurity is less than the percent of the highest class, then stop
        if stopPurity <= classElements[cls]/len(dataset):
            return Leaf(dataset)
        else:
            # 3) if gain is less than stopGain, then stop
            if gain <= stopGain:
                return Leaf(dataset)

    trueRows, falseRows = partition(dataset, question)

    leftBranch = buildTree(trueRows, 1)
    rightBranch = buildTree(falseRows, 0)

    return Node(leftBranch, rightBranch, anchored, question)

def printTree(node, spacing=''):
    # base case: we've reached a Leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # print the question at this node
    print(spacing + str(node.question))

    # call this function recursively on the true branch
    print(spacing + '--> True:')
    printTree(node.leftBranch, spacing + "\t")

    # call this function recursively on the false branch
    print(spacing + '--> False:')
    printTree(node.rightBranch, spacing + "\t")

temporalTree = buildTree(data, 0)
printTree(temporalTree)

def classify(instance, node):
    if isinstance(node, Leaf):
        return node.predictions
    matches, interval = node.question.match(instance)
    print(node.question, matches)
    print(instance['interval'], interval)
    # print(instance)
    print('\n\n\n')
    instance['interval'] = interval
    if matches:
        return classify(instance, node.leftBranch)
    else:
        return classify(instance, node.rightBranch)

def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs

test = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
test.astype(object)
test.loc[0] = ['C1', {}, {(3, 7)}, (-1, 0)]
#print(data.iloc[3])

print(printLeaf(classify(test.loc[0], temporalTree)))
