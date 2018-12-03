import pandas as pd
import numpy as np
import operator # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import time

# TODO: file configurazione
# TODO: file output come weka
# TODO: file output con l'albero
# TODO: gain < k, purity > k', |TS| < k'' (percentuale del dataset originale, prima domanda), l'ordine è da k'', k', k
# TODO: model checking on the initial dataset, that is, for each literal, for each X, produce the intervals of <X>p and [X]-p

start_time = time.time()

# stop criteria
# the priority are like the followings:
#   1) stopPOD: percent of the original dataset
#   2) stopPurity: purity degree, percent of the values of the class with highest # of elements
#   3) stopGain: gain stop criterium
stopPOD = 0.001
stopPurity = 1
stopGain = 0.0

intervalRelations = ['=', 'L', 'L_', 'A', 'A_', 'O', 'O_', 'E', 'E_', 'D', 'D_', 'B', 'B_']
#intervalRelations = ['E']
data = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
data.astype(object)
data.loc[0] = ['C1', {(3, 4)}, {(2, 4)}, (0, 1)]
data.loc[1] = ['C1', {(4, 5)}, {(3, 5)}, (0, 1)]
data.loc[2] = ['C2', {(3, 5)}, {(2, 4)}, (0, 1)]
data.loc[3] = ['C2', {}, {(4, 6)}, (0, 1)]
'''data.loc[0] = ['C1', {(0, 1)}, {}, (0, 1)]
data.loc[1] = ['C1', {(1, 2)}, {}, (0, 1)]
data.loc[2] = ['C2', {(0, 1)}, {}, (0, 1)]
data.loc[3] = ['C2', {(5, 6)}, {(4, 6)}, (0, 1)]'''
'''data.loc[4] = ['C2', {(3, 6)}, {(4, 9)}, (0, 1)]
data.loc[5] = ['C2', {(1, 5)}, {(3, 8)}, (0, 1)]
data.loc[6] = ['C2', {(2, 5), (3, 6)}, {(2, 5), (1, 7), (5, 8)}, (0, 1)]
data.loc[7] = ['C3', {(1, 2), (5, 10)}, {(5, 7)}, (0, 1)]
data.loc[8] = ['C2', {(1, 5), (3, 4)}, {(1, 2), (3, 4), (3, 8)}, (0, 1)]
data.loc[9] = ['C2', {(2, 5), (3, 6)}, {(2, 5), (1, 7), (5, 8)}, (0, 1)]
data.loc[10] = ['C1', {(3, 6)}, {(4, 9)}, (0, 1)]
data.loc[11] = ['C2', {(2, 5)}, {(3, 6)}, (0, 1)]
data.loc[12] = ['C1', {(1, 5)}, {(6, 8)}, (0, 1)]
data.loc[13] = ['C1', {}, {(5, 7)}, (0, 1)]'''

N = 3
print(data)
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

# classElements = countClassElements(data)
# print(classWithMostElements(classElements))

class Operator:
    def __init__(self, operator):
        self.operator = operator

    # verifying if 'currentInterval' is in relation 'operator' with 'checkInterval'
    # for instance, if we want to check if [0, 1] R_L [2, 3],
    # then currentInterval = [0, 1], checkInterval = [2, 3], and operator = 'L'
    def check(self, currentInterval, checkInterval):
        # EQ: x = x' and y = y'
        if self.operator == '=':
            if (currentInterval[0] == checkInterval[0] and
                currentInterval[1] == checkInterval[1]):
                return True
        # L: y < x'
        elif self.operator == 'L':
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
    # 'value'   : for categorical or numerical attributes

    def __init__(self, column=None, operator=None):
        self.literal = column
        self.operator = operator

    def match(self, example):
        # the interval is already set
        currentInterval = example['interval']
        # get all intervals from the projection on the literal
        intervals = example[self.literal]
        # for each interval
        for interval in intervals:
            # check the [currX, currX] operator [x, y]
            if self.operator.check(currentInterval, interval):
                return True, interval
        # TODO if we omit this control we could have 'None' as returned value, think about it
        return False, None

    def __repr__(self):
        # TODO treat the cases for categorial and numerical attributes
        if isinstance(self.operator, Operator):
            return "the timeline satisfies <%s>%s ?" % (
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

        # if exists an interval
        if existsInterval:
            # if it is the first row, make a copy of the row data and add it to trueRows
            if trueRows.index.empty:
                observation = dataset.loc[row].copy()
                observation['interval'] = interval
                trueRows.at[0] = observation
            # else, also make a copy, but append it to trueRows
            else:
                observation = dataset.loc[row].copy()
                observation['interval'] = interval
                trueRows.at[trueRows.index[-1] + 1] = observation
        # otherwise
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
    '''print('>>>>> trueRows:', trueRows.shape[0])
    print(trueRows)
    print('>>>>> entropy trueRows:', entropy(trueRows))
    print('>>>>> falseRows:', falseRows.shape[0])
    print(falseRows)
    print('>>>>> entropy falseRows:', entropy(falseRows)) '''

    return round(entropy(dataset) -
                 (len(trueRows)/len(dataset) * entropy(trueRows) +
                 len(falseRows)/len(dataset) * entropy(falseRows)), 4)

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
    bestQuestion = Question('fever', Operator(intervalRelations[0]))
    bestGain = temporalInformationGain(dataset, bestQuestion)

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
            if gain > bestGain:
                bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

''' bestGain, bestQuestion = findBestSplit(data)
print('best gain:', bestGain)
print('best question:', bestQuestion) '''

class Leaf:
    def __init__(self, dataset, pod=None, purity=None, gain=None):
        self.predictions = countClassElements(dataset)
        self.pod = pod
        self.purity = purity
        self.gain = gain

class Node:
    def __init__(self, leftBranch=None, rightBranch=None, gain=0, anchored=None, interval=None, question=None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.gain = gain
        self.anchored = anchored
        self.interval = interval
        self.question = question

def getIntervals(N):
    intervals = list()
    for i in range(N-1):
        for j in range(i, N):
            intervals += [(i, j)]
    return intervals

def AssignReferenceInterval(dataset, interval):
    for row in range(0, dataset.shape[0]):
        dataset.loc[row, 'interval'] = interval

def AnchoredSplit(dataset):
    gain, question = findBestSplit(dataset)             # O(12*|AP|*n*N^2)
    trueRows, falseRows = partition(dataset, question)  # O(n*N^2)
    return trueRows, falseRows, gain, question

def UnAnchoredSplit(dataset):
    #bestG = 0
    bestInterval = (0, 1)
    #print('considering interval:', bestInterval)
    AssignReferenceInterval(dataset, bestInterval)

    # TODO: necessary? if not, question and gain would not be returned
    bestG, bestQuestion = findBestSplit(dataset)

    #print('intervals:', getIntervals(N))
    for interval in getIntervals(N):
        #print('considering interval:', interval)
        AssignReferenceInterval(dataset, interval)
        gain, question = findBestSplit(dataset)

        if gain > bestG:
            bestG, bestQuestion, bestInterval = gain, question, interval

    AssignReferenceInterval(dataset, bestInterval)
    trueRows, falseRows = partition(dataset, bestQuestion)

    return trueRows, falseRows, bestG, bestQuestion, bestInterval


def buildTree(dataset, anchored=0):
    interval = None

    if anchored:
        trueRows, falseRows, gain, question = AnchoredSplit(dataset)
    else:
        trueRows, falseRows, gain, question, interval = UnAnchoredSplit(dataset)

    # stop criteria
    # 1) if the dimension of the current dataset is less than stopPOD, then stop
    if len(dataset) <= stopPOD:
        return Leaf(dataset, pod=len(dataset), purity=None, gain=None)
    else:
        # count the elements for each class
        classElements = countClassElements(dataset)
        # the class with the highest value
        cls = classWithMostElements(classElements)
        # 2) if stopPurity is less than the percent of the highest class, then stop
        if stopPurity <= classElements[cls]/len(dataset):
            return Leaf(dataset, len(dataset), classElements[cls]/len(dataset), gain=None)
        else:
            # 3) if gain is less than stopGain, then stop
            # because of this condition we need wait for the split to have the gain
            # therefore, we need to wait when verifying the stop condition
            if gain <= stopGain:
                return Leaf(dataset, len(dataset), classElements[cls]/len(dataset), gain)


    if anchored:
        leftBranch = buildTree(trueRows, 1)
        rightBranch = buildTree(falseRows, 1)
    else:
        leftBranch = buildTree(trueRows, 1)
        rightBranch = buildTree(falseRows, 0)

    return Node(leftBranch, rightBranch, gain, anchored, interval, question)

def buildTree2(dataset, anchored=0):
    # TODO: find best interval for an non-archored node
    interval = None
    gain, question = findBestSplit(dataset)

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
    return Node(leftBranch, rightBranch, anchored, interval, question, gain)


def printTree(node, spacing=''):
    # base case: we've reached a Leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions, 'pod:', node.pod, 'purity:', node.purity, 'gain:', node.gain)
        return

    # print the question at this node
    if node.interval != None:
        print(spacing + 'on [' + str(node.interval[0]) + ',' + str(node.interval[1]) + '], ' + str(node.question) + ' ' +str(node.gain))
    else:
        print(spacing + str(node.question) + ' ' + str(node.gain))
        
    # call this function recursively on the true branch
    print(spacing + '--> TRUE:')
    printTree(node.leftBranch, spacing + "\t")

    # call this function recursively on the false branch
    print(spacing + '--> FALSE:')
    printTree(node.rightBranch, spacing + "\t")

temporalTree = buildTree(data, 0)
printTree(temporalTree)

def classify(instance, node):
    if isinstance(node, Leaf):
        return node.predictions

    if not node.anchored:
        instance['interval'] = node.interval

    matches, interval = node.question.match(instance)
    print(node.question, matches)
    print('instance interval:', instance['interval'], 'interval:', interval)
    # print(instance)
    if interval != None:
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
test.loc[0] = ['C1', {(1, 3), (2, 5)}, {(3, 7)}, (-1, 0)]
#print(data.iloc[3])

print('\n\ntrying to classify:\n', test, '\n')
print(printLeaf(classify(test.loc[0], temporalTree)))

print('execution time in seconds:', time.time() - start_time)


N = 5

def buildTable(dataset):
    output = np.ndarray(shape=(N, N), dtype=set)
    for i in range(N):
        for j in range(N):
            output[i, j] = set()
    for row in range(len(dataset)):
        for attr in ['fever', 'head']:
            for checkInterval in dataset.loc[row, attr]:
                for currentInterval in getIntervals(N):
                    for op in intervalRelations:
                        operator = Operator(op)
                        if operator.check(currentInterval, checkInterval):
                            if op != '=':
                                output[currentInterval[0], currentInterval[1]].add('<' + op + '>' + attr)
                            else:
                                output[currentInterval[0], currentInterval[1]].add(attr)

    return output


trainingData = pd.DataFrame({'class': None, 'interval': None, 'fever': set(), 'head': set()})
trainingData.astype(object)
trainingData.loc[0] = ['C1', {(0, 4), (3, 4)}, {(0, 1), (2, 4)}, (0, 1)]
#trainingData.loc[1] = ['C1', {(0, 1)}, {(1, 4)}, (0, 1)]

print(trainingData)
'''out = buildTable(trainingData)
#print(out)
for i in range(0, N-1):
    for j in range(i+1, N):
        print('[' + str(i) + ',' + str(j) +']:' , out[i,j])'''

