import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

start_time = time.time()
plt.switch_backend('tkagg')

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# alpha for std dev
alpha = 1
# beta for percent of sub-intervals
beta = 0.7

# get the number of sub-intervals for an interval that has length k
def numberOfSubIntervals(k):
    return ((k * (k + 1)) / 2) - 1

# count those (proper) sub-intervals, already in the hashmap, of [i, j] that verify the property
def checkSubIntervalsVerifyingTheProperty(indexes, hashmap, property):
    i = indexes[0]
    j = indexes[1]
    count = 0
    # x ranges within [i, j)
    for x in range(i, j):
        # y ranges within [i+1, j+1)
        for y in range(i + 1, j + 1):
            # if the interval [x, y] belongs to hashmap[property], then increment the counter
            if (x, y) in hashmap[property]:
                count = count + 1
    return count

# get the intervals of a serie (that is a list) having the standard deviation as second parameter
def getIntervals(serie, std):
    # the length of the serie
    n = len(serie)
    # create a dictionary where the keys are 'incr', 'decr' and 'stab'
    hashmap = {'incr': set(), 'decr': set(), 'stab': set()}
    # k ranges within [1, n)
    for k in range(1, n):
        # i ranges within [0, n-k)
        for i in range(0, n-k):
            # j is i+k, that is, the start point plus the length of the considered intervals
            j = i+k
            # intervals of length equal to 1
            if j-i == 1:
                # increasing
                if serie[j] - serie[i] > 0 and abs(serie[j] - serie[i]) >= alpha * std:
                    hashmap['incr'].add((i, j))
                # decreasing
                elif serie[j] - serie[i] < 0 and abs(serie[j] - serie[i]) >= alpha * std:
                    hashmap['decr'].add((i, j))
                # stable
                else:
                    hashmap['stab'].add((i, j))
            # intervals of length greater than 1
            else:
                # probably increasing
                if serie[j] - serie[i] > 0 and abs(serie[j] - serie[i]) >= alpha * std:
                    count = checkSubIntervalsVerifyingTheProperty((i, j), hashmap, 'incr')
                    # enough witnesses to confirm that is increasing
                    if count / numberOfSubIntervals(k) >= beta:
                        hashmap['incr'].add((i, j))
                    # if there are no intervals labeled with 'incr' then [i, j] is increasing
                    elif count == 0:
                        hashmap['incr'].add((i, j))
                # probably decreasing
                elif serie[j] - serie[i] < 0 and abs(serie[j] - serie[i]) >= alpha * std:
                    count = checkSubIntervalsVerifyingTheProperty((i, j), hashmap, 'decr')
                    # enough witnesses to confirm that is decreasing
                    if count / numberOfSubIntervals(k) >= beta:
                        hashmap['decr'].add((i, j))
                    # if there are no intervals labeled with 'decr' then [i, j] is decreasing
                    elif count == 0:
                        hashmap['decr'].add((i, j))
                # probably stable
                else:
                    count = checkSubIntervalsVerifyingTheProperty((i, j), hashmap, 'stab')
                    # enough witnesses to confirm that is stable
                    if count / numberOfSubIntervals(k) >= beta:
                        hashmap['stab'].add((i, j))
                    # if there are no intervals labeled with 'stab' then [i, j] is stable
                    elif count == 0:
                        hashmap['stab'].add((i, j))

    df = pd.DataFrame({'incr': set(),
                       'decr': set(),
                       'stab': set()})
    df.astype(object)
    df.loc[0] = [hashmap['incr'], hashmap['decr'], hashmap['stab']]
    return df

df = pd.DataFrame({'x1': [], 'y1': [], 'z1': [],
                  'roll1': [], 'pitch1': [], 'yaw1': [],
                  'thumb1': [], 'forefinger1': [], 'middle_finger1': [], 'ring_finger1': [], 'little_finger1': [],

                  'x2': [],'y2': [], 'z2': [], 'roll2': [],
                  'pitch2': [], 'yaw2': [], 'thumb2': [],
                  'forefinger2': [], 'middle_finger2': [], 'ring_finger2': [], 'little_finger2': []}
                  )

# read all the data to gather a collection of it into a data frame to calculate the (mean) standard deviation
for dir in os.listdir('data'):
    # if the last character, converted to int, is in the list [1, 2, ..., 9]
    if int(dir[-1]) in range(1, 10):
        for file in os.listdir('data/' + dir):
            newDf = pd.read_table('data/' + dir + '/' + file,
                          sep='\t',
                          header=None,
                          names=['x1', 'y1', 'z1',
                                 'roll1','pitch1', 'yaw1',
                                 'thumb1', 'forefinger1', 'middle_finger1', 'ring_finger1', 'little_finger1',

                                 'x2', 'y2', 'z2',
                                 'roll2', 'pitch2', 'yaw2',
                                 'thumb2', 'forefinger2', 'middle_finger2', 'ring_finger2', 'little_finger2']
                          )

            df = df.append(newDf, ignore_index=True)

# calculating the standard deviation of all variables (i.e., attributes)
standardDeviation = dict()
for attr in df.columns.values:
    standardDeviation[attr] = df[attr].std()

# defining the data frame named 'data' where we will save all the processed data as we want
data = pd.DataFrame({'class': None,
                     'incr_x1': set(), 'decr_x1': set(), 'stab_x1': set(),
                     'incr_x2': set(), 'decr_x2': set(), 'stab_x2': set(),
                     'incr_y1': set(), 'decr_y1': set(), 'stab_y1': set(),
                     'incr_y2': set(), 'decr_y2': set(), 'stab_y2': set(),
                     'incr_z1': set(), 'decr_z1': set(), 'stab_z1': set(),
                     'incr_z2': set(), 'decr_z2': set(), 'stab_z2': set(),
                     'incr_roll1': set(), 'decr_roll1': set(), 'stab_roll1': set(),
                     'incr_roll2': set(), 'decr_roll2': set(), 'stab_roll2': set(),
                     'incr_pitch1': set(), 'decr_pitch1': set(), 'stab_pitch1': set(),
                     'incr_pitch2': set(), 'decr_pitch2': set(), 'stab_pitch2': set(),
                     'incr_yaw1': set(), 'decr_yaw1': set(), 'stab_yaw1': set(),
                     'incr_yaw2': set(), 'decr_yaw2': set(), 'stab_yaw2': set(),
                     'incr_thumb1': set(), 'decr_thumb1': set(), 'stab_thumb1': set(),
                     'incr_thumb2': set(), 'decr_thumb2': set(), 'stab_thumb2': set(),
                     'incr_forefinger1': set(), 'decr_forefinger1': set(), 'stab_forefinger1': set(),
                     'incr_forefinger2': set(), 'decr_forefinger2': set(), 'stab_forefinger2': set(),
                     'incr_middle_finger1': set(), 'decr_middle_finger1': set(), 'stab_middle_finger1': set(),
                     'incr_middle_finger2': set(), 'decr_middle_finger2': set(), 'stab_middle_finger2': set(),
                     'incr_ring_finger1': set(), 'decr_ring_finger1': set(), 'stab_ring_finger1': set(),
                     'incr_ring_finger2': set(), 'decr_ring_finger2': set(), 'stab_ring_finger2': set(),
                     'incr_little_finger1': set(), 'decr_little_finger1': set(), 'stab_little_finger1': set(),
                     'incr_little_finger2': set(), 'decr_little_finger2': set(), 'stab_little_finger2': set()
                     })
data.astype(object)

idx = -1
# extracting the intervals
for dir in os.listdir('data'):
    # if the last character, converted to int, is in the list [1, 2, ...]
    # the files are named 'tctoddX' where X in {1, ..., 9}
    fileNumber = int(dir[-1])
    if fileNumber in range(1, 10):
        for fileName in os.listdir('data/' + dir):
            label = fileName.split('.')[0][:-2]
            subject = dir[-1]
            sample = fileName.split('.')[0][-1]
            if label == 'all': # and subject == '1' and sample == '1':
                idx += 1
                print('index: ', idx)
                '''data.loc[idx] = [None,
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set(),
                                 set(), set(), set()]'''
                newDf = pd.read_table('data/' + dir + '/' + fileName,
                                       sep='\t',
                                       header=None,
                                       names=['x1', 'y1', 'z1',
                                              'roll1', 'pitch1', 'yaw1',
                                              'thumb1', 'forefinger1', 'middle_finger1', 'ring_finger1', 'little_finger1',

                                              'x2', 'y2', 'z2',
                                              'roll2', 'pitch2', 'yaw2',
                                              'thumb2', 'forefinger2', 'middle_finger2', 'ring_finger2', 'little_finger2']
                                       )

                # for each variable get its intervals and store them accordingly
                for attr in newDf.columns.values:
                #for attr in ['x1', 'y1']:
                    attributeIntervals = getIntervals(newDf[attr], standardDeviation[attr])
                    data.at[idx, 'incr_' + attr] = attributeIntervals.at[0, 'incr']
                    data.at[idx, 'decr_' + attr] = attributeIntervals.at[0, 'decr']
                    data.at[idx, 'stab_' + attr] = attributeIntervals.at[0, 'stab']
                # setting the label to 1 (i.e., the word is 'all')
                data.at[idx, 'class'] = 1
                print("==> %s seconds" % (time.time() - start_time))


# extract intervals from other 27 observations that are different from the word 'all'
i = 0
while i < 27:
    # get a random number from [1, 2, ..., 9] representing the file number
    fileNumber = random.choice(range(1,10))
    # get a random file name called 'fileName' from the 'data/tctodd' + fileNumber directory
    # https://stackoverflow.com/questions/701402/best-way-to-choose-a-random-file-from-a-directory
    fileName = random.choice([x for x in os.listdir('data/tctodd' + str(fileNumber) + '/')
                              if os.path.isfile(os.path.join('data/tctodd' + str(fileNumber) + '/', x))])
    # what is the label?
    label = fileName.split('.')[0][:-2]
    # if it is 'all', just continue with the cycle without increasing i
    # (i.e., we are focusing only on words different from 'all')
    if label == 'all':
        continue
    # if the label is different from 'all', then proceed with the extraction
    else:
        i += 1
        idx += 1
        print('index: ', idx)
        '''data.loc[idx] = [None,
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set(),
                         set(), set(), set()]'''
        newDf = pd.read_table('data/tctodd' + str(fileNumber) + '/' + fileName,
                              sep='\t',
                              header=None,
                              names=['x1', 'y1', 'z1',
                                     'roll1', 'pitch1', 'yaw1',
                                     'thumb1', 'forefinger1', 'middle_finger1', 'ring_finger1', 'little_finger1',

                                     'x2', 'y2', 'z2',
                                     'roll2', 'pitch2', 'yaw2',
                                     'thumb2', 'forefinger2', 'middle_finger2', 'ring_finger2', 'little_finger2']
                              )

        # for each variable get its intervals and store them accordingly
        for attr in newDf.columns.values:
            attributeIntervals = getIntervals(newDf[attr], standardDeviation[attr])
            data.at[idx, 'incr_' + attr] = attributeIntervals.at[0, 'incr']
            data.at[idx, 'decr_' + attr] = attributeIntervals.at[0, 'decr']
            data.at[idx, 'stab_' + attr] = attributeIntervals.at[0, 'stab']
        # setting the label to 0 (i.e., the word is not 'all')
        data.at[idx, 'class'] = 0
        print("==> %s seconds" % (time.time() - start_time))


''' print('incr: ')
c = 0
for elem in data.incr_x1:
    for e in elem:
        print(e)
        for i in e:
            c = c + 1
    print('\n')
print('len incr_x1: ', c)

print('decr: ')
c = 0
for elem in data.decr_x1:
    for e in elem:
        print(e)
        for i in e:
            c = c + 1
    print('\n')
print('len decr_x1: ', c)

print('stab: ')
c = 0
for elem in data.stab_x1:
    for e in elem:
        print(e)
        for i in e:
            c = c + 1
    print('\n')
print('len stab_x1: ', c) '''

''' print('print the entire data: ')
print(data.to_string(index=False))
print('\n\n')

print('print each value for each column:')
for attr in data.columns.values:
    print(data[attr])
print('\n\n') '''

print('print data.info()')
print(data.info())

print(data.stab_x1)

data.to_pickle('dataset a=' + str(alpha) + ' b= ' + str(beta) + '.pkl')
data.to_csv('dataset a=' + str(alpha) + ' b= ' + str(beta) + '.csv')
print("--- %s seconds ---" % (time.time() - start_time))