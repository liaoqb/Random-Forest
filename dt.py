#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import operator
import random

#use C4.5
class DecisionTree(object):
  def __init__(self, dataSet, items):
    """
      dataSet = [item1, item2, ..., label]
    """
    self.items = items
    self.dataSet = []

    for data in dataSet:
      self.dataSet.append(self.__filter_data(data))
      self.dataSet[-1].append(data[-1])

    self.tree = {}

  def __calculate_entropy(self, dataSet):
    #print '__calculate_entropy'
    numEntries = len(dataSet)
    labelCounts = {}

    for data in dataSet:
      currentLabel = data[-1]

      if currentLabel not in labelCounts.keys():
        labelCounts[currentLabel] = 0

      labelCounts[currentLabel] += 1

    entropy = 0.0

    for key in labelCounts:
      prob = labelCounts[key] / float(numEntries)
      entropy -= prob * log(prob, 2)

    return entropy

  def __split_dataset(self, dataSet, axis, threshold):
    #print '__split_dataset'
    small = []
    large = []

    for data in dataSet:
      reducedData = data[: axis]
      reducedData.extend(data[axis + 1:])

      if data[axis] <= threshold:
        small.append(reducedData)

      else:
        large.append(reducedData)

    return small, large

  def __choose_best_to_split(self, dataSet):
    #print '__choose_best_to_split'
    baseEntropy = self.__calculate_entropy(dataSet)
    bestGainRatio = 0.0
    bestFeature = -1
    bestSplit = 0.0
    bestSmall = []
    bestLarge = []

    lenDataset = len(dataSet)

    #print 'lenDataSet', len(dataSet[0])

    for i in xrange(len(dataSet[0]) - 1):
      featureList = sorted(list(set([item[i] for item in dataSet])))
      
      #delete i
      if len(featureList) == 1:
        return i, featureList[0], [], []

      delta = (featureList[-1] - featureList[0]) / 5.0

      thresholds = [delta * x + featureList[0] for x in xrange(1, 5)]

      for threshold in thresholds:
        #print thresholds
        small, large = self.__split_dataset(dataSet, i, threshold)

        probSmall = len(small) / float(lenDataset)
        probLarge = len(large) / float(lenDataset)

        gain = baseEntropy - (probSmall * self.__calculate_entropy(small) + \
        probLarge * self.__calculate_entropy(large))
        splitInfomation = -probSmall * log(probSmall, 2) - probLarge * log(probLarge, 2)

        gainRatio = gain / splitInfomation

        if gainRatio > bestGainRatio:
          bestGainRatio = gainRatio
          bestFeature = i
          bestSplit = threshold
          bestSmall = small
          bestLarge = large

      #print 'yes'

    return bestFeature, bestSplit, bestSmall, bestLarge
  
  @staticmethod
  def majorityLabels(dataSet):
    classCount = {}

    for vote in dataSet:
      if vote not in classCount.keys():
        classCount[vote] = 0

      classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]

  def __recursion_tree(self, dataSet, arr):
    #print '__recursion_tree'
    classList = [data[-1] for data in dataSet]

    if len(classList) == 0:
      return random.randint(1, 26)

    if classList.count(classList[0]) == len(classList):
      #print classList[0]
      return classList[0]

    if len(dataSet[0]) == 1:
      #print self.majorityLabels(classList)
      return self.majorityLabels(classList)

    bestFeature, threshold, small, large = self.__choose_best_to_split(dataSet)

    if len(small) == 0 or len(large) == 0:
      if bestFeature == -1:
        bestFeature = 0
      #print dataSet[0][:bestFeature] + dataSet[0][bestFeature + 1:]
      return self.__recursion_tree(map(lambda x: x[:bestFeature] + x[bestFeature + 1:], dataSet),\
        arr[:bestFeature] + arr[bestFeature + 1:])

    key = str(arr[bestFeature]) + ':' + str(threshold)
    del arr[bestFeature]
    myTree = {key: {}}

    myTree[key]['small'] = self.__recursion_tree(small, arr[:])
    myTree[key]['large'] = self.__recursion_tree(large, arr[:])

    return myTree

  def create_tree(self):
    self.tree = self.__recursion_tree(self.dataSet, [i for i in xrange(len(self.items))])

    return self.tree

  def __classify_data(self, data, tree):
    key = tree.keys()[0]
    value = tree[key]
    index = int(key.split(':')[0])
    threshold = float(key.split(':')[1])

    result = data[index]

    if result <= threshold:
      small = value['small']

      if type(small).__name__ == 'dict':
        return self.__classify_data(data, small)

      else:
        return value['small']

    else:
      large = value['large']

      if type(large).__name__ == 'dict':
        return self.__classify_data(data, large)

      else:
        return value['large']

  def __filter_data(self, data):
    filterData = []

    for item in self.items:
      filterData.append(data[item])

    return filterData

  def predict(self, data):
    return self.__classify_data(self.__filter_data(data), self.tree)