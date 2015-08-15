#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dt import *
import time
from multiprocessing import Process, Queue, Pool, Manager
import random
from math import sqrt

def load_data(filename):
  """Read training data from file"""
  f = open(filename, 'r')
  data = []

  f.readline()

  for line in f.readlines():
    arr = map(lambda x: float(x), line.split(','))
    data.append(arr[1:])

  f.close()

  return data

def save_test_result(filename, label):
  """Read test data to file"""
  f = open(filename, 'w')
  f.write('id,label\n')
  
  for x in xrange(len(label)):
    f.write(str(x) + ',' + str(label[x]) + '\n')

  f.close()

def run(data, test, itemNumber, queue, lock):
  newData = []
  items = []
  lenData = len(data)

  count = int(sqrt(itemNumber))

  for x in xrange(lenData):
    newData.append(data[random.randint(0, lenData - 1)])

  #print len(newData)

  while len(set(items)) != count:
    items.append(random.randint(0, itemNumber - 1))

  decisionTree = DecisionTree(newData, items)
  tree = decisionTree.create_tree()

  # print tree

  predictClass = []

  for item in test:
    predictClass.append(decisionTree.predict(item))
  
  #use lock to avoid async
  lock.acquire()
  print predictClass
  queue.put(predictClass)
  lock.release()

  return predictClass

def test():
  f = open('test.txt', 'r')
  data = []
  items = []

  for x in xrange(6238):
    data.append(map(lambda x: float(x), f.readline().split(',')))

  for line in f.readlines():
    items.append(int(line))

  f.close()

  decisionTree = DecisionTree(data, items)
  tree = decisionTree.create_tree()

  print tree

if __name__ == '__main__':
  times = input('Please input training times: ')
  # test()
  data = load_data('train.csv')
  test = load_data('test.csv')

  #multiple process
  manager = Manager()
  queue = manager.Queue()
  lock = manager.Lock()
  processes = Pool(processes = 4)

  start = time.clock()

  for x in xrange(times):
    processes.apply_async(run, args = (data, test, 617, queue, lock))

  processes.close()
  processes.join()

  result = []

  for x in xrange(times):
    result.append(queue.get())

  print '\n\n'
  print time.clock() - start

  #single process
  # result = []

  # for x in xrange(100):
  #   result.append(run(data, test, 617, None, None))

  voteResult = []

  for x in xrange(len(test)):
    voteResult.append(int(DecisionTree.majorityLabels([r[x] for r in result])))

  save_test_result('result.csv', voteResult)