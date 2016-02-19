# Chris Riederer
# 2015-02-17

"""Trying to put dplyr-style operations on top of pandas DataFrame."""

import sys
import types

import numpy as np
import pandas
from pandas import DataFrame


# TODOs:
# * "CompoundLater": something like (X.x + X.y) > 5
# * Reflection thing in Later
# * Group, ungroup
# * Mutate 

# Scratch
# https://mtomassoli.wordpress.com/2012/03/18/currying-in-python/


class Manager(object):
  def __getattr__(self, attr):
    return Later(attr)


X = Manager()


class Later(object):
  def __init__(self, name):
    self.name = name
    self.todo = [lambda df: df[self.name]]

    # fcnsToChange = ['__add__', '__sub__']
    # for f in fcnsToChange:
    #   def newThing(self, arg):
    #     print "called"
    #     if type(arg) == Later:
    #       self.todo.append(lambda df: df.__getattr__(f)(arg.applyFcns(self.origDf)))
    #     else:  
    #       self.todo.append(lambda df: df.__getattr__(f)(arg))
    #     return self
    #   newThing = types.MethodType(newThing, self)
    #   print f
    #   self.__setattr__(f, newThing)

  def applyFcns(self, df):
    self.origDf = df
    stmt = df
    for func in self.todo:
      stmt = func(stmt)
    return stmt

  # TODO: use reflection on __fcns__ to set everything up
  # func_name = sys._getframe().f_code.co_name
  #     print func_name
    
  def __add__(self, arg):
    if type(arg) == Later:
      self.todo.append(lambda df: df.__add__(arg.applyFcns(self.origDf)))
    else:  
      self.todo.append(lambda df: df.__add__(arg))
    return self

  # def __sub__(self, arg):
  #   if type(arg) == Later:
  #     self.todo.append(lambda df: df.__sub__(arg.applyFcns(self.origDf)))
  #   else:  
  #     self.todo.append(lambda df: df.__sub__(arg))
  #   return self

  def __eq__(self, arg):
    self.todo.append(lambda df: df.__eq__(arg))
    return self

  def __gt__(self, arg):
    self.todo.append(lambda df: df.__gt__(arg))
    return self

  def __ge__(self, arg):
    self.todo.append(lambda df: df.__ge__(arg))
    return self

  def __lt__(self, arg):
    self.todo.append(lambda df: df.__lt__(arg))
    return self

  def __le__(self, arg):
    self.todo.append(lambda df: df.__le__(arg))
    return self

fcnsToChange = ['__add__', '__sub__']
for f in fcnsToChange:
  def newThing(self, arg):
    if type(arg) == Later:
      self.todo.append(lambda df: df.__getattr__(f)(arg.applyFcns(self.origDf)))
    else:  
      self.todo.append(lambda df: df.__getattr__(f)(arg))
    return self
  setattr(Later, f, newThing)


class DplyFrame(DataFrame):
  def __or__(self, delayedFcn):
    otherDf = self.copy(deep=True)
    return delayedFcn(otherDf)


def dfilter(*args):
  def f(df):
    # TODO: This function is a candidate for improvement!
    final_filter = pandas.Series([True for t in xrange(len(df))])
    final_filter.index = df.index
    for arg in args:
      stmt = arg.applyFcns(df)
      final_filter = final_filter & stmt
    if final_filter.dtype != bool:
      raise Exception("Inputs to filter must be boolean")
    return df[final_filter]
  return f


def select(*args):
  names =[column.name for column in args]
  return lambda df: DplyFrame(df[names])


def mutate(**kwargs):
  def addColumns(df):
    for key, val in kwargs.iteritems():
      df[key] = val.applyFcns(df)
    return DplyFrame(df)
  return addColumns


foo | _filter(X.cut == 'Ideal')


# dsummarize

# darrange

# dgroup_by

# dungroup

# head

# How does this work with group?
# sample_n, sample_frac

