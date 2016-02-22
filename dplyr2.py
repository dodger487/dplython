# Chris Riederer
# 2015-02-17

"""Trying to put dplyr-style operations on top of pandas DataFrame."""

import operator
import sys
import types

import numpy as np
import pandas
from pandas import DataFrame


# TODOs:
# * Group, ungroup
# * Summarize
# * Arrange
# * Add more tests
# * Reflection thing in Later -- understand this better
# * make sure to implement reverse methods like "radd" so 1 + X.x will work
# * Should rename some things to be clearer. "df" isn't really a df in the 
    # __radd__ code, for example 
# * sample_n, sample_frac

# Scratch
# https://mtomassoli.wordpress.com/2012/03/18/currying-in-python/
# http://stackoverflow.com/questions/16372229/how-to-catch-any-method-called-on-an-object-in-python
# Code somewhere for making operators like |%|, could use instead of |


class Manager(object):
  def __getattr__(self, attr):
    return Later(attr)


X = Manager()


operator_hooks = [name for name in dir(operator) if name.startswith('__') and 
                  name.endswith('__')]


def instrument_operator_hooks(cls):
  def add_hook(name):
    operator_func = getattr(operator, name.strip('_'), None)
    existing = getattr(cls, name, None)

    def op_hook(self, *args, **kw):
      print "Hooking into {}".format(name)
      self._function = operator_func
      self._params = (args, kw)
      print args
      # if existing is not None:
      #   return existing(self, *args, **kw)
      # TODO: multiple arguments...
      if len(args) > 0 and type(args[0]) == Later:
        self.todo.append(lambda df: getattr(df, name)(args[0].applyFcns(self.origDf)))
      else:  
        self.todo.append(lambda df: getattr(df, name)(*args, **kw))
      return self

    try:
      setattr(cls, name, op_hook)
    except (AttributeError, TypeError):
      print "Skipping", name
      pass  # skip __name__ and __doc__ and the like

  for hook_name in operator_hooks:
    print "Adding hook to", hook_name
    add_hook(hook_name)
  return cls


@instrument_operator_hooks
class Later(object):
  def __init__(self, name):
    self.name = name
    self.todo = [lambda df: df[self.name]]

  def applyFcns(self, df):
    self.origDf = df
    stmt = df
    for func in self.todo:
      print func
      stmt = func(stmt)
    return stmt
    
  def __getattr__(self, attr):
    self.todo.append(lambda df: getattr(df, attr))
    return self

  def __call__(self, *args, **kwargs):
    self.todo.append(lambda foo: foo.__call__(*args, **kwargs))
    return self

  # TODO: need to implement the other reverse methods
  def __radd__(self, arg):
    if type(arg) == Later:
      self.todo.append(lambda df: df.__add__(arg.applyFcns(self.origDf)))
    else:  
      self.todo.append(lambda df: df.__add__(arg))
    return self


def CreateLaterFunction(fcn, *args, **kwargs):
  laterFcn = Later(fcn.func_name + "_FUNCTION")
  laterFcn.fcn = fcn
  laterFcn.args = args
  laterFcn.kwargs = kwargs
  def apply_function(self, df):
    self.origDf = df
    args = [a.applyFcns(self.origDf) if type(a) == Later else a for a in self.args]
    kwargs = {k: v.applyFcns(self.origDf) if type(v) == Later else v 
        for k, v in self.kwargs.iteritems()}
    print args
    print kwargs
    return self.fcn(*args, **kwargs)
  laterFcn.todo = [lambda df: apply_function(laterFcn, df)]
  return laterFcn
  

def DelayFunction(fcn):
  def DelayedFcnCall(*args, **kwargs):
    # Check to see if any args or kw are Later. If not, return normal fcn.
    checkIfLater = lambda x: type(x) == Later
    if (len(filter(checkIfLater, args)) == 0 and 
        len(filter(checkIfLater, kwargs.values())) == 0):
      return fcn(*args, **kwargs)
    else:
      return CreateLaterFunction(fcn, *args, **kwargs)

  return DelayedFcnCall


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
      print stmt
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
      if type(val) == Later:
        df[key] = val.applyFcns(df)
      else:
        df[key] = val
    return DplyFrame(df)
  return addColumns


def head(n=10):
  return lambda df: df[:n]


@DelayFunction
def PairwiseGreater(series1, series2):
  index = series1.index
  newSeries = pandas.Series([max(s1, s2) for s1, s2 in zip(series1, series2)])
  newSeries.index = index
  return newSeries


# summarize

# arrange

# group_by

# ungroup


# diamonds | select(-X.cut)

# Should I allow: diamonds | head
# or keept it mandatory to diamonds | head()

# How does this work with group?
# sample_n, sample_frac

