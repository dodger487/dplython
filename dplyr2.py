# Chris Riederer
# 2016-02-17

"""Trying to put dplyr-style operations on top of pandas DataFrame."""

import itertools
import operator
import sys
import types

import numpy as np
import pandas
from pandas import DataFrame


# TODOs:
# * Summarize

# * Cast output from DplyFrame methods to be DplyFrame
# * make sure to implement reverse methods like "radd" so 1 + X.x will work
# * Can we deal with cases x + y, where x does not have __add__ but y has __radd__?
# * diamonds | select(-X.cut)
# * Move special function Later code into Later object
# * Add more tests
# * Reflection thing in Later -- understand this better
# * Should rename some things to be clearer. "df" isn't really a df in the 
    # __radd__ code, for example 
# * Decorator to let us pipe the whole DF into functions
# * implement the other reverse methods
# * lint
# * Let users use strings instead of Laters in certain situations
#     e.g. select("cut", "carat")
# * What about implementing Manager as a container as well? This would help
#     with situations where column names have spaces. X["type of horse"]

# Scratch
# https://mtomassoli.wordpress.com/2012/03/18/currying-in-python/
# http://stackoverflow.com/questions/16372229/how-to-catch-any-method-called-on-an-object-in-python
# Sort of define your own operators: http://code.activestate.com/recipes/384122/


class Manager(object):
  def __getattr__(self, attr):
    return Later(attr)

  # def __getitem__(self, key):
  #   return Later(key)


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
    if name == "_":
      self.todo = [lambda df: df]
    else:
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
  laterFcn = Later("_FUNCTION")
  # laterFcn = Later(fcn.func_name + "_FUNCTION")
  laterFcn.fcn = fcn
  laterFcn.args = args
  laterFcn.kwargs = kwargs
  def apply_function(self, df):
    self.origDf = df
    args = [a.applyFcns(self.origDf) if type(a) == Later else a 
        for a in self.args]
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
  def __init__(self, df=None, **kwargs):
    super(DplyFrame, self).__init__(df, **kwargs)
    if hasattr(df, "grouped"):
      self.grouped = df.grouped
    else:
      self.grouped = False
    if hasattr(df, "group_indicies"):
      self.group_indicies = df.group_indicies
    else:
      self.group_indicies = False
    # self.group_indicies = []
    self.groups = []

  # def __getitem__(self, *args, **kwargs):
  #   out = super(DplyFrame, self).__getitem__(*args, **kwargs)
  #   out = DplyFrame(out)
  #   out.grouped = self.grouped
  #   out.group_indicies = self.group_indicies
  #   out.groups = self.groups

  def __or__(self, delayedFcn):
    otherDf = DplyFrame(self.copy(deep=True))
    otherDf.grouped = self.grouped  # TODO: this is bad to copy here!!
    otherDf.group_indicies = self.group_indicies
    print delayedFcn
    print UngroupDF
    print "foobar"

    # TODO: decide if we like this feature
    if type(delayedFcn) == Later:
      return delayedFcn.applyFcns(self)

    if delayedFcn == UngroupDF:
      print "ungrouping..."
      return delayedFcn(otherDf)

    if self.grouped:
      groups = [delayedFcn(otherDf[inds]) for inds in self.group_indicies]
      outDf = DplyFrame(pandas.concat(groups))
      outDf.grouped = True
      outDf.group_indicies = self.group_indicies
      return outDf
    else:
      return DplyFrame(delayedFcn(otherDf))


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
  names = [column.name for column in args]
  return lambda df: df[names]


def mutate(**kwargs):
  def addColumns(df):
    for key, val in kwargs.iteritems():
      if type(val) == Later:
        df[key] = val.applyFcns(df)
      else:
        df[key] = val
    return df
  return addColumns


# TODO: might make sense to change this to pipeable thing
# or use df | X._.head
def head(n=10):
  return lambda df: df[:n]


@DelayFunction
def PairwiseGreater(series1, series2):
  index = series1.index
  newSeries = pandas.Series([max(s1, s2) for s1, s2 in zip(series1, series2)])
  newSeries.index = index
  return newSeries


def CreateGroupIndices(df, names, values):
  final_filter = pandas.Series([True for t in xrange(len(df))])
  final_filter.index = df.index
  for (name, val) in zip(names, values):
    final_filter = final_filter & (df[name] == val)
  return final_filter


def group_by(*args):
  def GroupDF(df):
    names = [arg.name for arg in args]
    values = [set(df[name]) for name in names]  # use dplyr here?
    df.group_indicies = [CreateGroupIndices(df, names, v) for v in 
        itertools.product(*values)]
    df.grouped = True
    return df
  return GroupDF
  # options: 
  # make a list of indices
  # make a list of smaller dataframes


def summarize(**kwargs):
  def CreateSummarizedDf(df):
    input_dict = {k: val.applyFcns(df) for k, val in kwargs.iteritems()}
    # DataFrame weirdly chokes on init if given a dictionary whose keys are not
    # iterable. It needs to be told explicitly the index.
    # if hasattr(input_dict.values()[0], '__iter__'):
    #   index = range(len(input_dict.values()[0]))
    # else:
    index = [0]
    return DplyFrame(input_dict, index=index)
  return CreateSummarizedDf


def UngroupDF(df):
  df.group_indicies = []
  df.grouped = False
  return df


def ungroup():
  return UngroupDF
  

def arrange(*args):
  # TODO: add in descending and ascending
  # TODO(cjr): my current version of Pandas isn't up-to-date, so this doesn't
  # exist yet. How embarrassing!
  # return lambda df: df.sort_values(names)
  names = [column.name for column in args]
  return lambda df: DplyFrame(df.sort(names))


def sample_n(n):
  # return X._.sample(n=n)
  return lambda df: DplyFrame(df.sample(n))


def sample_frac(frac):
  # return X._.sample(frac=frac)
  return lambda df: DplyFrame(df.sample(frac=frac))
