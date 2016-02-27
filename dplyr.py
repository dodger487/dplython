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
# * make sure to implement reverse methods like "radd" so 1 + X.x will work
# * Can we deal with cases x + y, where x does not have __add__ but y has __radd__?
# * implement the other reverse methods
# * Can use __int__ (etc) with pandas.Series.astype

# * Descending and ascending for arrange
# * diamonds >> select(-X.cut)
# * Move special function Later code into Later object
# * Add more tests
# * Reflection thing in Later -- understand this better
# * Should rename some things to be clearer. "df" isn't really a df in the 
    # __radd__ code, for example 
# * lint
# * Let users use strings instead of Laters in certain situations
#     e.g. select("cut", "carat")
# * What about implementing Manager as a container as well? This would help
#     with situations where column names have spaces. X["type of horse"]

# Scratch
# https://mtomassoli.wordpress.com/2012/03/18/currying-in-python/
# http://stackoverflow.com/questions/16372229/how-to-catch-any-method-called-on-an-object-in-python
# Sort of define your own operators: http://code.activestate.com/recipes/384122/
# http://pandas.pydata.org/pandas-docs/stable/internals.html
# I think it might be possible to override __rrshift__ and possibly leave 
#   the pandas dataframe entirely alone.
# http://www.rafekettler.com/magicmethods.html


class Manager(object):
  def __getattr__(self, attr):
    return Later(attr)

  # def __getitem__(self, key):
  #   return Later(key)


X = Manager()


reversible_operators = [
  ["__add__", "__radd__"],
  ["__sub__", "__rsub__"],
  ["__mul__", "__rmul__"],
  ["__floordiv__", "__rfloordiv__"],
  ["__div__", "__rdiv__"],
  ["__truediv__", "__rtruediv__"],
  ["__mod__", "__rmod__"],
  ["__divmod__", "__rdivmod__"],
  ["__pow__", "__rpow__"],
  ["__lshift__", "__rlshift__"],
  ["__rshift__", "__rrshift__"],
  ["__and__", "__rand__"],
  ["__or__", "__ror__"],
  ["__xor__", "__rxor__"],
]

normal_operators = [
    "__abs__", "__concat__", "__contains__", "__delitem__", "__delslice__",
    "__eq__", "__file__", "__ge__", "__getitem__", "__getslice__", "__gt__", 
    "__iadd__", "__iand__", "__iconcat__", "__idiv__", "__ifloordiv__", 
    "__ilshift__", "__imod__", "__imul__", "__index__", "__inv__", "__invert__",
    "__ior__", "__ipow__", "__irepeat__", "__irshift__", "__isub__", 
    "__itruediv__", "__ixor__", "__le__", "__lt__", "__ne__", "__neg__",
    "__not__", "__package__", "__pos__", "__repeat__", "__setitem__",
    "__setslice__", "__radd__", "__rsub__", "__rmul__", "__rfloordiv__",
    "__rdiv__", "__rtruediv__", "__rmod__", "__rdivmod__", "__rpow__", 
    "__rlshift__", "__rrshift__",  "__rand__",  "__ror__",  "__rxor__", 
]


# operator_hooks = [name for name in dir(operator) if name.startswith('__') and 
#                   name.endswith('__')]
# operator_hooks.remove("__add__")


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

  for hook_name in normal_operators:
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

  # def __add__(self, arg):
  #   print "special add"
  #   func_name = "__add__"
  #   rfunc_name = "__radd__"
  #   def TryReverseIfNoRegular(df):
  #     if func_name in dir(df):
  #       return getattr(df, func_name)(arg)
  #     else:
  #       return getattr(arg, rfunc_name)(df)
  #   def TryReverseIfNoRegularLater(df):
  #     if func_name in dir(df):
  #       return getattr(df, func_name)(arg.applyFcns(self.origDf))
  #     else:
  #       return getattr(arg.applyFcns(self.origDf), rfunc_name)(df)
  #       # return arg.applyFcns(self.origDf).__radd__(df)
    
  #   if type(arg) == Later:
  #     self.todo.append(TryReverseIfNoRegularLater)
  #     # self.todo.append(lambda df: df.__add__(arg.applyFcns(self.origDf)))
  #   else:  
  #     self.todo.append(TryReverseIfNoRegular)
  #     # self.todo.append(lambda df: df.__add__(arg))
  #   return self

  # TODO: need to implement the other reverse methods
  def __radd__(self, arg):
    if type(arg) == Later:
      self.todo.append(lambda df: df.__add__(arg.applyFcns(self.origDf)))
    else:  
      self.todo.append(lambda df: df.__add__(arg))
    return self

  # def __rrshift__(self, df):
  #   otherDf = DplyFrame(df.copy(deep=True))
  #   return self.applyFcns(otherDf)

def create_reversible_func(func_name, rfunc_name):
  def special_add(self, arg):
    def TryReverseIfNoRegular(df):
      if func_name in dir(df):
        return getattr(df, func_name)(arg)
      else:
        return getattr(arg, rfunc_name)(df)

    def TryReverseIfNoRegularLater(df):
      if func_name in dir(df):
        return getattr(df, func_name)(arg.applyFcns(self.origDf))
      else:
        return getattr(arg.applyFcns(self.origDf), rfunc_name)(df)

    if type(arg) == Later:
      self.todo.append(TryReverseIfNoRegularLater)
    else:  
      self.todo.append(TryReverseIfNoRegular)
    return self
  return special_add

for func_name, rfunc_name in reversible_operators:
  setattr(Later, func_name, create_reversible_func(func_name, rfunc_name))

# for hook_name in reversible_operators:
#   print "Adding hook to", hook_name
#     def TryRaddIfNoAdd(df):
#       if "__add__" in dir(df):
#         return df.__add__(arg)
#       else:
#         return arg.__radd__(df)
#     def TryRaddIfNoAddLater(df):
#       if "__add__" in dir(df):
#         return df.__add__(arg.applyFcns(self.origDf))
#       else:
#         return arg.applyFcns(self.origDf).__radd__(df)
    
#     if type(arg) == Later:
#       self.todo.append(TryRaddIfNoAddLater)
#       # self.todo.append(lambda df: df.__add__(arg.applyFcns(self.origDf)))
#     else:  
#       self.todo.append(TryRaddIfNoAdd)

#   add_hook(hook_name)

#       if len(args) > 0 and type(args[0]) == Later:
#         self.todo.append(lambda df: getattr(df, name)(args[0].applyFcns(self.origDf)))
#       else:  
#         self.todo.append(lambda df: getattr(df, name)(*args, **kw))
#       return self

#     try:
#       setattr(cls, name, op_hook)
#     except (AttributeError, TypeError):
#       print "Skipping", name
#       pass  # skip __name__ and __doc__ and the like





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
  

class DplyFrame(DataFrame):
  _metadata = ["_grouped_on", "_group_dict"]

  def __init__(self, *args, **kwargs):
    super(DplyFrame, self).__init__(*args, **kwargs)
    self._grouped_on = None
    self._group_dict = None
    self._current_group = None
    if len(args) == 1 and isinstance(args[0], DplyFrame):
      self._copy_attrs(args[0])

  def _copy_attrs(self, df):
    for attr in self._metadata:
      self.__dict__[attr] = getattr(df, attr, None)

  @property
  def _constructor(self):
    return DplyFrame

  def CreateGroupIndices(self, names, values):
    final_filter = pandas.Series([True for t in xrange(len(self))])
    final_filter.index = self.index
    for (name, val) in zip(names, values):
      final_filter = final_filter & (self[name] == val)
    return final_filter

  def group_self(self, names):
    self._grouped_on = names
    values = [set(self[name]) for name in names]  # use dplyr here?
    self._group_dict = {v: self.CreateGroupIndices(names, v) for v in 
        itertools.product(*values)}

  def apply_on_groups(self, delayedFcn, otherDf):
    self.group_self(self._grouped_on)  # TODO: think about removing
    groups = []
    for group_vals, group_inds in self._group_dict.iteritems():
      subsetDf = otherDf[group_inds]
      if len(subsetDf) > 0:
        subsetDf._current_group = dict(zip(self._grouped_on, group_vals))
        groups.append(delayedFcn(subsetDf))

    outDf = DplyFrame(pandas.concat(groups))
    outDf.index = range(len(outDf))
    return outDf

  def __rshift__(self, delayedFcn):
    otherDf = DplyFrame(self.copy(deep=True))

    if type(delayedFcn) == Later:
      return delayedFcn.applyFcns(self)

    if delayedFcn == UngroupDF:
      return delayedFcn(otherDf)

    if self._group_dict:
      outDf = self.apply_on_groups(delayedFcn, otherDf)
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


def group_by(*args):
  def GroupDF(df):
    df.group_self([arg.name for arg in args])
    return df
  return GroupDF


def summarize(**kwargs):
  def CreateSummarizedDf(df):
    input_dict = {k: val.applyFcns(df) for k, val in kwargs.iteritems()}
    if len(input_dict) == 0:
      return DplyFrame({}, index=index)
    if hasattr(df, '_current_group') and df._current_group:
      input_dict.update(df._current_group)
    index = [0]
    return DplyFrame(input_dict, index=index)
  return CreateSummarizedDf


def UngroupDF(df):
  df._grouped_on = None
  df._group_dict = None
  return df


def ungroup():
  return UngroupDF
  

def arrange(*args):
  # TODO: add in descending and ascending
  names = [column.name for column in args]
  return lambda df: DplyFrame(df.sort(names))


# TODO: might make sense to change this to pipeable thing
# or use df >> X._.head
def head(n=10):
  return lambda df: df[:n]


def sample_n(n):
  # return X._.sample(n=n)
  return lambda df: DplyFrame(df.sample(n))


def sample_frac(frac):
  # return X._.sample(frac=frac)
  return lambda df: DplyFrame(df.sample(frac=frac))


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


@DelayFunction
def PairwiseGreater(series1, series2):
  index = series1.index
  newSeries = pandas.Series([max(s1, s2) for s1, s2 in zip(series1, series2)])
  newSeries.index = index
  return newSeries


nrow = X._.__len__