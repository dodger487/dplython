# Chris Riederer
# 2016-02-17

"""Dplyr-style operations on top of pandas DataFrame."""

from functools import wraps
import itertools
import operator
import sys
import types
import warnings
warnings.simplefilter("once")

import six
from six.moves import range

import numpy as np
import pandas
from pandas import DataFrame


__version__ = "0.0.4"


class Manager(object):
  """Object which helps create a delayed computational unit.

  Typically will be set as a global variable ``X``.
  ``X.foo`` will refer to the ``"foo"`` column of the DataFrame in which it is later
  applied. 

  Manager can be used in two ways: 

  1. attribute notation: ``X.foo``
  2. item notation: ``X["foo"]``

  Attribute notation is preferred but item notation can be used in cases where 
  column names contain characters on which python will choke, such as spaces, 
  periods, and so forth.
  """
  def __getattr__(self, attr):
    return Later(attr)

  def __getitem__(self, key):
    return Later(key)


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
    "__rlshift__",  "__rand__",  "__ror__",  "__rxor__",  # "__rrshift__",
]


def create_reversible_func(func_name):
  def reversible_func(self, arg):
    self._UpdateStrAttr(func_name)
    self._UpdateStrCallArgs([arg], {})
    def use_operator(df):
      if isinstance(arg, Later):
        altered_arg = arg.applyFcns(self.origDf)
      else:
        altered_arg = arg
      return getattr(operator, func_name)(df, altered_arg)

    self.todo.append(use_operator)
    return self
  return reversible_func


def instrument_operator_hooks(cls):
  def add_hook(name):
    def op_hook(self, *args, **kwargs):
      self._UpdateStrAttr(name)
      self._UpdateStrCallArgs(args, kwargs)
      if len(args) > 0 and type(args[0]) == Later:
        self.todo.append(lambda df: getattr(df, name)(args[0].applyFcns(self.origDf)))
      else:  
        self.todo.append(lambda df: getattr(df, name)(*args, **kwargs))
      return self

    try:
      setattr(cls, name, op_hook)
    except (AttributeError, TypeError):
      pass  # skip __name__ and __doc__ and the like

  for hook_name in normal_operators:
    add_hook(hook_name)

  for func_name, rfunc_name in reversible_operators:
    setattr(cls, func_name, create_reversible_func(func_name))

  return cls


def _addQuotes(item):
  return '"' + item + '"' if isinstance(item, str) else item

class Operator(object):

  def __init__(self, symbol, precedence):
    self.symbol = symbol
    self.precedence = precedence

  def format_exp(self, exp):
    if type(exp) == Later and exp._precedence < self.precedence:
      return "({0})".format(str(exp))
    else:
      return str(exp)


class BinaryOperator(Operator):
  
  def format_operation(self, obj, args, kwargs):
    return "{0} {1} {2}".format(self.format_exp(obj),
                                self.symbol,
                                self.format_exp(args[0]))


class ReverseBinaryOperator(Operator):
  
  def format_operation(self, obj, args, kwargs):
    return "{2} {1} {0}".format(self.format_exp(obj),
                                self.symbol,
                                self.format_exp(args[0]))


class UnaryOperator(Operator):

  def format_operation(self, obj, args, kwargs):
    return "{0}{1}".format(self.symbol,
                           self.format_exp(obj))


class FunctionOperator(Operator):

  def __init__(self, symbol):
    super(FunctionOperator, self).__init__(symbol, 15)

  def format_args(self, args, kwargs):
      # We sort here because keyword arguments get arbitrary ordering inside the 
      # function call. Support PEP 0468 to help fix this issue!
      # https://www.python.org/dev/peps/pep-0468/
      kwarg_strs = sorted(["{0}={1}".format(k, _addQuotes(v)) 
                            for k, v in kwargs.items()])
      arg_strs = list(map(str, args))
      full_str = ", ".join(arg_strs + kwarg_strs)
      return full_str

  def format_operation(self, obj, args, kwargs):
    return "{0}({1})".format(self.symbol,
                             self.format_args(args, kwargs))


class MethodOperator(FunctionOperator):

  def format_operation(self, obj, args, kwargs):
    return "{0}.{1}({2})".format(self.format_exp(obj),
                                 self.symbol,
                                 self.format_args(args, kwargs))


class BracketOperator(Operator):

  def format_operation(self, obj, args, kwargs):
    return "{0}[{1}]".format(self.format_exp(obj),
                             self.format_arg(args[0]))

  def format_arg(self, arg):
    if isinstance(arg, slice):
      _str = ""
      (start, stop, step) = (arg.start, arg.stop, arg.step)
      if start is not None:
        _str += str(start)
      _str += ":"
      if stop is not None:
        _str += str(stop)
      if step is not None:
        _str += ":" + str(step)
      return _str
    else:
      return str(arg)


class SliceOperator(BracketOperator):
  """Necessary for compatibility."""

  def format_operation(self, obj, args, kwargs):
    return "{0}[{1}]".format(self.format_exp(obj),
                             self.format_arg(slice(args[0], args[1])))


@instrument_operator_hooks
class Later(object):
  """Object which represents a computation to be carried out later.

  The Later object allows us to save computation that cannot currently be 
  executed. It will later receive a DataFrame as an input, and all computation 
  will be carried out upon this DataFrame object.

  Thus, we can refer to columns of the DataFrame as inputs to functions without 
  having the DataFrame currently available:

  >>> diamonds >> sift(X.carat > 4) >> select(X.carat, X.price)
  Out:
         carat  price
  25998   4.01  15223
  25999   4.01  15223
  27130   4.13  17329
  27415   5.01  18018
  27630   4.50  18531

  The special Later name, ``"_"`` will refer to the entire DataFrame. For example,

  >>> diamonds >> sample_n(6) >> select(X.carat, X.price) >> X._.T
  Out:
           18966    19729   9445   49951    3087    33128
  carat     1.16     1.52     0.9    0.3     0.74    0.31
  price  7803.00  8299.00  4593.0  540.0  3315.00  816.00
  """

  _OPERATORS = {
    # see https://docs.python.org/2/reference/expressions.html#operator-precedence
    "__lt__": BinaryOperator("<", 6),
    "__le__": BinaryOperator("<=", 6),
    "__eq__": BinaryOperator("==", 6),
    "__ne__": BinaryOperator("!=", 6),
    "__ge__": BinaryOperator(">=", 6),
    "__gt__": BinaryOperator(">", 6),
    "__or__": BinaryOperator("|", 7),
    "__xor__": BinaryOperator("^", 8),
    "__and__": BinaryOperator("&", 9),
    "__lshift__": BinaryOperator("<<", 10),
    "__rlshift__": ReverseBinaryOperator("<<", 10),
    "__rshift__": BinaryOperator(">>", 10),
    "__rrshift__": ReverseBinaryOperator(">>", 10),
    "__add__": BinaryOperator("+", 11),
    "__radd__": ReverseBinaryOperator("+", 11),
    "__sub__": BinaryOperator("-", 11),
    "__rsub__": ReverseBinaryOperator("-", 11),
    "__mul__": BinaryOperator("*", 12),
    "__rmul__": ReverseBinaryOperator("*", 12),
    "__div__": BinaryOperator("/", 12),
    "__truediv__": BinaryOperator("/", 12),
    "__rdiv__": ReverseBinaryOperator("/", 12),
    "__floordiv__": BinaryOperator("//", 12),
    "__rfloordiv__": ReverseBinaryOperator("//", 12),
    "__mod__": BinaryOperator("%", 12),
    "__rmod__": ReverseBinaryOperator("%", 12),
    "__neg__": UnaryOperator("-", 13),
    "__pos__": UnaryOperator("+", 13),
    "__invert__": UnaryOperator("~", 13),
    "__pow__": BinaryOperator("**", 14),
    "__rpow__": ReverseBinaryOperator("**", 14),
    "__getitem__": BracketOperator("", 15),
    "__getslice__": SliceOperator("", 15)
  }

  def __init__(self, name):
    self.name = name
    if name == "_":
      self.todo = [lambda df: df]
    else:
      self.todo = [lambda df: df[self.name]]
    self._str = 'X["{0}"]'.format(name)
    self._op = None
    self._precedence = 17
  
  def applyFcns(self, df):
    self.origDf = df
    stmt = df
    for func in self.todo:
      stmt = func(stmt)
    return stmt
    
  def __str__(self):
    return "{0}".format(self._str)

  def __repr__(self):
    return "{0}".format(self._str)

  def __getattr__(self, attr):
    self.todo.append(lambda df: getattr(df, attr))
    self._UpdateStrAttr(attr)
    return self

  def __call__(self, *args, **kwargs):
    self.todo.append(lambda foo: foo.__call__(*args, **kwargs))
    self._UpdateStrCallArgs(args, kwargs)
    return self

  def __rrshift__(self, df):
    otherDf = DplyFrame(df.copy(deep=True))
    return self.applyFcns(otherDf)

  def __nonzero__(self):
    raise ValueError("This python code evaluates if this Later is 'True' or "
                     "'False' immediately, instead of waiting for the values "
                     "to become available. This is ambiguous. Try writing your "
                     "code inside a DelayFunction or use if_else.")

  def _UpdateStrAttr(self, attr):
    self._op = self._OPERATORS.get(attr, MethodOperator(attr))

  def _UpdateStrCallArgs(self, args, kwargs):
    self._str = self._op.format_operation(self, args, kwargs)
    self._precedence = self._op.precedence


def CreateLaterFunction(fcn, *args, **kwargs):
  laterFcn = Later(fcn.__name__)
  laterFcn.fcn = fcn
  laterFcn.args = args
  laterFcn.kwargs = kwargs
  def apply_function(self, df):
    self.origDf = df
    args = [a.applyFcns(self.origDf) if type(a) == Later else a 
        for a in self.args]
    kwargs = {k: v.applyFcns(self.origDf) if type(v) == Later else v 
        for k, v in six.iteritems(self.kwargs)}
    return self.fcn(*args, **kwargs)
  laterFcn.todo = [lambda df: apply_function(laterFcn, df)]
  laterFcn._op = FunctionOperator(fcn.__name__)
  laterFcn._UpdateStrCallArgs(args, kwargs)
  return laterFcn
  

def DelayFunction(fcn):
  def DelayedFcnCall(*args, **kwargs):
    # Check to see if any args or kw are Later. If not, return normal fcn.
    if (len([a for a in args if isinstance(a, Later)]) == 0 and
        len([v for k, v in kwargs.items() if isinstance(v, Later)]) == 0):
      return fcn(*args, **kwargs)
    else:
      return CreateLaterFunction(fcn, *args, **kwargs)

  return DelayedFcnCall


class DplyFrame(DataFrame):
  """A subclass of the pandas DataFrame with methods for function piping.

  This class implements two main features on top of the pandas DataFrame. First,
  dplyr-style groups. In contrast to SQL-style or pandas style groups, rows are 
  not collapsed and replaced with a function value.
  Second, >> is overloaded on the DataFrame so that functions on the right-hand
  side of this equation are called on the object. For example,
  
  >>> df >> select(X.carat)
  
  will call a function (created from the "select" call) on df.

  Currently, these inputs need to be one of the following:

  * A "Later" 
  * The "ungroup" function call
  * A function that returns a pandas DataFrame or DplyFrame.
  """
  _metadata = ["_grouped_on", "_grouped_self"]

  def __init__(self, *args, **kwargs):
    super(DplyFrame, self).__init__(*args, **kwargs)
    self._grouped_on = None
    self._current_group = None
    self._grouped_self = None
    if len(args) == 1 and isinstance(args[0], DplyFrame):
      self._copy_attrs(args[0])

  def _copy_attrs(self, df):
    for attr in self._metadata:
      self.__dict__[attr] = getattr(df, attr, None)

  @property
  def _constructor(self):
    return DplyFrame

  def group_self(self, names):
    self._grouped_on = names
    self._grouped_self = self.groupby(names)

  def ungroup(self):
    self._grouped_on = None
    self._grouped_self = None

  def apply_on_groups(self, delayedFcn):
    outDf = self._grouped_self.apply(delayedFcn)

    # Remove multi-index created from grouping and applying
    for grouped_name in outDf.index.names[:-1]:
      if grouped_name in outDf:
        outDf.reset_index(level=0, drop=True, inplace=True)
      else:
        outDf.reset_index(level=0, inplace=True)

    # Drop all 0 index, created by summarize
    if (outDf.index == 0).all():
      outDf.reset_index(drop=True, inplace=True)

    outDf.group_self(self._grouped_on)
    return outDf

  def __rshift__(self, delayedFcn):

    if type(delayedFcn) == Later:
      return delayedFcn.applyFcns(self)

    if delayedFcn == UngroupDF:
      otherDf = DplyFrame(self.copy(deep=True))
      return delayedFcn(otherDf)

    if self._grouped_self:
      outDf = self.apply_on_groups(delayedFcn)
      return outDf
    else:
      otherDf = DplyFrame(self.copy(deep=True))
      return delayedFcn(otherDf)


def ApplyToDataframe(fcn):
  @wraps(fcn)
  def DplyrFcn(*args, **kwargs):
    data_arg = None
    if len(args) > 0 and isinstance(args[0], pandas.DataFrame):
      # data_arg = args[0].copy(deep=True)
      data_arg = args[0]
      args = args[1:]
    fcn_to_apply = fcn(*args, **kwargs)
    if data_arg is None:
      return fcn_to_apply
    else:
      return data_arg >> fcn_to_apply
  return DplyrFcn


@ApplyToDataframe
def sift(*args):
  """Filters rows of the data that meet input criteria.

  Giving multiple arguments to sift is equivalent to a logical "and".
  
  >>> df >> sift(X.carat > 4, X.cut == "Premium")
  # Out:
  # carat      cut color clarity  depth  table  price      x  ...
  #  4.01  Premium     I      I1   61.0     61  15223  10.14
  #  4.01  Premium     J      I1   62.5     62  15223  10.02
  
  As in pandas, use bitwise logical operators like ``|``, ``&``:
  
  >>> df >> sift((X.carat > 4) | (X.cut == "Ideal")) >> head(2)
  # Out:  carat    cut color clarity  depth ...
  #        0.23  Ideal     E     SI2   61.5     
  #        0.23  Ideal     J     VS1   62.8     
  """
  def f(df):
    # TODO: This function is a candidate for improvement!
    final_filter = pandas.Series([True for t in range(len(df))])
    final_filter.index = df.index
    for arg in args:
      stmt = arg.applyFcns(df)
      final_filter = final_filter & stmt
    if final_filter.dtype != bool:
      raise Exception("Inputs to filter must be boolean")
    return df[final_filter]
  return f


def dfilter(*args, **kwargs):
  warnings.warn("'dfilter' is deprecated. Please use 'sift' instead.",
                DeprecationWarning)
  return sift(*args, **kwargs)


@ApplyToDataframe
def select(*args):
  """Select specific columns from DataFrame. 

  Output will be DplyFrame type. Order of columns will be the same as input into
  select.

  >>> diamonds >> select(X.color, X.carat) >> head(3)
  Out:
    color  carat
  0     E   0.23
  1     E   0.21
  2     E   0.23
  """
  return lambda df: df[[column.name for column in args]]


def _dict_to_possibly_ordered_tuples(dict_):
  order = dict_.pop("__order", None)
  if order:
    ordered_keys = set(order)
    dict_keys = set(dict_)

    missing_order = ordered_keys - dict_keys
    if missing_order:
      raise ValueError(", ".join(missing_order) +
                       " in __order not found in keyword arguments")

    missing_kwargs = dict_keys - ordered_keys
    if missing_kwargs:
      raise ValueError(", ".join(missing_kwargs) + " not found in __order")

    return [(key, dict_[key]) for key in order]
  else:
    return sorted(dict_.items(), key=lambda e: e[0])


@ApplyToDataframe
def mutate(*args, **kwargs):
  """Adds a column to the DataFrame.

  This can use existing columns of the DataFrame as input.

  >>> (diamonds >> 
          mutate(carat_bin=X.carat.round()) >> 
          group_by(X.cut, X.carat_bin) >> 
          summarize(avg_price=X.price.mean()))
  Out:
         avg_price  carat_bin        cut
  0     863.908535          0      Ideal
  1    4213.864948          1      Ideal
  2   12838.984078          2      Ideal
  ...
  27  13466.823529          3       Fair
  28  15842.666667          4       Fair
  29  18018.000000          5       Fair
  """
  def addColumns(df):
    for arg in args:
      if isinstance(arg, Later):
        df[str(arg)] = arg.applyFcns(df)
      else:
        df[str(arg)] = arg

    for key, val in _dict_to_possibly_ordered_tuples(kwargs):
      if type(val) == Later:
        df[key] = val.applyFcns(df)
      else:
        df[key] = val
    return df
  return addColumns


@ApplyToDataframe
def group_by(*args, **kwargs):
  def GroupDF(df):
    group_columns = list(kwargs.keys())
    mutate_columns = kwargs
    for arg in args:
      if len(arg.todo) == 1:
        group_columns.append(arg.name)
      else:
        group_columns.append(str(arg))
        mutate_columns[str(arg)] = arg

    if mutate_columns:
      df = df >> mutate(**mutate_columns)
    df.group_self(group_columns)
    return df
  return GroupDF


@ApplyToDataframe
def summarize(**kwargs):
  def CreateSummarizedDf(df):
    input_dict = {k: val.applyFcns(df) for k, val in six.iteritems(kwargs)}
    if len(input_dict) == 0:
      return DplyFrame({}, index=index)
    if hasattr(df, "_current_group") and df._current_group:
      input_dict.update(df._current_group)
    index = [0]
    return DplyFrame(input_dict, index=index)
  return CreateSummarizedDf


def UngroupDF(df):
  # df._grouped_on = None
  # df._group_dict = None
  df.ungroup()
  return df


@ApplyToDataframe
def ungroup():
  return UngroupDF
  

@ApplyToDataframe
def arrange(*args):
  """Sort DataFrame by the input column arguments.

  >>> diamonds >> sample_n(5) >> arrange(X.price) >> select(X.depth, X.price)
  Out:
         depth  price
  28547   61.0    675
  35132   59.1    889
  42526   61.3   1323
  3468    61.6   3392
  23829   62.0  11903
  """
  names = [column.name for column in args]
  def f(df):
    sortby_df = df >> mutate(*args)
    index = sortby_df.sort_values([str(arg) for arg in args]).index
    return df.loc[index]
  return f


@ApplyToDataframe
def head(*args, **kwargs):
  """Returns first n rows"""
  return lambda df: df.head(*args, **kwargs)


@ApplyToDataframe
def sample_n(n):
  """Randomly sample n rows from the DataFrame"""
  return lambda df: DplyFrame(df.sample(n))


@ApplyToDataframe
def sample_frac(frac):
  """Randomly sample `frac` fraction of the DataFrame"""
  return lambda df: DplyFrame(df.sample(frac=frac))


@ApplyToDataframe
def sample(*args, **kwargs):
  """Convenience method that calls into pandas DataFrame's sample method"""
  return lambda df: df.sample(*args, **kwargs)


@ApplyToDataframe
def nrow():
  return lambda df: len(df)


@ApplyToDataframe
def rename(**kwargs):
  """Rename one or more columns, leaving other columns unchanged

  Example usage:
    diamonds >> rename(new_name=old_name)
  """
  def rename_columns(df):
    column_assignments = {old_name_later.name: new_name
                          for new_name, old_name_later in kwargs.items()}
    return df.rename(columns=column_assignments)
  return rename_columns


@ApplyToDataframe
def transmute(**kwargs):
  """ Similar to `select` but allows mutation in column definitions.

  In : (diamonds >>
          head(3) >>
          transmute(new_price=X.price * 2, x_plus_y=X.x + X.y))
  Out:
        new_price  x_plus_y
    0        652      7.93
    1        652      7.73
    2        654      8.12
  """
  mutate_dateframe_fn = mutate(**dict(kwargs))
  column_names = [name for name, _ in _dict_to_possibly_ordered_tuples(kwargs)]
  return lambda df: mutate_dateframe_fn(df)[column_names]


@DelayFunction
def PairwiseGreater(series1, series2):
  index = series1.index
  newSeries = pandas.Series([max(s1, s2) for s1, s2 in zip(series1, series2)])
  newSeries.index = index
  return newSeries


@DelayFunction
def if_else(bool_series, series_true, series_false):
  index = bool_series.index
  newSeries = pandas.Series([s1 if b else s2 for b, s1, s2
      in zip(bool_series, series_true, series_false)])
  newSeries.index = index
  return newSeries
