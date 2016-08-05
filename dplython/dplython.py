# Chris Riederer
# 2016-02-17

"""Dplyr-style operations on top of pandas DataFrame."""

from functools import wraps
import itertools
import sys
import types
import warnings
warnings.simplefilter("once")

import six
from six.moves import range

import numpy as np
import pandas
from pandas import DataFrame

try:
  from .later import (Later, CreateLaterFunction, DelayFunction, X, Manager)
except:
  from later import (Later, CreateLaterFunction, DelayFunction, X, Manager)


__version__ = "0.0.7"


def _addQuotes(item):
  return '"' + item + '"' if isinstance(item, str) else item


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

  def regroup(self, names):
    self.group_self(names)
    return self

  def apply_on_groups(self, delayedFcn):
    handled_classes = (mutate, sift, inner_join, full_join, left_join, 
                       right_join, semi_join, anti_join, summarize)
    if isinstance(delayedFcn, handled_classes):
      return delayedFcn(self)

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
      return delayedFcn.evaluate(self)

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
      if 'join' not in fcn.__name__:
        args = args[1:]
    fcn_to_apply = fcn(*args, **kwargs)
    if data_arg is None:
      return fcn_to_apply
    else:
      return data_arg >> fcn_to_apply
  return DplyrFcn


class Verb(object):

  def __new__(cls, *args, **kwargs):
    if len(args) > 0 and isinstance(args[0], pandas.DataFrame):
      verb = cls(*args[1:], **kwargs)
      return verb(args[0].copy(deep=True))
    else:
      return super(Verb, cls).__new__(cls)

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def do(self):
    raise NotImplementedError()


class sift(Verb):
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

  __name__ = "sift"

  def __call__(self, df):
    # TODO: This function is a candidate for improvement!
    final_filter = pandas.Series([True for t in range(len(df))])
    final_filter.index = df.index
    grouped = "transform" if df._grouped_on else None
    for arg in self.args:
      stmt = arg.evaluate(df, special=grouped)
      final_filter = final_filter & stmt
    if final_filter.dtype != bool:
      raise Exception("Inputs to filter must be boolean")
    df = df[final_filter]
    if grouped:
      df.group_self(df._grouped_on)
    return df

  def __rrshift__(self, other):
    return self.__call__(other)



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

  Grouping variables are implied in selection.
  >>> df >> group_by(X.a, X.b) >> select(X.c)
  returns a dataframe like `df[[X.a, X.b, X.c]]` with the variables appearing in
  grouped order before the selected column(s), unless a grouped variable is
  explicitly selected

  >>> df >> group_by(X.a, X.b) >> select(X.c, X.b)
  returns a dataframe like `df[[X.a, X.c, X.b]]`
  """
  def select_columns(df, args):
    columns = [column._name for column in args]
    if df._grouped_on:
      for col in df._grouped_on[::-1]:
        if col not in columns:
          columns.insert(0, col)
    return columns
  return lambda df: df[select_columns(df, args)]


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


class mutate(Verb):
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

  __name__ = "mutate"

  def __call__(self, df):
    for arg in self.args:
      if isinstance(arg, Later):
        df[str(arg)] = arg.evaluate(df, special="transform")
      else:
        df[str(arg)] = arg

    for key, val in _dict_to_possibly_ordered_tuples(self.kwargs):
      if isinstance(val, Later):
        df[key] = val.evaluate(df, special="transform")
      else:
        df[key] = val
    return df

  def __rrshift__(self, other):
    return self.__call__(DplyFrame(other.copy(deep=True)))


@ApplyToDataframe
def group_by(*args, **kwargs):
  def GroupDF(df):
    group_columns = list(kwargs.keys())
    mutate_columns = kwargs
    for arg in args:
      if arg._name is not None:
        group_columns.append(arg._name)
      else:
        group_columns.append(str(arg))
        mutate_columns[str(arg)] = arg

    if mutate_columns:
      df = df >> mutate(**mutate_columns)
    df.group_self(group_columns)
    return df
  return GroupDF


class summarize(Verb):
  """Summarizes a dataset via functions
  >>>(diamonds >>
  ...        group_by(X.cut, X.carat_bin) >>
  ...        summarize(avg_price=X.price.mean()))
  If the dataset has grouping, summarizing will be done for each group,
  otherwise, will return a single row
  """

  def __call__(self, df):
    def summarize(df):
      if df._grouped_on:
        input_dict = {k: val.evaluate(df, special="agg")
            for k, val in six.iteritems(self.kwargs)}
        return DplyFrame(input_dict).reset_index()
      else:
        input_dict = {k: val.evaluate(df) for k, val
                      in six.iteritems(self.kwargs)}

      if len(input_dict) == 0:
        return DplyFrame({}, index=index)
      if hasattr(df, "_current_group") and df._current_group:
        input_dict.update(df._current_group)
      index = [0]
      return DplyFrame(input_dict, index=index)

    outDf = summarize(df)

    # Remove multi-index created from grouping and applying
    for grouped_name in outDf.index.names[:-1]:
      if grouped_name in outDf:
        outDf.reset_index(level=0, drop=True, inplace=True)
      else:
        outDf.reset_index(level=0, inplace=True)

    # Drop all 0 index, created by summarize
    if (outDf.index == 0).all():
      outDf.reset_index(drop=True, inplace=True)
    return outDf >> ungroup()


@ApplyToDataframe
def count(*args, **kwargs):
  def CreateCountDf(df):
    col_name = df.columns[0]
    return (df >> group_by(*args, **kwargs) >>
                  summarize(n=X[col_name].__len__()) >> ungroup())
  return CreateCountDf
  

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
  names = [column._name for column in args]
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
def tail(*args, **kwargs):
  """Returns last n rows"""
  return lambda df: df.tail(*args, **kwargs)


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
    column_assignments = {old_name_later._name: new_name
                          for new_name, old_name_later in kwargs.items()}
    return df.rename(columns=column_assignments)
  return rename_columns


@ApplyToDataframe
def transmute(*args, **kwargs):
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
  mutate_dateframe_fn = mutate(*args, **dict(kwargs))
  column_names_args = [str(arg) for arg in args]
  column_names_kwargs = [name for name, _
                         in _dict_to_possibly_ordered_tuples(kwargs)]
  column_names = column_names_args + column_names_kwargs
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


def get_join_cols(by_entry):
  """ helper function used for joins
  builds left and right join list for join function
  """
  left_cols = []
  right_cols = []
  for col in by_entry:
    if isinstance(col, str):
      left_cols.append(col)
      right_cols.append(col)
    else:
      left_cols.append(col[0])
      right_cols.append(col[1])
  return left_cols, right_cols


def mutating_join(*args, **kwargs):
  """ generic function for mutating dplyr-style joins
  """
  # candidate for improvement
  left = args[0]
  right = args[1]
  if 'by' in kwargs:
    left_cols, right_cols = get_join_cols(kwargs['by'])
  else:
    left_cols, right_cols = None, None
  if 'suffixes' in kwargs:
    dsuffixes = kwargs['suffixes']
  else:
    dsuffixes = ('_x', '_y')
  if left._grouped_on:
    outDf = (DplyFrame((left >> ungroup())
                       .merge(right, how=kwargs['how'], left_on=left_cols, 
                              right_on=right_cols, suffixes=dsuffixes))
             .regroup(left._grouped_on))
  else:
    outDf = DplyFrame(left.merge(right, how=kwargs['how'], left_on=left_cols, 
                                 right_on=right_cols, suffixes=dsuffixes))
  return outDf


class Join(Verb):
  """ Generic class for two-table verbs
  """

  def __new__(cls, *args, **kwargs):
    if (len(args) > 1 and 
        isinstance(args[0], pandas.DataFrame) and 
        isinstance(args[1], pandas.DataFrame)):
      verb = cls(*args[1:], **kwargs)
      return verb(args[0].copy(deep=True))
    else:
      return super(Verb, cls).__new__(cls)

  def __rrshift__(self, other):
      return self.__call__(other)


class inner_join(Join):
  """ Perform sql style inner join
  >>> left_data >> inner_join(right_data[
  ...                            , by=[join_columns_in_list_as_single_or_tuple][
  ...                            , suffixes=('_x', _y)]])
  e.g. flights2 >> inner_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe preserving any grouping from left dataframe

  Select all rows from both tables where there are matches on specified columns.

  The by argument takes a list of columns. For a list like ['A', 'B'], it 
  assumes 'A' and 'B' are columns in both dataframes.

  For a list like [('A', 'B')], it assumes column 'A' in the left dataframe is 
  the same as column 'B' in the right dataframe.

  Can mix and match (e.g. by=['A', ('B', 'C')] will assume both dataframes have 
  column 'A', and column 'B' in the left dataframe is the same as column 'C' 
  in the right dataframe.

  If by is not specified, then all shared columns will be assumed to be the 
  join columns.

  suffixes will be used to rename columns that are common to both dataframes,
  but not used in the join operation.
  e.g. `suffixes=('_1', '_2')`

  If suffixes is not included, then the pandas default will be used ('_x', '_y')
  """
  __name__ = 'inner_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'inner'})
    return mutating_join(df, self.args[0], **self.kwargs)


class full_join(Join):
  """ Perform sql style outer/full join
  >>> left_data >> full_join(right_data[
  ...                            , by=[join_columns_in_list_as_single_or_tuple][
  ...                            , suffixes=('_x', _y)]])
  e.g. flights2 >> full_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe preserving any grouping from left dataframe

  Select all rows from both tables, matching when possible, filling in 
  missing values where data doesn't match.

  The by argument takes a list of columns. For a list like ['A', 'B'], it 
  assumes 'A' and 'B' are columns in both dataframes.
  
  For a list like [('A', 'B')], it assumes column 'A' in the left dataframe 
  is the same as column 'B' in the right dataframe.

  Can mix and match (e.g. by=['A', ('B', 'C')] will assume both dataframes 
  have  column 'A', and column 'B' in the left dataframe is the same as column 
  'C' in the right dataframe. If by is not specified, then all shared columns 
  will be assumed to be the join columns.

  suffixes will be used to rename columns that are common to both dataframes,
  but not used in the join operation.
  e.g. `suffixes=('_1', '_2')`.
  If suffixes is not included, then the pandas default will be used ('_x', '_y')
  """

  __name__ = 'full_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'outer'})
    return mutating_join(df, self.args[0], **self.kwargs)


class left_join(Join):
  """ Perform sql style left join
  >>> left_data >> left_join(right_data[
  ...                            , by=[join_columns_in_list_as_single_or_tuple][
  ...                            , suffixes=('_x', _y)]])
  e.g. flights2 >> full_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe preserving any grouping from left dataframe

  Select all rows from the left table, and corresponding rows from the right 
  table where values match, filling in missing values where data doesn't match.
  

  The by argument takes a list of columns. For a list like ['A', 'B'], it 
  assumes 'A' and 'B' are columns in both dataframes. For a list like [('A', 
  'B')], it assumes column 'A' in the left dataframe is the same as column 'B' 
  in the right dataframe. Can mix and match (e.g. by=['A', ('B', 'C')] will 
  assume both dataframes have column 'A', and column 'B' in the left dataframe 
  is the same as column 'C' in the right dataframe. If by is not specified, then
  all shared columns will be assumed to be the join columns.

  suffixes will be used to rename columns that are common to both dataframes,
  but not used in the join operation.
  e.g. `suffixes=('_1', '_2')`.
  If suffixes is not included, then the pandas default will be used ('_x', '_y')
  """

  __name__ = 'left_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'left'})
    return mutating_join(df, self.args[0], **self.kwargs)


class right_join(Join):
  """ Perform sql style right join
  >>> left_data >> right_join(right_data[
  ...                            , by=[join_columns_in_list_as_single_or_tuple][
  ...                            , suffixes=('_x', _y)]])
  e.g. flights2 >> right_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe preserving any grouping from left dataframe

  Select all rows from the right table, and corresponding rows from the left 
  table where values match, filling in missing values where data doesn't match.
  

  The by argument takes a list of columns. For a list like ['A', 'B'], it 
  assumes 'A' and 'B' are columns in both dataframes. For a list like [('A', 
  'B')], it assumes column 'A' in the left dataframe is the same as column 'B' 
  in the right dataframe. Can mix and match (e.g. by=['A', ('B', 'C')] will 
  assume both dataframes have column 'A', and column 'B' in the left dataframe 
  is the same as column 'C' in the right dataframe. If by is not specified, then
  all shared columns will be assumed to be the join columns.

  suffixes will 
  be used to rename columns that are common to both dataframes, but not used in 
  the join operation.
  e.g. `suffixes=('_1', '_2')`.
  If suffixes is not included, then the pandas default will be used ('_x', '_y')
  """

  __name__ = 'right_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'right'})
    return mutating_join(df, self.args[0], **self.kwargs)


def filtering_join(*args, **kwargs):
  left = args[0]
  right = args[1]
  if 'by' in kwargs:
    left_cols, right_cols = get_join_cols(kwargs['by'])
    cols = lambda right, left: right_cols
  else:
    left_cols, right_cols = None, None
    cols = lambda right, left: [x for x in left.columns.values.tolist() if x in right.columns.values.tolist()]
  if left._grouped_on:
    outDf = DplyFrame((left >> ungroup())
                      .merge(right[cols(left, right)].drop_duplicates(), 
                             how=kwargs['how'], left_on=left_cols, 
                             right_on=right_cols, indicator=True, 
                             suffixes=('', '_y'))
                      .query(kwargs['query'])
                      .regroup(left._grouped_on)
                      .iloc[:, range(0, len(left.columns))])
  else:
    outDf = DplyFrame(left.merge(right[cols(left, right)]
                                 .drop_duplicates(), how=kwargs['how'], 
                                 left_on=left_cols, right_on=right_cols,
                                 indicator=True, suffixes=('', '_y'))
                      .query(kwargs['query'])
                      .iloc[:, range(0, len(left.columns))])
  return outDf


class semi_join(Join):
  """ Perform filtering semi join
  >>> left_data >> semi_join(right_data[
  ...                                    , by=[join_columns_in_list_as_single_or_tuple]])
  e.g. flights2 >> semi_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe preserving any grouping from left dataframe

  Filters the left table by including observations that are also found in the
  right table. Never returns more observations than are found in the left table
  (i.e. multiple keys in the right table won't cause duplicate observations to 
  appear in the right table).
 
  The by argument takes a list of columns. For a
  list like ['A', 'B'], it assumes 'A' and 'B' are columns in both dataframes. 
  For a list like [('A', 'B')], it assumes column 'A' in the left dataframe is 
  the same as column 'B' in the right dataframe. Can mix and match (e.g. 
  by=['A', ('B', 'C')] will assume both dataframes have column 'A', and column 
  'B' in the left dataframe is the same as column 'C' in the right dataframe. If
  by is not specified, then all shared columns will be assumed to be the join 
  columns.
  
  """

  __name__ = 'semi_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'inner', 'query': '_merge=="both"'})
    return filtering_join(df, self.args[0], **self.kwargs)


class anti_join(Join):
  """ Perform filtering anti join
  >>> left_data >> anti_join(right_data[, by=[join_columns_in_list_as_single_or_tuple]])
  e.g. flights2 >> anti_join(airports, by=[('origin', 'faa')]) >> head(5)

  returns dataframe with any grouping preserved from left dataframe

  Filters the left table by including observations that are not found in the 
  right table. Never returns more observations than are found in the left table 
  (i.e. multiple keys in the right table won't cause duplicate observations to 
  appear in the right table).

  The by argument takes a list of columns. For a 
  list like ['A', 'B'], it assumes 'A' and 'B' are columns in both dataframes. 
  For a list like [('A', 'B')], it assumes column 'A' in the left dataframe is 
  the same as column 'B' in the right dataframe. Can mix and match (e.g. 
  by=['A', ('B', 'C')] will assume both dataframes have column 'A', and column 
  'B' in the left dataframe is the same as column 'C' in the right dataframe. If
   by is not specified, then all shared columns will be assumed to be the join 
  columns. 
  """

  __name__ = 'anti_join'

  def __call__(self, df):
    self.kwargs.update({'how': 'left', 'query': '_merge=="left_only"'})
    return filtering_join(df, self.args[0], **self.kwargs)
