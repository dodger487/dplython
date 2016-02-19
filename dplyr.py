# Chris Riederer
# 2015-10-30

"""Trying to put dplyr-style operations on top of pandas DataFrame."""

import numpy as np
import pandas
from pandas import DataFrame


class Manager(object):
  def __init__(self):
    self.currentDf = None

  def __getattr__(self, attr):
    return self.currentDf.__getattr__(attr)


X = Manager()


class Later(object):
  def __init__(self):
    self.num_args = 0
    self.fcn = None


class DplyFrame(DataFrame):
  global X
  def __or__(self, other):
    return other(self)

  def __call__(self):
    X.currentDf = self
    return self


class GroupedDplyFrame(object):
  global X

  def __init__(self, df, names):
    self.groups = {}
    name = names[0]
    vals = set(df[name])
    for v in vals:
      self.groups[v] = df[df[name] == v]

  def __or__(self, other):
    if other == UnGroupSignal:
      return self.ungroup()
    for k, v in self.groups.iteritems():
      self.groups[k] = other(v)
    return self

  def ungroup(self):
    return pandas.concat(self.groups.values())


class UnGroupSignal():
  pass


def _filter(stmt):
  return lambda df: DplyFrame(df[stmt])


def _select(*args):
  names =[column.name for column in args]
  return lambda df: DplyFrame(df[names])


def _mutate(**kwargs):
  def addColumns(df):
    for key, val in kwargs.iteritems():
      df[key] = val
    return DplyFrame(df)  # Probably don't need all these casts
  return addColumns


def _summarize(**kwargs):
  def addColumnsAndSelect(df):
    # DataFrame weirdly chokes on init if given a dictionary whose keys are not
    # iterable. It needs to be told explicitly the index.
    if not hasattr(kwargs.values()[0], '__iter__'):
      return DplyFrame(kwargs, index=[0])
    else:
      return DplyFrame(kwargs)
  return addColumnsAndSelect


def _arrange(*args):
  # TODO: add in descending and ascending
  names = [column.name for column in args]
  # TODO(cjr): my current version of Pandas isn't up-to-date, so this doesn't
  # exist yet. How embarrassing!
  # return lambda df: df.sort_values(names)
  return lambda df: DplyFrame(df.sort(names))


def _group_by(*args):
  names = [column.name for column in args]
  return lambda df: GroupedDplyFrame(df, names)


def _ungroup():
  return UnGroupSignal



def main():
  x = pandas.read_csv('./diamonds.csv')
  foo = DplyFrame(x)
  # print foo.columns
  # print foo.cut
  bar = (foo() | 
            _filter(np.random.rand(len(x)) < 0.01) |
            # _filter(X.cut == 'Ideal') | 
            _group_by(X.cut) |
            _mutate(better_carat=X.carat + 1, even_better_carat=X.carat + 2) | 
            _summarize(mean=X.carat.mean()))
            # _ungroup()) 
  print bar.groups

  # bar = (foo() | 
  #           # _filter(X.cut == 'Ideal') | 
  #           _group_by(X.cut) |
  #           _mutate(better_carat=X.carat + 1, even_better_carat=X.carat + 2) |
  #           _arrange(X.carat) |
  #           _summarize(mean=X.carat.mean())) 
  print bar
  return


if __name__ == '__main__':
  main()
