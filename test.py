# Chris Riederer
# 2016-02-21

"""Testing for python dplyr."""

import unittest

import pandas

from dplyr2 import *


class TestMutates(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def test_equality(self):
    self.assertTrue(self.diamonds.equals(self.diamonds))

  def test_addconst(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["ones"] = 1
    diamonds_dp = self.diamonds | mutate(ones=1)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_newcolumn(self):
    diamonds_pd = self.diamonds.copy()
    newcol = range(len(diamonds_pd))
    diamonds_pd["newcol"] = newcol
    diamonds_dp = self.diamonds | mutate(newcol=newcol)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_dupcolumn(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"]
    diamonds_dp = self.diamonds | mutate(copy_x=X.x)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_editcolumn(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = 2 * diamonds_pd["x"]
    diamonds_dp = self.diamonds | mutate(copy_x=X.x * 2)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_multi(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"]
    diamonds_pd["copy_y"] = diamonds_pd["y"]
    diamonds_dp = self.diamonds | mutate(copy_x=X.x, copy_y=X.y)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def test_combine(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"] + diamonds_pd["y"]
    diamonds_dp = self.diamonds | mutate(copy_x=X.x + X.y)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def test_orignalUnaffect(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_dp = self.diamonds | mutate(copy_x=X.x, copy_y=X.y)
    self.assertTrue(diamonds_pd.equals(self.diamonds))    

  def testCallMethodOnLater(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgX"] = diamonds_pd.x.mean()
    diamonds_dp = self.diamonds | mutate(avgX=X.x.mean())
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testCallMethodOnCombinedLater(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgX"] = (diamonds_pd.x + diamonds_pd.y).mean()
    diamonds_dp = self.diamonds | mutate(avgX=(X.x + X.y).mean())
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    


class TestSelects(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def testOne(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["cut"]]
    diamonds_dp = self.diamonds | select(X.cut)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testTwo(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["cut", "carat"]]
    diamonds_dp = self.diamonds | select(X.cut, X.carat)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testChangeOrder(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["carat", "cut"]]
    diamonds_dp = self.diamonds | select(X.carat, X.cut)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    


class TestFilters(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def testFilterEasy(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[diamonds_pd.cut == "Ideal"]
    diamonds_dp = self.diamonds | dfilter(X.cut == "Ideal")
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterNone(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_dp = self.diamonds | dfilter()
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithMultipleArgs(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") & 
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds | dfilter(X.cut == "Ideal", X.carat > 3)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithAnd(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") & 
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds | dfilter((X.cut == "Ideal") & (X.carat > 3))
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithOr(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") |
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds | dfilter((X.cut == "Ideal") | (X.carat > 3))
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterMultipleLaterColumns(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[diamonds_pd.carat > diamonds_pd.x - diamonds_pd.y]
    diamonds_dp = self.diamonds | dfilter(X.carat > X.x - X.y)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    


# TODO: Add more multiphase tests


class TestGroupBy(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def testGroupbyDoesntDie(self):
    self.diamonds | group_by(X.color)

  def testUngroupDoesntDie(self):
    self.diamonds | ungroup()

  def testGroupbyAndUngroupDoesntDie(self):
    self.diamonds | group_by(X.color) | ungroup()

  def testOneGroupby(self):
    diamonds_pd = self.diamonds.copy()
    carats_pd = set(diamonds_pd[diamonds_pd.carat > 3.5].groupby('color').mean()["carat"])
    diamonds_dp = (self.diamonds | 
                    dfilter(X.carat > 3.5) |
                    group_by(X.color) | 
                    mutate(caratMean=X.carat.mean()))
    carats_dp = set(diamonds_dp["caratMean"].values)
    self.assertEquals(carats_pd, carats_dp)

  def testTwoGroupby(self):
    diamonds_pd = self.diamonds.copy()
    carats_pd = set(diamonds_pd[diamonds_pd.carat > 3.5].groupby(["color", "cut"]).mean()["carat"])
    diamonds_dp = (self.diamonds | 
                    dfilter(X.carat > 3.5) |
                    group_by(X.color, X.cut) | 
                    mutate(caratMean=X.carat.mean()))
    carats_dp = set(diamonds_dp["caratMean"].values)
    self.assertEquals(carats_pd, carats_dp)

  def testGroupThenFilterDoesntDie(self):
    diamonds_dp = (self.diamonds | 
                    group_by(X.color) | 
                    dfilter(X.carat > 3.5) |
                    mutate(caratMean=X.carat.mean()))

  def testGroupThenFilterDoesntDie2(self):
    diamonds_dp = (self.diamonds | 
                    group_by(X.color) | 
                    dfilter(X.carat > 3.5, X.color != "I") |
                    mutate(caratMean=X.carat.mean()))


class TestSummarize(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def testArrangeDoesntDie(self):
    self.diamonds | arrange(X.cut)

  def testArrangeThenSelect(self):
    self.diamonds | arrange(X.color) | select(X.color)

  def testMultipleSort(self):
    self.diamonds | arrange(X.color, X.cut) | select(X.color)

  def testArrangeSorts(self):
    sortedColor_pd = self.diamonds.copy().sort("color")["color"]
    sortedColor_dp = (self.diamonds | arrange(X.color))["color"]
    self.assertEquals(sortedColor_pd, sortedColor_dp)

  def testMultiArrangeSorts(self):
    sortedCarat_pd = self.diamonds.copy().sort(["color", "carat"])["carat"]
    sortedCarat_dp = (self.diamonds | arrange(X.color))["carat"]
    self.assertEquals(sortedCarat_pd, sortedCarat_dp)


class TestSummarize(unittest.TestCase):
  diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

  def testSummarizeDoesntDie(self):
    self.diamonds | summarize(sumX=X.x.sum())

  def testSummarizeX(self):
    diamonds_pd = self.diamonds.copy()
    sumX_pd = diamonds_pd.sum()["x"]
    sumX_dp = (self.diamonds | summarize(sumX=X.x.sum()))["sumX"][0]
    self.assertEquals(round(sumX_pd), round(sumX_dp))

  def testSummarizeGroupedX(self):
    diamonds_pd = self.diamonds.copy()
    sumX_pd = diamonds_pd.groupby("cut").sum()["x"]["Fair"]
    sumX_dp = (self.diamonds | group_by(X.cut) | summarize(sumX=X.x))["sumX"]
    self.assertEquals(sumX_pd, sumX_dp)
    

if __name__ == '__main__':
  unittest.main()