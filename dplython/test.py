# Chris Riederer
# 2016-02-21

"""Testing for python dplyr."""

import math
import unittest
import os

import pandas as pd

from dplython import *


def load_diamonds():
    root = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root, 'data', 'diamonds.csv')
    return DplyFrame(pd.read_csv(path))


class TestLaterStrMethod(unittest.TestCase):
  
  def test_column_name(self):
    foo = X.foo
    self.assertEqual(foo._str, 'X["foo"]')

  def test_str(self):
    foo = X.foo
    self.assertEqual(str(foo), 'X["foo"]')

  def test_later_with_method(self):
    foo = X.foo.mean()
    self.assertEqual(str(foo), 'X["foo"].mean()')

  def test_later_with_method_call(self):
    foo = X.foo.mean()
    self.assertEqual(str(foo), 'X["foo"].mean()')
    foo = X.foo.mean(1)
    self.assertEqual(str(foo), 'X["foo"].mean(1)')
    foo = X.foo.mean(1, 2)
    self.assertEqual(str(foo), 'X["foo"].mean(1, 2)')
    foo = X.foo.mean(numeric_only=True)
    self.assertEqual(str(foo), 'X["foo"].mean(numeric_only=True)')
    # The order is different here, because the original order of the kwargs is
    # lost when kwargs are passed to the function. To insure consistent results,
    #  the kwargs are sorted alphabetically by key. To help deal with this
    # issue, support PEP 0468: https://www.python.org/dev/peps/pep-0468/
    foo = X.foo.mean(numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X["foo"].mean(level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(1, numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X["foo"].mean(1, level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(1, 2, numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X["foo"].mean(1, 2, level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(X.y.mean())
    self.assertEqual(str(foo), 'X["foo"].mean('
                               'X["y"].mean())')

  def test_later_with_delayed_function(self):
    mylen = DelayFunction(len)
    foo = mylen(X.foo)
    self.assertEqual(str(foo), 'len(X["foo"])')

  def test_more_later_ops_str(self):
    mylen = DelayFunction(len)
    foo = -mylen(X.foo) + X.y.mean() // X.y.median()
    self.assertEqual(str(foo), '-len(X["foo"]) + '
                               'X["y"].mean() // '
                               'X["y"].median()')
    bar = -(mylen(X.bar) + X.y.mean()) * X.y.median()
    self.assertEqual(str(bar), '-(len(X["bar"]) + X["y"].mean()) * '
                               'X["y"].median()')
    baz = 6 + (X.y.mean() % 4) - X.bar.sum()
    self.assertEqual(str(baz), '6 + X["y"].mean() % 4 - X["bar"].sum()')
    buzz = (X.bar / 4) == X.baz
    self.assertEqual(str(buzz), 'X["bar"] / 4 == X["baz"]')
    biz = X.foo[4] / X.bar[2:3] + X.baz[::2]
    self.assertEqual(str(biz), 'X["foo"][4] / X["bar"][2:3] + X["baz"][::2]')


class TestMutates(unittest.TestCase):
  diamonds = load_diamonds()

  def test_equality(self):
    self.assertTrue(self.diamonds.equals(self.diamonds))

  def test_addconst(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["ones"] = 1
    diamonds_dp = self.diamonds >> mutate(ones=1)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_newcolumn(self):
    diamonds_pd = self.diamonds.copy()
    newcol = range(len(diamonds_pd))
    diamonds_pd["newcol"] = newcol
    diamonds_dp = self.diamonds >> mutate(newcol=newcol)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_dupcolumn(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"]
    diamonds_dp = self.diamonds >> mutate(copy_x=X.x)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_editcolumn(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = 2 * diamonds_pd["x"]
    diamonds_dp = self.diamonds >> mutate(copy_x=X.x * 2)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def test_multi(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"]
    diamonds_pd["copy_y"] = diamonds_pd["y"]
    diamonds_dp = self.diamonds >> mutate(copy_x=X.x, copy_y=X.y)
    # Keep the new DplyFrame columns in the original order
    diamonds_dp = diamonds_dp[diamonds_pd.columns]
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def test_combine(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["copy_x"] = diamonds_pd["x"] + diamonds_pd["y"]
    diamonds_dp = self.diamonds >> mutate(copy_x=X.x + X.y)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def test_orignalUnaffect(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_dp = self.diamonds >> mutate(copy_x=X.x, copy_y=X.y)
    self.assertTrue(diamonds_pd.equals(self.diamonds))    

  def testCallMethodOnLater(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgX"] = diamonds_pd.x.mean()
    diamonds_dp = self.diamonds >> mutate(avgX=X.x.mean())
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testCallMethodOnCombinedLater(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgX"] = (diamonds_pd.x + diamonds_pd.y).mean()
    diamonds_dp = self.diamonds >> mutate(avgX=(X.x + X.y).mean())
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testReverseThings(self):
    self.diamonds >> mutate(foo=1 - X.carat, bar=7 // X.x, baz=4 % X.y.round())

  def testMethodFirst(self):
    diamonds_dp = self.diamonds >> mutate(avgDiff=X.x.mean() - X.x)
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgDiff"] = diamonds_pd["x"].mean() - diamonds_pd["x"]
    self.assertTrue(diamonds_dp["avgDiff"].equals(diamonds_pd["avgDiff"]))

  def testArgsNotKwargs(self):
    diamonds_dp = mutate(self.diamonds, X.carat+1)
    diamonds_pd = self.diamonds.copy()
    diamonds_pd['X["carat"] + 1'] = diamonds_pd.carat + 1
    self.assertTrue(diamonds_pd.equals(diamonds_dp))


class TestSelects(unittest.TestCase):
  diamonds = load_diamonds()

  def testOne(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["cut"]]
    diamonds_dp = self.diamonds >> select(X.cut)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testTwo(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["cut", "carat"]]
    diamonds_dp = self.diamonds >> select(X.cut, X.carat)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testChangeOrder(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[["carat", "cut"]]
    diamonds_dp = self.diamonds >> select(X.carat, X.cut)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    


class TestFilters(unittest.TestCase):
  diamonds = load_diamonds()

  def testFilterEasy(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[diamonds_pd.cut == "Ideal"]
    diamonds_dp = self.diamonds >> sift(X.cut == "Ideal")
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def testFilterNone(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_dp = self.diamonds >> sift()
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithMultipleArgs(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") & 
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds >> sift(X.cut == "Ideal", X.carat > 3)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithAnd(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") & 
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds >> sift((X.cut == "Ideal") & (X.carat > 3))
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterWithOr(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[(diamonds_pd.cut == "Ideal") |
                              (diamonds_pd.carat > 3)]
    diamonds_dp = self.diamonds >> sift((X.cut == "Ideal") | (X.carat > 3))
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    

  def testFilterMultipleLaterColumns(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd = diamonds_pd[diamonds_pd.carat > diamonds_pd.x - diamonds_pd.y]
    diamonds_dp = self.diamonds >> sift(X.carat > X.x - X.y)
    self.assertTrue(diamonds_pd.equals(diamonds_dp))    


# TODO: Add more multiphase tests


class TestGroupBy(unittest.TestCase):
  diamonds = load_diamonds()

  def testGroupbyDoesntDie(self):
    self.diamonds >> group_by(X.color)

  def testUngroupDoesntDie(self):
    self.diamonds >> ungroup()

  def testGroupbyAndUngroupDoesntDie(self):
    self.diamonds >> group_by(X.color) >> ungroup()

  def testOneGroupby(self):
    diamonds_pd = self.diamonds.copy()
    carats_pd = set(diamonds_pd[diamonds_pd.carat > 3.5].groupby('color').mean()["carat"])
    diamonds_dp = (self.diamonds >> 
                    sift(X.carat > 3.5) >>
                    group_by(X.color) >> 
                    mutate(caratMean=X.carat.mean()))
    carats_dp = set(diamonds_dp["caratMean"].values)
    self.assertEqual(carats_pd, carats_dp)

  def testTwoGroupby(self):
    diamonds_pd = self.diamonds.copy()
    carats_pd = set(diamonds_pd[diamonds_pd.carat > 3.5].groupby(["color", "cut"]).mean()["carat"])
    diamonds_dp = (self.diamonds >> 
                    sift(X.carat > 3.5) >>
                    group_by(X.color, X.cut) >> 
                    mutate(caratMean=X.carat.mean()))
    carats_dp = set(diamonds_dp["caratMean"].values)
    self.assertEqual(carats_pd, carats_dp)

  def testGroupThenFilterDoesntDie(self):
    diamonds_dp = (self.diamonds >> 
                    group_by(X.color) >> 
                    sift(X.carat > 3.5) >>
                    mutate(caratMean=X.carat.mean()))

  def testGroupThenFilterDoesntDie2(self):
    diamonds_dp = (self.diamonds >> 
                    group_by(X.color) >> 
                    sift(X.carat > 3.5, X.color != "I") >>
                    mutate(caratMean=X.carat.mean()))

  def testGroupUngroupSummarize(self):
    num_rows = (self.diamonds >> group_by(X.cut) >> ungroup() >> 
                      summarize(total=X.price.sum()) >> nrow())
    self.assertEqual(num_rows, 1)
    sum_price = self.diamonds.sum()["price"]
    sum_price_dp = (self.diamonds >> group_by(X.cut) >> ungroup() >> 
                      summarize(total=X.price.sum()) >> X.total[0])
    self.assertEqual(sum_price, sum_price_dp)

  def testDfRemainsGroupedAfterOperation(self):
    diamonds_dp = (self.diamonds >>
                    group_by(X.color) >>
                    mutate(caratMean1=X.carat.mean()) >>
                    mutate(caratMean2=X.carat.mean()))
    self.assertTrue(diamonds_dp["caratMean1"].equals(diamonds_dp["caratMean2"]))


class TestArrange(unittest.TestCase):
  diamonds = load_diamonds()

  def testArrangeDoesntDie(self):
    self.diamonds >> arrange(X.cut)

  def testArrangeThenSelect(self):
    self.diamonds >> arrange(X.color) >> select(X.color)

  def testMultipleSort(self):
    self.diamonds >> arrange(X.color, X.cut) >> select(X.color)

  def testArrangeSorts(self):
    sortedColor_pd = self.diamonds.copy().sort_values("color")["color"]
    sortedColor_dp = (self.diamonds >> arrange(X.color))["color"]
    self.assertTrue(sortedColor_pd.equals(sortedColor_dp))

  def testMultiArrangeSorts(self):
    sortedCarat_pd = self.diamonds.copy().sort_values(["color", "carat"])["carat"]
    sortedCarat_dp = (self.diamonds >> arrange(X.color, X.carat))["carat"]
    self.assertTrue(sortedCarat_pd.equals(sortedCarat_dp))

  def testArrangeDescending(self):
    sortedCarat_pd = self.diamonds.copy().sort_values("carat", ascending=False)
    sortedCarat_dp = self.diamonds >> arrange(-X.carat)
    self.assertTrue((sortedCarat_pd.carat == sortedCarat_dp.carat).all())

  def testArrangeByComputedLater(self):
    sortedDf = self.diamonds >> arrange((X.carat-3)**2)
    self.assertEqual((sortedDf.iloc[0].carat-3)**2,
                     min((self.diamonds.carat-3)**2))
    self.assertEqual((sortedDf.iloc[-1].carat-3)**2,
                     max((self.diamonds.carat-3)**2))


class TestSample(unittest.TestCase):
  diamonds = load_diamonds()

  def testSamplesDontDie(self):
    self.diamonds >> sample_n(5)
    self.diamonds >> sample_frac(0.5)

  def testSamplesGetsRightNumber(self):
    shouldBe5 = self.diamonds >> sample_n(5) >> X._.__len__()
    self.assertEqual(shouldBe5, 5)
    frac = len(self.diamonds) * 0.1
    shouldBeFrac = self.diamonds >> sample_frac(0.1) >> X._.__len__()
    self.assertEqual(shouldBeFrac, frac)

  def testSampleEqualsPandasSample(self):
    for i in [1, 10, 100, 1000]:
      shouldBeI = self.diamonds >> sample_n(i) >> X._.__len__()
      self.assertEqual(shouldBeI, i)
    for i in [.1, .01, .001]:
      shouldBeI = self.diamonds >> sample_frac(i) >> X._.__len__()
      self.assertEqual(shouldBeI, round(len(self.diamonds)*i))

  def testSample0(self):
    shouldBe0 = self.diamonds >> sample_n(0) >> X._.__len__()
    self.assertEqual(shouldBe0, 0)
    shouldBeFrac = self.diamonds >> sample_frac(0) >> X._.__len__()
    self.assertEqual(shouldBeFrac, 0.)

  def testGroupedSample(self):
    num_groups = len(set(self.diamonds["cut"]))
    for i in [0, 1, 10, 100, 1000]:
      numRows = self.diamonds >> group_by(X.cut) >> sample_n(i) >> X._.__len__()
      self.assertEqual(numRows, i*num_groups)
    for i in [.1, .01, .001]:
      shouldBeI = self.diamonds >> group_by(X.cut) >> sample_frac(i) >> X._.__len__()
      out = sum([len(self.diamonds[self.diamonds.cut == c].sample(frac=i)) for c in set(self.diamonds.cut)])
      # self.assertEqual(shouldBeI, math.floor(len(self.diamonds)*i))
      self.assertEqual(shouldBeI, out)


class TestSummarize(unittest.TestCase):
  diamonds = load_diamonds()

  def testSummarizeDoesntDie(self):
    self.diamonds >> summarize(sumX=X.x.sum())

  def testSummarizeX(self):
    diamonds_pd = self.diamonds.copy()
    sumX_pd = diamonds_pd.sum()["x"]
    sumX_dp = (self.diamonds >> summarize(sumX=X.x.sum()))["sumX"][0]
    self.assertEqual(round(sumX_pd), round(sumX_dp))

  def testSummarizeGroupedX(self):
    diamonds_pd = self.diamonds.copy()
    sumX_pd = diamonds_pd.groupby("cut").sum()["x"]
    val_pd = sumX_pd.values.copy()
    val_pd.sort()
    valX_dp = (self.diamonds >> group_by(X.cut) >>
                summarize(sumX=X.x.sum()) >> X._["sumX"]).values.copy()
    valX_dp.sort()
    for i, j in zip(val_pd, valX_dp):
      self.assertEqual(round(i), round(j))


class TestAlternateAttrGrab(unittest.TestCase):
  diamonds = load_diamonds()
  diamonds["o m g"] = range(len(diamonds))
  diamonds["0"] = range(len(diamonds))

  def testSelect(self):
    equality = self.diamonds[["o m g"]] == (self.diamonds >> select(X["o m g"]))
    self.assertTrue(equality.all()[0])

  def testMutate(self):
    pd = self.diamonds[["0"]] * 2
    dp = self.diamonds >> mutate(foo=X["0"]*2) >> select(X.foo)
    equality = pd["0"] == dp["foo"]
    self.assertTrue(equality.all())


class TestNrow(unittest.TestCase):
  diamonds = load_diamonds()

  def testSimpleNrow(self):
    diamonds_pd = self.diamonds.copy()
    self.assertEqual(len(diamonds_pd), self.diamonds >> nrow())

  def testMultipleNrow(self):
    diamonds_pd = self.diamonds.copy()
    self.assertEqual(len(diamonds_pd), self.diamonds >> nrow())
    self.assertEqual(len(diamonds_pd), self.diamonds >> nrow())

    small_d = diamonds_pd[diamonds_pd.carat > 4]
    self.assertEqual(
        len(small_d), self.diamonds >> sift(X.carat > 4) >> nrow())


class TestFunctionForm(unittest.TestCase):
  diamonds = load_diamonds()

  def testsift(self):
    normal = self.diamonds >> sift(X.carat > 4)
    function = sift(self.diamonds, X.carat > 4)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> sift()
    function = sift(self.diamonds)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> sift(X.carat < 4, X.color == "D")
    function = sift(self.diamonds, X.carat < 4, X.color == "D")
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> sift((X.carat < 4) | (X.color == "D"), 
                                      X.carat >= 4)
    function = sift(self.diamonds, (X.carat < 4) | (X.color == "D"), 
                                      X.carat >= 4)
    self.assertTrue(normal.equals(function))

  def testSelect(self):
    normal = self.diamonds >> select(X.carat)
    function = select(self.diamonds, X.carat)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> select(X.cut, X.carat, X.carat)
    function = select(self.diamonds, X.cut, X.carat, X.carat)
    self.assertTrue(normal.equals(function))

  def testMutate(self):
    normal = self.diamonds >> mutate(foo=X.carat)
    function = mutate(self.diamonds, foo=X.carat)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> mutate(a=X.cut, b=X.x/2, c=32)
    function = mutate(self.diamonds, a=X.cut, b=X.x/2, c=32)
    self.assertTrue(normal.equals(function))

  def testGroupBy(self):
    normal = (self.diamonds >> 
                    sift(X.carat > 3.5) >>
                    group_by(X.color) >> 
                    mutate(caratMean=X.carat.mean()))
    function = self.diamonds >> sift(X.carat > 3.5)
    function = group_by(function, X.color)
    function = function >> mutate(caratMean=X.carat.mean())
    self.assertTrue(normal.equals(function))

  def testGroupBy2(self):
    normal = (self.diamonds >> 
                    group_by(X.color, X.cut) >> 
                    mutate(caratMean=X.carat.mean()))
    function = group_by(self.diamonds, X.color, X.cut)
    function = function >> mutate(caratMean=X.carat.mean())
    self.assertTrue(normal.equals(function))

  def testUngroup(self):
    normal = (self.diamonds >> 
                    sift(X.carat > 3.5) >>
                    group_by(X.color) >> 
                    ungroup())
    function = (self.diamonds >> 
                    sift(X.carat > 3.5) >>
                    group_by(X.color))
    function = ungroup(function)
    self.assertTrue(normal.equals(function))

  def testArrange(self):
    normal = self.diamonds >> arrange(X.color, X.carat)
    function = arrange(self.diamonds, X.color, X.carat)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> arrange(X.cut, X.carat, X.y)
    function = arrange(self.diamonds, X.cut, X.carat, X.y)
    self.assertTrue(normal.equals(function))

  def testSummarize(self):
    normal = (self.diamonds >> summarize(sumX=X.x.sum()))
    function = summarize(self.diamonds, sumX=X.x.sum())
    self.assertTrue(normal.equals(function))

  def testSummarizeGroupedX(self):
    normal = self.diamonds >> group_by(X.cut) >> summarize(sumX=X.x.sum())
    function = self.diamonds >> group_by(X.cut)
    function = summarize(function, sumX=X.x.sum())
    self.assertTrue(normal.equals(function))

  def testUtilities(self):
    normal = self.diamonds >> head()
    function = head(self.diamonds)
    self.assertTrue(normal.equals(function))

    normal = self.diamonds >> sample_n(10)
    function = sample_n(self.diamonds, 10)
    self.assertEqual(len(normal), len(function))
    
    normal = self.diamonds >> sample_frac(0.1)
    function = sample_frac(self.diamonds, 0.1)
    self.assertEqual(len(normal), len(function))
    

if __name__ == '__main__':
  unittest.main()