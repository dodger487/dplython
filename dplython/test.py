# Chris Riederer
# 2016-02-21

"""Testing for python dplyr."""

from collections import Counter
import math
import unittest
import os

# import this for quick dataframe creation
import sys
if sys.version_info[0] < 3:
  from StringIO import StringIO
else:
  from io import StringIO

import numpy as np
import numpy.testing as npt
import pandas as pd

from dplython import *


def load_diamonds():
    root = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root, 'data', 'diamonds.csv')
    return DplyFrame(pd.read_csv(path))


class TestLaterStrMethod(unittest.TestCase):
  
  def test_column_name(self):
    foo = X.foo
    self.assertEqual(foo._str, 'X.foo.')

  def test_str(self):
    foo = X.foo
    self.assertEqual(str(foo), 'X.foo')

  def test_bracket(self):
    foo = X["foo"]
    self.assertEqual(str(foo), 'X["foo"]')

  def test_later_with_method(self):
    foo = X.foo.mean()
    self.assertEqual(str(foo), 'X.foo.mean()')

  def test_later_with_method_call(self):
    foo = X.foo.mean()
    self.assertEqual(str(foo), 'X.foo.mean()')
    foo = X.foo.mean(1)
    self.assertEqual(str(foo), 'X.foo.mean(1)')
    foo = X.foo.mean(1, 2)
    self.assertEqual(str(foo), 'X.foo.mean(1, 2)')
    foo = X.foo.mean(numeric_only=True)
    self.assertEqual(str(foo), 'X.foo.mean(numeric_only=True)')
    # The order is different here, because the original order of the kwargs is
    # lost when kwargs are passed to the function. To insure consistent results,
    #  the kwargs are sorted alphabetically by key. To help deal with this
    # issue, support PEP 0468: https://www.python.org/dev/peps/pep-0468/
    foo = X.foo.mean(numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X.foo.mean(level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(1, numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X.foo.mean(1, level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(1, 2, numeric_only=True, level="bar")
    self.assertEqual(str(foo), 'X.foo.mean(1, 2, level="bar", '
                               'numeric_only=True)')
    foo = X.foo.mean(X.y.mean())
    self.assertEqual(str(foo), 'X.foo.mean('
                               'X.y.mean())')

  def test_later_with_delayed_function(self):
    mylen = DelayFunction(len)
    foo = mylen(X.foo)
    self.assertEqual(str(foo), 'len(X.foo)')

  def test_more_later_ops_str(self):
    mylen = DelayFunction(len)
    foo = -mylen(X.foo) + X.y.mean() // X.y.median()
    self.assertEqual(str(foo), '-len(X.foo) + '
                               'X.y.mean() // '
                               'X.y.median()')
    bar = -(mylen(X.bar) + X.y.mean()) * X.y.median()
    self.assertEqual(str(bar), '-(len(X.bar) + X.y.mean()) * '
                               'X.y.median()')
    baz = 6 + (X.y.mean() % 4) - X.bar.sum()
    self.assertEqual(str(baz), '6 + X.y.mean() % 4 - X.bar.sum()')
    buzz = (X.bar / 4) == X.baz
    self.assertEqual(str(buzz), 'X.bar / 4 == X.baz')
    biz = X.foo[4] / X.bar[2:3] + X.baz[::2]
    self.assertEqual(str(biz), 'X.foo[4] / X.bar[2:3] + X.baz[::2]')


class TestDelayFunctions(unittest.TestCase):
  diamonds = load_diamonds()

  def test_function_args(self):
    foo_pd = PairwiseGreater(self.diamonds["x"], self.diamonds["y"])
    foo_dp = self.diamonds >> mutate(foo=PairwiseGreater(X.x, X.y)) >> X._.foo
    self.assertTrue((foo_pd == foo_dp).all())

  def test_function_kwargs(self):
    @DelayFunction
    def PairwiseGreaterKwargs(series1=None, series2=None):
      index = series1.index
      newSeries = pd.Series(np.zeros(len(series1)))
      s1_ind = series1 > series2
      s2_ind = series1 <= series2
      newSeries[s1_ind] = series1[s1_ind]
      newSeries[s2_ind] = series2[s2_ind]
      newSeries.index = index
      return newSeries

    foo_pd = PairwiseGreaterKwargs(
        series1=self.diamonds["x"], series2=self.diamonds["y"])
    foo_dp = (self.diamonds >>
                mutate(foo=PairwiseGreaterKwargs(series1=X.x, series2=X.y)) >>
                X._.foo)
    self.assertTrue((foo_pd == foo_dp).all())


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
    dp = self.diamonds >> mutate(one_minus_carat=1 - X.carat, 
                                 carat_minus_one=X.carat-1)
    self.assertTrue(dp.one_minus_carat.equals(1 - self.diamonds.carat))
    self.assertTrue(dp.carat_minus_one.equals(self.diamonds.carat - 1))

  def testMethodFirst(self):
    diamonds_dp = self.diamonds >> mutate(avgDiff=X.x.mean() - X.x)
    diamonds_pd = self.diamonds.copy()
    diamonds_pd["avgDiff"] = diamonds_pd["x"].mean() - diamonds_pd["x"]
    self.assertTrue(diamonds_dp["avgDiff"].equals(diamonds_pd["avgDiff"]))

  def testArgsNotKwargs(self):
    diamonds_dp = mutate(self.diamonds, X.carat+1)
    diamonds_pd = self.diamonds.copy()
    diamonds_pd['X.carat + 1'] = diamonds_pd.carat + 1
    self.assertTrue(diamonds_pd.equals(diamonds_dp))

  def testOrderedKwargs(self):
    # without __order, is alphabetical
    diamonds_dp = mutate(self.diamonds,
                         carat2=X.carat+2,
                         carat1=X.carat+1,
                         carat3=X.carat+3)
    col_names = diamonds_dp.columns.values
    self.assertEqual(col_names[-3], "carat1")
    self.assertEqual(col_names[-2], "carat2")
    self.assertEqual(col_names[-1], "carat3")

    diamonds_dp = mutate(self.diamonds,
                         carat2=X.carat+2,
                         carat1=X.carat2-1,
                         carat3=X.carat1+2,
                         __order=["carat2", "carat1", "carat3"])
    
    col_names = diamonds_dp.columns.values
    self.assertEqual(col_names[-3], "carat2")
    self.assertEqual(col_names[-2], "carat1")
    self.assertEqual(col_names[-1], "carat3")
    self.assertTrue((diamonds_dp.carat + 3 == diamonds_dp.carat3).all())

  def testOrderedKwargsError(self):
  	self.assertRaisesRegexp(ValueError, "carat2", mutate,
  							self.diamonds, carat1 = X.carat + 1,
  							__order = ["carat1", "carat2"])

  	self.assertRaisesRegexp(ValueError, "carat3", mutate,
  							self.diamonds, carat1 = X.carat + 1, carat3 = X.carat + 3,
  							__order = ["carat1"])


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

  def testGroupingImpliedSelect(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_grouped_1 = diamonds_pd >> group_by(X.cut, X.clarity) >> select(X.x)
    diamonds_selected_1 = diamonds_pd >> select(X.cut, X.clarity, X.x)
    diamonds_grouped_2 = diamonds_pd >> group_by(X.cut, X.clarity) >> select(X.x, X.clarity)
    diamonds_selected_2 = diamonds_pd >> select(X.cut, X.x, X.clarity)
    self.assertTrue(diamonds_grouped_1.equals(diamonds_selected_1))
    self.assertTrue(diamonds_grouped_2.equals(diamonds_selected_2))

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

  def testGroupByWithArgsAndKwargs(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd['appearance'] = diamonds_pd['color'] + diamonds_pd['clarity']
    diamonds_pd = diamonds_pd.groupby(['cut', 'appearance'])['price'].sum()
    diamonds_grouped = (
      self.diamonds >>
      group_by(X.cut, appearance=X.color + X.clarity) >>
      summarize(total_price=X.price.sum())
    )
    for row in diamonds_grouped.itertuples():
      self.assertEqual(row.total_price, diamonds_pd[row.cut][row.appearance])

  def testGroupyByWithKwargsOnly(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd['appearance'] = diamonds_pd['color'] + diamonds_pd['clarity']
    diamonds_pd = diamonds_pd.groupby(['appearance'])['price'].sum()
    diamonds_grouped = (
      self.diamonds >>
      group_by(appearance=X.color + X.clarity) >>
      summarize(total_price=X.price.sum())
    )
    for row in diamonds_grouped.itertuples():
      self.assertEqual(row.total_price, diamonds_pd[row.appearance])

  def testGroupByWithPositionalArg(self):
    diamonds_pd = self.diamonds.copy()
    diamonds_pd['appearance'] = diamonds_pd['color'] + diamonds_pd['clarity']
    diamonds_pd = diamonds_pd.groupby(['cut', 'appearance'])['price'].sum()
    diamonds_grouped = (
      self.diamonds >>
      group_by(X.cut, X.color + X.clarity) >>
      summarize(total_price=X.price.sum())
    )
    self.assertTrue((diamonds_grouped.total_price ==  diamonds_pd.values).all())

  def testGroupByIterable(self):
    diamonds_pd = self.diamonds.copy()
    random_labels = np.random.choice(['a', 'b'], len(self.diamonds))
    diamonds_pd['label'] = random_labels
    diamonds_pd = diamonds_pd.groupby(['label'])['price'].sum()
    diamonds_grouped = (
      self.diamonds >>
      group_by(label=random_labels) >>
      summarize(total_price=X.price.sum())
    )
    for row in diamonds_grouped.itertuples():
      self.assertEqual(row.total_price, diamonds_pd[row.label])

  def testGroupByMutate(self):
    df = self.diamonds >> mutate(bin=X["Unnamed: 0"] % 5000)
    gbinp = df.groupby("bin")
    df["foo2"] = (df >> group_by(X.bin) >> mutate(foo=X.x.mean() + X.y.mean()))["foo"]
    df["foo1"] = gbinp.x.transform('mean') + gbinp.y.transform('mean')
    npt.assert_allclose(df.foo1, df.foo2)
    # self.assertAlmostEqual(df.foo1, df.foo2)


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


class TestCount(unittest.TestCase):
  diamonds = load_diamonds()

  def testCount(self):
    counts_pd = Counter(self.diamonds.cut)
    counts_dp = self.diamonds >> count(X.cut)

    counts_pd = sorted(list(counts_pd.values()))
    counts_dp = sorted(list(counts_dp.n))

    self.assertEqual(counts_pd, counts_dp)

  def testCountExpression(self):
    counts_pd = Counter(self.diamonds.carat // 1)
    counts_dp = self.diamonds >> count(X.carat // 1)

    counts_pd = sorted(list(counts_pd.values()))
    counts_dp = sorted(list(counts_dp.n))

    self.assertEqual(counts_pd, counts_dp)

  def testCountMulti(self):
    counts_pd = Counter(zip(self.diamonds["cut"], self.diamonds["color"]))
    counts_dp = self.diamonds >> count(X.cut, X.color)

    counts_pd = sorted(list(counts_pd.values()))
    counts_dp = sorted(list(counts_dp.n))

    self.assertEqual(counts_pd, counts_dp)


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

  def testSummarizeGrouping(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                                , 'y': range(1, 5)}))
    a_summarized = a >> group_by(X.x) >> summarize(mean_y = X.y.mean())
    self.assertIsNone(a_summarized._grouped_on)


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
    

class TestIfElse(unittest.TestCase):
  diamonds = load_diamonds()

  def test_if_else(self):
    foo = self.diamonds >> mutate(
        conditional_results= if_else(X.cut == "Premium", X.color, X.clarity))
    bar = [color if cut == "Premium" else clarity for color, clarity, cut
        in zip(self.diamonds.color, self.diamonds.clarity, self.diamonds.cut)]
    bar = pd.Series(bar)
    self.assertTrue(foo["conditional_results"].equals(bar))

  # Porting dplyr tests:
  # https://github.com/hadley/dplyr/blob/master/tests/testthat/test-if-else.R
  def test_if_else_work(self):
    x = pd.Series([-1, 0, 1])
    zeros = pd.Series([0, 0, 0])
    self.assertTrue(if_else(x < 0, x, zeros).equals(pd.Series([-1, 0, 0])))
    self.assertTrue(if_else(x > 0, x, zeros).equals(pd.Series([0, 0, 1])))


class TestRename(unittest.TestCase):
  diamonds = load_diamonds()

  def test_rename(self):
    renamed_df = self.diamonds >> rename(chair=X.table)
    self.assertNotIn('table', renamed_df.columns)
    self.assertIn('chair', renamed_df.columns)


class TestTransmute(unittest.TestCase):
  diamonds = load_diamonds()

  def test_transmute(self):
    mutate_select_df = (self.diamonds >>
                        mutate(new_price=X.price * 2, x_plus_y=X.x + X.y,
                               __order=['new_price', 'x_plus_y']) >>
                        select(X.new_price, X.x_plus_y))
    transmute_df = (self.diamonds >>
                    transmute(new_price=X.price * 2, x_plus_y=X.x + X.y,
                              __order=('new_price', 'x_plus_y')))
    self.assertTrue(mutate_select_df.equals(transmute_df))

  def test_transmute_args(self):
    mutate_select_df = (self.diamonds >>
                        mutate(X.price * 2, X.x + X.y) >>
                        select(X["X.price * 2"], X["X.x + X.y"]))
    transmute_df = (self.diamonds >>
                    transmute(X.price * 2, X.x + X.y))
    self.assertTrue(mutate_select_df.equals(transmute_df))


class TestMutatingJoins(unittest.TestCase):

  def test_inner_join(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                              , 'y': [1, 2, 3, 4]}))
    b = DplyFrame(pd.DataFrame({'x': [1, 2, 2, 4]
                              , 'z': [1, 2, 3, 4]}))
    j_inner_test = a >> inner_join(b, by=['x'])
    j_inner_pd = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2]
                       , 'y': [1, 2, 3, 3]
                       , 'z': [1, 1, 2, 3]}))
    self.assertTrue(j_inner_test.equals(j_inner_pd))
    self.assertTrue(len(j_inner_test.columns.difference(j_inner_pd.columns)) == 0)
    # test normal function
    j1 = inner_join(a, b)
    j2 = a >> inner_join(b)
    self.assertTrue(j1.equals(j2))
    j1 = inner_join(a, b, by=['x'])
    j2 = a >> inner_join(b, by=['x'])
    self.assertTrue(j1.equals(j2))
    # test on grouped data
    j1 = a >> group_by(X.x) >> inner_join(b)
    # dataframes compare equal...
    self.assertTrue(j1.equals(j2))
    # but have different grouping
    self.assertTrue(j1._grouped_on == ['x'])
    self.assertTrue(j2._grouped_on is None)



  def test_full_join(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 2, 3]
                                , 'y': [2, 3, 4]}))
    b = DplyFrame(pd.DataFrame({'x': [3, 4, 5]
                                , 'z': [3, 4, 5]}))
    j_full_test = a >> full_join(b, by=['x'])
    j_full_pd = DplyFrame(pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0] # pandas promotes ints to floats in the join
                              , 'y': [2, 3, 4, np.nan, np.nan]
                              , 'z': [np.nan, np.nan, 3, 4, 5]}))
    self.assertTrue(j_full_test.equals(j_full_pd))
    self.assertTrue(len(j_full_test.columns.difference(j_full_pd.columns)) == 0)
    # test normal form
    j1 = full_join(a, b)
    j2 = a >> full_join(b)
    self.assertTrue(j1.equals(j2))
    j1 = full_join(a, b, by=['x'])
    j2 = a >> full_join(b, by=['x'])
    self.assertTrue(j1.equals(j2))
    # test on grouped data
    j1 = a >> group_by(X.x) >> full_join(b)
    # dataframes compare equal...
    self.assertTrue(j1.equals(j2))
    # but have different grouping
    self.assertTrue(j1._grouped_on == ['x'])
    self.assertTrue(j2._grouped_on is None)


  def test_left_join(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                                , 'y': [1, 2, 3, 4]}))
    b = DplyFrame(pd.DataFrame({'x': [1, 2, 2, 4]
                                , 'z': [1, 2, 3, 4]}))
    j_left_test = a >> left_join(b, by=['x'])
    j_left_pd = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2, 3]
                              , 'y': [1, 2, 3, 3, 4]
                              , 'z': [1, 1, 2, 3, np.nan]}))
    j_left_test.equals(j_left_pd)
    self.assertTrue(len(j_left_test.columns.difference(j_left_pd.columns)) == 0)
    # test normal
    j1 = left_join(a, b)
    j2 = a >> left_join(b)
    self.assertTrue(j1.equals(j2))
    j1 = left_join(a, b, by=['x'])
    j2 = a >> left_join(b, by=['x'])
    self.assertTrue(j1.equals(j2))
    # test on grouped data
    j1 = a >> group_by(X.x) >> left_join(b)
    # dataframes compare equal...
    self.assertTrue(j1.equals(j2))
    # but have different grouping
    self.assertTrue(j1._grouped_on == ['x'])
    self.assertTrue(j2._grouped_on is None)

  def test_right_join(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                                , 'y': [1, 2, 3, 4]}))
    b = DplyFrame(pd.DataFrame({'x': [1, 2, 2, 4]
                                , 'z': [1, 2, 3, 4]}))
    j_right_test = a >> right_join(b, by=['x'])
    j_right_pd = DplyFrame(pd.DataFrame({'x': [1.0, 1.0, 2.0, 2.0, 4.0]
                              , 'y': [1.0, 2.0, 3.0, 3.0, np.nan]
                              , 'z': [1, 1, 2, 3, 4]}))
    self.assertTrue(j_right_test.equals(j_right_pd))
    self.assertTrue(len(j_right_test.columns.difference(j_right_pd.columns)) == 0)
    # test normal form
    j1 = right_join(a, b)
    j2 = a >> right_join(b)
    self.assertTrue(j1.equals(j2))
    j1 = right_join(a, b, by=['x'])
    j2 = a >> right_join(b, by=['x'])
    self.assertTrue(j1.equals(j2))
    # test on grouped data
    j1 = a >> group_by(X.x) >> right_join(b)
    # dataframes compare equal...
    self.assertTrue(j1.equals(j2))
    # but have different grouping
    self.assertTrue(j1._grouped_on == ['x'])
    self.assertTrue(j2._grouped_on is None)

  def test_suffixes_join(self):
    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                                , 'z': [1, 2, 3, 4]}))
    b = DplyFrame(pd.DataFrame({'x': [1, 2, 2, 4]
                                , 'z': [1, 2, 3, 4]}))
    j_suffix_test = a >> left_join(b, by=['x'])
    j_suffix_pd = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2, 3]
                              , 'z_x': [1, 2, 3, 3, 4]
                              , 'z_y': [1.0, 1.0, 2.0, 3.0, np.nan]}))
    self.assertTrue((j_suffix_test.columns == j_suffix_pd.columns).all())
    j_suffix_test = a >> left_join(b, by=['x'], suffixes=('_1', '_2'))
    j_suffix_pd = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2, 3]
                              , 'z_1': [1, 2, 3, 3, 4]
                              , 'z_2': [1.0, 1.0, 2.0, 3.0, np.nan]}))
    self.assertTrue((j_suffix_test.columns == j_suffix_pd.columns).all())

  def test_multiple_columns_join(self):

    a = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 3]
                                , 'y': [1, 1, 2, 3]
                                , 'a': [1, 2, 3, 4]})[['x', 'y', 'a']])
    b = DplyFrame(pd.DataFrame({'x': [1, 2, 2, 4]
                                , 'y': [1, 2, 2, 4]
                                , 'b': [1, 2, 3, 4]})[['x', 'y', 'b']])
    j_multiple_col_test = a >> left_join(b, by=['x', 'y'])
    j_multiple_col_pd = DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2, 3]
                                      , 'y': [1, 1, 2, 2, 3]
                                      , 'a': [1, 2, 3, 3, 4]
                                      , 'b': [1.0, 1.0, 2.0, 3.0, np.nan]}))[['x', 'y', 'a', 'b']]
    self.assertTrue(j_multiple_col_test.equals(j_multiple_col_pd))
    b = DplyFrame(pd.DataFrame({'z': [1, 2, 2, 4]
                                , 'y': [1, 2, 2, 4]
                                , 'b': [1, 2, 3, 4]})[['z', 'y', 'b']])
    j_multiple_col_test = a >> left_join(b, by=[('x', 'z'), 'y'])
    j_multiple_col_pd =DplyFrame(pd.DataFrame({'x': [1, 1, 2, 2, 3]
                                      , 'y': [1, 1, 2, 2, 3]
                                      , 'a': [1, 2, 3, 3, 4]
                                      , 'z': [1.0, 1.0, 2.0, 2.0, np.nan]
                                      , 'b': [1.0, 1.0, 2.0, 3.0, np.nan]}))[['x', 'y', 'a', 'z', 'b']]
    self.assertTrue(j_multiple_col_test.equals(j_multiple_col_pd))


class TestFilteringJoins(unittest.TestCase):

  superheroes = """name,alignment,gender,publisher
Magneto,bad,male,Marvel
Storm,good,female,Marvel
Mystique,bad,female,Marvel
Batman,good,male,DC
Joker,bad,male,DC
Catwoman,bad,female,DC
Hellboy,good,male,DarkHorseComics
"""
  superheroes = DplyFrame(pd.read_csv(StringIO(superheroes)))

  publishers = """publisher,yr_founded
DC,1934
Marvel,1939
Image,1992
"""
  publishers = DplyFrame(pd.read_csv(StringIO(publishers)))
  publishers_2 = publishers.rename(columns={'publisher': 'publisher_2', 'yr_founded': 'yr_founded_2'})

  def test_semi_join_1(self):
    j_test = self.superheroes >> semi_join(self.publishers)
    j_pd = DplyFrame(pd.read_csv(StringIO("""name,alignment,gender,publisher
Magneto,bad,male,Marvel
Storm,good,female,Marvel
Mystique,bad,female,Marvel
Batman,good,male,DC
Joker,bad,male,DC
Catwoman,bad,female,DC""")))
    self.assertTrue(j_test.equals(j_pd))
    # names don't matter
    j_test = self.superheroes >> semi_join(self.publishers_2, by=[('publisher', 'publisher_2')])
    self.assertTrue(j_test.equals(j_pd))
    # works in normal form
    j1 = semi_join(self.superheroes, self.publishers)
    j2 = self.superheroes >> semi_join(self.publishers)
    self.assertTrue(j1.equals(j2))
    # works on grouped data
    j1 = self.superheroes >> group_by(X.publisher) >> semi_join(self.publishers)
    # compares equal
    self.assertTrue(j1.equals(j2))
    # but don't group the same
    self.assertTrue(j1._grouped_on == ['publisher'])
    self.assertTrue(j2._grouped_on is None)


  def test_anti_join_1(self):
    j_test = self.superheroes >> anti_join(self.publishers)
    j_pd = DplyFrame(pd.read_csv(StringIO("""index,name,alignment,gender,publisher
6,Hellboy,good,male,DarkHorseComics""")).set_index(['index']))
    j_pd.index.name = None
    self.assertTrue(j_test.equals(j_pd))
    j_test = self.superheroes >> anti_join(self.publishers_2, by=[('publisher', 'publisher_2')])
    # names don't matter
    self.assertTrue(j_test.equals(j_pd))
    # works in normal form
    j1 = anti_join(self.superheroes, self.publishers)
    j2 = self.superheroes >> anti_join(self.publishers)
    self.assertTrue(j1.equals(j2))
    # works on grouped data
    j1 = self.superheroes >> group_by(X.publisher) >> anti_join(self.publishers)
    # compares equal
    self.assertTrue(j1.equals(j2))
    # but don't group the same
    self.assertTrue(j1._grouped_on == ['publisher'])
    self.assertTrue(j2._grouped_on is None)

  def test_semi_join_2(self):
    j_test = self.publishers >> semi_join(self.superheroes)
    j_pd = DplyFrame(pd.read_csv(StringIO("""publisher,yr_founded
DC,1934
Marvel,1939""")))
    self.assertTrue(j_test.equals(j_pd))
    # names don't matter
    j_test = self.publishers_2 >> semi_join(self.superheroes, by=[('publisher_2', 'publisher')])
    j_pd = DplyFrame(pd.read_csv(StringIO("""publisher_2,yr_founded_2
DC,1934
Marvel,1939""")))
    self.assertTrue(j_test.equals(j_pd))

  def test_anti_join_2(self):
    j_test = self.publishers >> anti_join(self.superheroes)
    j_pd = DplyFrame(pd.read_csv(StringIO("""index,publisher,yr_founded
2,Image,1992""")).set_index(['index']))
    self.assertTrue(j_test.equals(j_pd))
    # names don't matter
    j_test = self.publishers_2 >> anti_join(self.superheroes, by=[('publisher_2', 'publisher')])
    j_pd = DplyFrame(pd.read_csv(StringIO("""index,publisher_2,yr_founded_2
2,Image,1992""")).set_index(['index']))
    self.assertTrue(j_test.equals(j_pd))

  a = DplyFrame(pd.read_csv(StringIO("""x,y
1,1
1,2
2,3
3,4""")))
  b = DplyFrame(pd.read_csv(StringIO("""x,z
1,1
2,2
2,3
4,4""")))

  def test_semi_join_dplyr_1(self):
    j_test_1 = self.a >> semi_join(self.b)
    j_test_2 = self.b >> semi_join(self.a)
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""x,y
1,1
1,2
2,3""")))
    j_pd_2 = DplyFrame(pd.read_csv(StringIO("""x,z
1,1
2,2
2,3""")))
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))

  def test_anti_join_dplyr_1(self):
    j_test_1 = self.a >> anti_join(self.b)
    j_test_2 = self.b >> anti_join(self.a)
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""index,x,y
3,3,4""")).set_index(['index']))
    j_pd_1.index.name = None
    j_pd_2 = DplyFrame(pd.read_csv(StringIO("""index,x,z
3,4,4""")).set_index(['index']))
    j_pd_2.index.name = None
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))

  c = DplyFrame(pd.read_csv(StringIO("""x,y,a
1,1,1
1,1,2
2,2,3
3,3,4""")))
  d = DplyFrame(pd.read_csv(StringIO("""x,y,b
1,1,1
2,2,2
2,2,3
4,4,4""")))

  def test_semi_join_dplyr_2(self):
    # bivariate keys
    j_test_1 = self.c >> semi_join(self.d)
    j_test_2 = self.d >> semi_join(self.c)
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""x,y,a
1,1,1
1,1,2
2,2,3""")))
    j_pd_2 = DplyFrame(pd.read_csv(StringIO("""x,y,b
1,1,1
2,2,2
2,2,3""")))
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))
    # include column names
    j_test_1 = self.c >> semi_join(self.d, by=['x', 'y'])
    j_test_2 = self.d >> semi_join(self.c, by=['x', 'y'])
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))
    # use different column names
    alt_c = self.c.rename(columns={'x': 'x_2'})
    j_test_1 = alt_c >> semi_join(self.d, by=[('x_2', 'x'), 'y'])
    j_test_2 = self.d >> semi_join(alt_c, by=[('x', 'x_2'), 'y'])
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""x_2,y,a
1,1,1
1,1,2
2,2,3""")))
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))

  def test_anti_join_dplyr_2(self):
    # bivariate keys
    j_test_1 = self.c >> anti_join(self.d)
    j_test_2 = self.d >> anti_join(self.c)
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""index,x,y,a
3,3,3,4""")).set_index(['index']))
    j_pd_1.index.name = None
    j_pd_2 = DplyFrame(pd.read_csv(StringIO("""index,x,y,b
3,4,4,4""")).set_index(['index']))
    j_pd_2.index.name = None
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))
    # use column names
    j_test_1 = self.c >> anti_join(self.d, by=['x', 'y'])
    j_test_2 = self.d >> anti_join(self.c, by=['x', 'y'])
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))
    # use different column names
    alt_c = self.c.rename(columns={'x': 'x_2'})
    j_test_1 = alt_c >> anti_join(self.d, by=[('x_2', 'x'), 'y'])
    j_test_2 = self.d >> anti_join(alt_c, by=[('x', 'x_2'), 'y'])
    j_pd_1 = DplyFrame(pd.read_csv(StringIO("""index,x_2,y,a
3,3,3,4""")).set_index(['index']))
    j_pd_1.index.name = None
    self.assertTrue(j_test_1.equals(j_pd_1))
    self.assertTrue(j_test_2.equals(j_pd_2))



if __name__ == '__main__':
  unittest.main()
