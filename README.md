## Dplython: Dplyr for Python

Welcome to Dplython: Dplyr for Python.
The goal of this project is to implement the functionality of the R package Dplyr on top of Python's pandas.

* Dplyr: [Click here](https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html)
* Pandas: [Click here](http://pandas.pydata.org/pandas-docs/stable/10min.html)

This is version 0.0.1. 
It's experimental and subject to change.

```python
from dplyr import *

diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))

diamonds >> select(X.cut) >> head()
"""
Out:
         cut
0      Ideal
1    Premium
2       Good
3    Premium
4       Good
5  Very Good
6  Very Good
7  Very Good
8       Fair
9  Very Good
"""

diamonds >> select(X.carat, X.cut, X.price) >> head(5)
"""
Out:
   carat        cut  price
0   0.23      Ideal    326
1   0.21    Premium    326
2   0.23       Good    327
3   0.29    Premium    334
4   0.31       Good    335
"""

diamonds >> dfilter(X.carat > 4) >> select(X.carat, X.cut, X.depth, X.price)
"""
Out:
       carat      cut  depth  price
25998   4.01  Premium   61.0  15223
25999   4.01  Premium   62.5  15223
27130   4.13     Fair   64.8  17329
27415   5.01     Fair   65.5  18018
27630   4.50     Fair   65.8  18531
"""

(diamonds >> 
  mutate(carat_bin=X.carat.round()) >> 
  group_by(X.cut, X.carat_bin) >> 
  summarize(avg_price=X.price.mean()))
"""
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

# If you have column names that don't work as attributes, you can use an 
# alternate "get item" notation with X.
diamonds["column w/ spaces"] = range(len(diamonds))
diamonds >> select(X["column w/ spaces"]) >> head()
"""
Out:
   column w/ spaces
0                 0
1                 1
2                 2
3                 3
4                 4
5                 5
6                 6
7                 7
8                 8
9                 9
"""

# It's possible to pass the entire dataframe using X._ 
diamonds >> sample_n(6) >> select(X.carat, X.price) >> X._.T
"""
Out:
         18966    19729   9445   49951    3087    33128
carat     1.16     1.52     0.9    0.3     0.74    0.31
price  7803.00  8299.00  4593.0  540.0  3315.00  816.00
"""

# To pass the DataFrame or columns into functions, apply @DelayFunction
@DelayFunction
def PairwiseGreater(series1, series2):
  index = series1.index
  newSeries = pandas.Series([max(s1, s2) for s1, s2 in zip(series1, series2)])
  newSeries.index = index
  return newSeries

diamonds >> PairwiseGreater(X.x, X.y)


# Passing entire dataframe into ggplot
from ggplot import *
ggplot = DelayFunction(ggplot)  # Simple installation
diamonds = DplyFrame(pandas.read_csv('./diamonds.csv'))  # Masked in ggplot pkg
(diamonds >> ggplot(aes(x="carat", y="price", color="cut"), data=X._) + 
  geom_point() + facet_wrap("color"))
(diamonds >>
  dfilter((X.clarity == "I1") | (X.clarity == "IF")) >> 
  ggplot(aes(x="carat", y="price", color="color"), X._) + 
    geom_point() + 
    facet_wrap("clarity"))
```

This is very new and I'm matching changes. 
Let me know if you'd like to see a feature or think there's a better way I can do something.

Other approaches:
* [pandas-ply](http://pythonhosted.org/pandas-ply/)
