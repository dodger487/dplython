import operator

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
  In : diamonds >> sift(X.carat > 4) >> select(X.carat, X.price)
  Out:
         carat  price
  25998   4.01  15223
  25999   4.01  15223
  27130   4.13  17329
  27415   5.01  18018
  27630   4.50  18531

  The special Later name, "_" will refer to the entire DataFrame. For example, 
  In: diamonds >> sample_n(6) >> select(X.carat, X.price) >> X._.T
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
    "__ror__": ReverseBinaryOperator("|", 7),
    "__xor__": BinaryOperator("^", 8),
    "__rxor__": ReverseBinaryOperator("^", 8),
    "__and__": BinaryOperator("&", 9),
    "__rand__": BinaryOperator("&", 9),
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
    "__rdiv__": ReverseBinaryOperator("/", 12),
    "__truediv__": BinaryOperator("/", 12),
    "__rtruediv__": ReverseBinaryOperator("/", 12),
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


