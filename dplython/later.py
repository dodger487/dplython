from six.moves import reduce

import operator


# reversible_operators = [
#     ["__add__", "__radd__"],
#     ["__sub__", "__rsub__"],
#     ["__mul__", "__rmul__"],
#     ["__floordiv__", "__rfloordiv__"],
#     ["__div__", "__rdiv__"],
#     ["__truediv__", "__rtruediv__"],
#     ["__mod__", "__rmod__"],
#     ["__divmod__", "__rdivmod__"],
#     ["__pow__", "__rpow__"],
#     ["__lshift__", "__rlshift__"],
#     ["__rshift__", "__rrshift__"],
#     ["__and__", "__rand__"],
#     ["__or__", "__ror__"],
#     ["__xor__", "__rxor__"],
# ]

# normal_operators = [
#     "__abs__", "__concat__", "__contains__", "__delitem__", "__delslice__",
#     "__eq__", "__file__", "__ge__", "__getitem__", "__getslice__", "__gt__", 
#     "__iadd__", "__iand__", "__iconcat__", "__idiv__", "__ifloordiv__", 
#     "__ilshift__", "__imod__", "__imul__", "__index__", "__inv__", "__invert__",
#     "__ior__", "__ipow__", "__irepeat__", "__irshift__", "__isub__", 
#     "__itruediv__", "__ixor__", "__le__", "__lt__", "__ne__", "__neg__",
#     "__not__", "__package__", "__pos__", "__repeat__", "__setitem__",
#     "__setslice__", "__radd__", "__rsub__", "__rmul__", "__rfloordiv__",
#     "__rdiv__", "__rtruediv__", "__rmod__", "__rdivmod__", "__rpow__", 
#     "__rlshift__",  "__rand__",  "__ror__",  "__rxor__",  # "__rrshift__",
# ]

class Operator(object):

  def __init__(self, name, symbol, precedence):
    self.name = name
    self.format_string = self.get_format_string(symbol)
    self.precedence = precedence

  def apply(self, *args, **kwargs):
    func = getattr(operator, self.name)
    return func(*args)

  def format_exp(self, exp):
    if isinstance(exp, Later) and exp._step.precedence < self.precedence:
      return "({0})".format(str(exp))
    else:
      return str(exp)

  def format_operation(self, *args, **kwargs):
    return self.format_string.format(*args)


class BinaryOperator(Operator):

  def apply(self, *args, **kwargs):
    func = getattr(operator, self.name)
    return func(*args)

  def get_format_string(self, symbol):
    return "{0} " + symbol + " {1}"


class ReverseBinaryOperator(Operator):
  
  def apply(self, *args, **kwargs):
    # Note: this is a binary operator, so we know there are only two args.
    # This is the reverse, so we have to flip them.
    func = getattr(operator, self.name)
    return func(args[1], args[0])

  def get_format_string(self, symbol):
    return "{1} " + symbol + " {0}"


class UnaryOperator(Operator):

  def get_format_string(self, symbol):
    return symbol + "{0}"


class Step(object):

  def __init__(self, precedence, *args, **kwargs):
    self.precedence = precedence
    self.args = args
    self.kwargs = kwargs

  def format_operation(self, obj):
    raise NotImplementedError

  def format_exp(self, exp):
    if isinstance(exp, Later) and exp._step.precedence < self.precedence:
      return "({0})".format(str(exp))
    else:
      return str(exp)

  def evaluated_args(self, df, **kwargs):
    return [(arg.evaluate(df, **kwargs) if isinstance(arg, Later) else arg)
            for arg in self.args]

  def evaluated_kwargs(self, df, **kwargs):
    return {k: v.evaluate(df, **kwargs) if isinstance(v, Later) else v
            for k, v in self.kwargs.items()}


class OperatorStep(Step):

  def __init__(self, operator, *args, **kwargs):
    self.operator = operator
    super(OperatorStep, self).__init__(operator.precedence, 
                                       *args, 
                                       **kwargs)

  def evaluate(self, previousResult, original, **kwargs):
    return self.operator.apply(previousResult, 
                               *self.evaluated_args(original, **kwargs),
                               **self.evaluated_kwargs(original, **kwargs))

  def format_operation(self, obj):
    formatted_args = [self.format_exp(arg) for arg in [obj] + list(self.args)]
    return self.operator.format_operation(*formatted_args, **self.kwargs)


class ArgStep(Step):

  def format_args(self):
      # We sort here because keyword arguments get arbitrary ordering inside the 
      # function call. Support PEP 0468 to help fix this issue!
      # https://www.python.org/dev/peps/pep-0468/
      kwarg_strs = sorted(["{0}={1}".format(k, _addQuotes(v)) 
                            for k, v in self.kwargs.items()])
      arg_strs = list(map(str, self.args))
      full_str = ", ".join(arg_strs + kwarg_strs)
      return full_str


class CallStep(ArgStep):

  def __init__(self, *args, **kwargs):
    super(CallStep, self).__init__(15, *args, **kwargs)

  def format_operation(self, obj):
    return "{0}({1})".format(obj,
                             self.format_args())

  def evaluate(self, previousResult, original, **kwargs):
    return previousResult.__call__(*self.evaluated_args(original, **kwargs),
                                   **self.evaluated_kwargs(original, **kwargs))


class FunctionStep(ArgStep):

  def __init__(self, func, *args, **kwargs):
    self.func = func
    super(FunctionStep, self).__init__(15, *args, **kwargs)

  def format_operation(self, obj):
    return "{0}({1})".format(self.func.__name__,
                             self.format_args())

  def evaluate(self, previousResult, original, **kwargs):
    return self.func(*self.evaluated_args(original, **kwargs),
                     **self.evaluated_kwargs(original, **kwargs))


class AttributeStep(Step):

  def __init__(self, attr):
    self.attr = attr
    super(AttributeStep, self).__init__(15)

  def format_operation(self, obj):
    return "{0}.{1}".format(self.format_exp(obj),
                            self.attr)

  def evaluate(self, previousResult, original, **kwargs):
    return getattr(previousResult, self.attr)


class BracketStep(Step):

  def __init__(self, key):
    self.key = key
    super(BracketStep, self).__init__(15)

  def format_operation(self, obj):
    return '{0}[{1}]'.format(self.format_exp(obj),
                             self.format_arg())

  def format_arg(self):
    if isinstance(self.key, slice):
      _str = ""
      (start, stop, step) = (self.key.start, self.key.stop, self.key.step)
      if start is not None:
        _str += str(start)
      _str += ":"
      if stop is not None:
        _str += str(stop)
      if step is not None:
        _str += ":" + str(step)
      return _str
    elif isinstance(self.key, str):
      return '"{0}"'.format(self.key)
    else:
      return str(self.key)

  def evaluate(self, previousResult, original, **kwargs):
    return operator.getitem(previousResult, self.key)


class SliceStep(BracketStep):
  """Necessary for compatibility."""

  def format_operation(self, obj, args, kwargs):
    return "{0}[{1}]".format(self.format_exp(obj),
                             self.format_arg(slice(args[0], args[1])))


class IdentityStep(Step):

  def __init__(self):
    super(IdentityStep, self).__init__(17)

  def evaluate(self, previousResult, original, **kwargs):
    return previousResult

  def format_operation(self, obj):
    return "X._"


OPERATORS = {
  # see https://docs.python.org/2/reference/expressions.html#operator-precedence
  "__lt__": BinaryOperator("__lt__", "<", 6),
  "__le__": BinaryOperator("__le__", "<=", 6),
  "__eq__": BinaryOperator("__eq__", "==", 6),
  "__ne__": BinaryOperator("__ne__", "!=", 6),
  "__ge__": BinaryOperator("__ge__", ">=", 6),
  "__gt__": BinaryOperator("__gt__", ">", 6),
  "__or__": BinaryOperator("__or__", "|", 7),
  "__ror__": ReverseBinaryOperator("__or__", "|", 7),
  "__xor__": BinaryOperator("__xor__", "^", 8),
  "__rxor__": ReverseBinaryOperator("__xor__", "^", 8),
  "__and__": BinaryOperator("__and__", "&", 9),
  "__rand__": ReverseBinaryOperator("__and__", "&", 9),
  "__lshift__": BinaryOperator("__lshift__", "<<", 10),
  "__rlshift__": ReverseBinaryOperator("__lshift__", "<<", 10),
  "__rshift__": BinaryOperator("__rshift__", ">>", 10),
  "__rrshift__": ReverseBinaryOperator("__rshift__", ">>", 10),
  "__add__": BinaryOperator("__add__", "+", 11),
  "__radd__": ReverseBinaryOperator("__add__", "+", 11),
  "__sub__": BinaryOperator("__sub__", "-", 11),
  "__rsub__": ReverseBinaryOperator("__sub__", "-", 11),
  "__mul__": BinaryOperator("__mul__", "*", 12),
  "__rmul__": ReverseBinaryOperator("__mul__", "*", 12),
  "__div__": BinaryOperator("__div__", "/", 12),
  "__rdiv__": ReverseBinaryOperator("__div__", "/", 12),
  "__truediv__": BinaryOperator("__truediv__", "/", 12),
  "__rtruediv__": ReverseBinaryOperator("__truediv__", "/", 12),
  "__floordiv__": BinaryOperator("__floordiv__", "//", 12),
  "__rfloordiv__": ReverseBinaryOperator("__floordiv__", "//", 12),
  "__mod__": BinaryOperator("__mod__", "%", 12),
  "__rmod__": ReverseBinaryOperator("__mod__", "%", 12),
  "__neg__": UnaryOperator("__neg__", "-", 13),
  "__pos__": UnaryOperator("__pos__", "+", 13),
  "__invert__": UnaryOperator("__invert__", "~", 13),
  "__pow__": BinaryOperator("__pow__", "**", 14),
  "__rpow__": ReverseBinaryOperator("__pow__", "**", 14),
}

def add_operator_hooks(cls):

  def get_hook(name):
    operator = OPERATORS[name]
    def op_hook(self, *args, **kwargs):
      return Later(OperatorStep(operator, *args, **kwargs), self)
    return op_hook

  for name in OPERATORS:
    setattr(cls, name, get_hook(name))

  return cls


def _addQuotes(item):
  return '"' + item + '"' if isinstance(item, str) else item


@add_operator_hooks
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

  def __init__(self, step, previous, name = None):
    if isinstance(step, IdentityStep) or name is not None:
      self._queue = []
    else:
      self._queue = previous._queue
    self._step = step
    self._queue.append(step)
    self._previous = previous
    self._name = name

  def evaluate(self, previousResult, original=None, special=None):
    if special not in {None, "transform", "agg"}:
      raise Exception("Special must be one of None, 'transform', or 'agg'.")

    original = original if original is not None else previousResult

    if (original._grouped_self
        and special 
        and not isinstance(self._queue[0], FunctionStep)):
      name = GetName(self)

      # TODO: Rewrite this, this is terrible.
      go_to_index = len(self._queue)
      for idx, item in enumerate(self._queue):
        if isinstance(item, OperatorStep):
          go_to_index = idx
          break

      transform_input = lambda x: reduce(
          lambda prevResult, f: f.evaluate(prevResult, original, special=special),
          self._queue[1:go_to_index],
          x
      )
      if special == "transform":
        out = original._grouped_self[name].transform(transform_input)
      elif special == "agg":
        out = original._grouped_self[name].agg(transform_input)
      out = reduce(lambda prevResult, f: f.evaluate(prevResult, original, special=special),
                      self._queue[go_to_index:],
                      out)
      return out
    else:
      output = reduce(lambda prevResult, f: f.evaluate(prevResult, original),
                      self._queue,
                      original)
      return output
    
  def __str__(self):
    return self._step.format_operation(self._previous)

  def __repr__(self):
    return str(self)

  def __getattr__(self, attr):
    return Later(AttributeStep(attr), self)

  def __getitem__(self, attr):
    return Later(BracketStep(attr), self)

  def __call__(self, *args, **kwargs):
    return Later(CallStep(*args, **kwargs), self)

  def __rrshift__(self, df):
    otherDf = DplyFrame(df.copy(deep=True))
    return self.evaluate(otherDf)


def CreateLaterFunction(fcn, *args, **kwargs):
  return Later(FunctionStep(fcn, *args, **kwargs), IdentityStep(), "function")

def DelayFunction(fcn):
  def DelayedFcnCall(*args, **kwargs):
    # Check to see if any args or kw are Later. If not, return normal fcn.
    if (len([a for a in args if isinstance(a, Later)]) == 0 and
        len([v for k, v in kwargs.items() if isinstance(v, Later)]) == 0):
      return fcn(*args, **kwargs)
    else:
      return CreateLaterFunction(fcn, *args, **kwargs)

  return DelayedFcnCall


def GetName(later):
  while not later._name:
    later = later._previous
  return later._name


class Manager(object):
  """Object which helps create a delayed computational unit.

  Typically will be set as a global variable X.
  X.foo will refer to the "foo" column of the DataFrame in which it is later
  applied. 

  Manager can be used in two ways: 
  (1) attribute notation: X.foo
  (2) item notation: X["foo"]

  Attribute notation is preferred but item notation can be used in cases where 
  column names contain characters on which python will choke, such as spaces, 
  periods, and so forth.
  """
  def __getattr__(self, attr):
    if attr == "_":
      return Later(IdentityStep(), self, "_")
    else:
      return Later(AttributeStep(attr), self, attr)

  def __getitem__(self, key):
    if key == "_":
      return Later(IdentityStep(), self, "_")
    else:
      return Later(BracketStep(key), self, key)

  def evaluate(self, previousResult, original, **kwargs):
    return original

  def __str__(self):
    return "X"

  def __repr__(self):
    return str(self)


X = Manager()

