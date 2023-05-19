from enum import Enum, auto
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Iterator, Literal
from dataclasses import dataclass, field
from copy import deepcopy
from sys import argv as __sys_argv
from os.path import exists, sep
from functools import reduce
from re import compile as regxp_compile

def ItSelf(x): return x

class Array(list):
    def __init__(self, args):               super().__init__(args)
    def __iter__(self):              return super().__iter__()
    def __len__(self) -> int:        return super().__len__()
    def __getitem__(self, key):      return Array(super().__getitem__(key))
    def __setitem__(self, key, val): return Array(super().__setitem__(key, val))
    def __delitem__(self, key):      return super().__delitem__(key)
    def __contains__(self, item):    return super().__contains__(item)
    def __reversed__(self):          return super().__reversed__()
    def __str__(self):               return "{" + ", ".join(str(x) for x in self) + "}"
    def __repr__(self):              return "Array(" + ", ".join(x.__repr__() for x in self) + ")"
    def __add__(self, other):        return Array(list(self) + list(other))
    def forEach(self, func):
        for x in self: func(x)
    def sorted(self, key=None, reverse=False): self.sort(key=key, reverse=reverse); return self
    def map(self, func):             return Array([func(x) for x in self])
    def filter(self, func):          return Array([x for x in self if func(x)])
    def withItems(self, *items):     return self.extend(items); return self
    def collect(self, target, func): return reduce(func, self, target)
    def has(self, item):             return item in self

sys_argv: Array = Array(__sys_argv)
del __sys_argv

__candy_location    : str                                   = __file__
__candy_interpreter : Literal["PY", "EXE"]                  = __candy_location.split(".")[-1].upper()
__candy_file        : Union[str, None]                      = sys_argv.filter (lambda loc: loc.endswith(".candy"))\
                                                                      .map    (lambda loc: loc.replace("\\", sep).replace("/", sep))\
                                                                      .collect(None, lambda target, loc: loc if target == None else target)
__candy_mode        : Literal["RUN", "COMPILE", "TERMINAL"] = "TERMINAL" if __candy_file == None else "COMPILE" if "-compile" in sys_argv else "RUN"
__candy_setting     : Dict[str, Any]                        = {"timer": "-timer" in sys_argv}
if __candy_setting["timer"]:
    from time import time as __time
if type(__candy_file) == str and not exists(__candy_file): 
    print("Unable to find file \""+__candy_file+"\"")
    exit(1)
del exists, sep

EXPRESSION_SPACE = "[ |\t]+"
EXPRESSION_NAME = "[a-z|A-Z|_][\w|\d|_]*"
EXPRESSION_VARIABLE = "\$" + EXPRESSION_NAME
EXPRESSION_INTEGER = "-?[1-9][0-9]*|-?0"
EXPRESSION_DECIMAL = "-?[0-9]*?.[0-9]+"
COMPILED_EXPRESSION_EMPTY        = regxp_compile("[ |\t]+")
COMPILED_EXPRESSION_VARIABLE     = regxp_compile(EXPRESSION_VARIABLE)
COMPILED_EXPRESSION_STRING       = regxp_compile("\"[^\"\n]*\"")
COMPILED_EXPRESSION_INTEGER      = regxp_compile(EXPRESSION_INTEGER)
COMPILED_EXPRESSION_DECIMAL      = regxp_compile(EXPRESSION_DECIMAL)
COMPILED_EXPRESSION_BOOLEAN      = regxp_compile("(true|false)")
COMPILED_EXPRESSION_SET          = regxp_compile("set"+EXPRESSION_SPACE+"("+EXPRESSION_VARIABLE+")"+EXPRESSION_SPACE+"to"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_CALL_FUNC    = regxp_compile("("+EXPRESSION_NAME+")[ |\t]*([^\n]*)")
COMPILED_EXPRESSION_WORD         = regxp_compile("\\w+")
COMPILED_EXPRESSION_BLOCKHOLDER  = regxp_compile("(if|elif|else|for|while|loop|function)"+"([^\n]*):")
COMPILED_EXPRESSION_LOOP_VALUE   = regxp_compile("loop-value-\\d+")
COMPILED_EXPRESSION_LOOP_CONTROL = regxp_compile("(break|continue)")
del EXPRESSION_SPACE, EXPRESSION_NAME, EXPRESSION_VARIABLE, EXPRESSION_INTEGER, EXPRESSION_DECIMAL
del regxp_compile

OPERATOR_TOKENS = ("+", "-", "*", "/", "%", "//", "**", 
                   "==", "!=", "<", ">", "<=", ">=",)

@dataclass(frozen=True)
class Void():
    data: str

VOID = Void("VOID")
MISSING_ARG = Void("MISSING_ARG")
RELAY_SET_RETURN = Void("RELAY_SET_RETURN")
RELAY_LOOP_BREAK = Void("RELAY_LOOP_BREAK")
RELAY_LOOP_CONTINUE = Void("RELAY_LOOP_CONTINUE")

class StackType(Enum):
    ROOT = 0
    FUNCTION = 1
    #MODULE = 2

@dataclass(frozen=True)
class ColRange():
    start: int
    end: int

@dataclass()
class LoopData():
    values: Dict[int, Any] = field(default_factory=dict)
    next_index: int = 0

@dataclass()
class Variable():
    value: Any

@dataclass()
class Stack():
    loc: str # file
    line: int
    type: StackType
    col: ColRange
    master: Union["Stack", None] = None
    loop: LoopData = field(default_factory=LoopData)
    data: Dict[str, Variable] = field(default_factory=dict)
    def getData(self, name: str):
        if name in self.data: return self.data[name].value
        elif self.master != None: return self.master.getData(name)
        else: raise NameError("name '"+name+"' is not defined")

class StackSet():
    def __init__(self, loc:str="Candy", __global: Dict[str, Any] = None) -> None:
        self.stacks: List[Stack] = [Stack(loc, 0, type=StackType.ROOT, col=ColRange(0, 0))]
        __global = {} if __global == None else __global
        for key, value in _dict.items():
            if not (key in __global): __global[key] = Variable(value)
        self.stacks[-1].data = __global
    def enter(self, newStack: Stack) -> None:
        self.stacks.append(newStack)
        if len(self.stacks): self.stacks[-1].master = self.stacks[-2]
    def exit(self) -> None:
        if len(self.stacks) <= 1: raise Exception("StackSet: cannot exit from root stack")
        del self.stacks[-1]
    @property
    def last(self) -> Stack: return self.stacks[-1]
    @property
    def history(self) -> Tuple[Stack, ...]: return tuple(self.stacks)
    def getKeys(self)->List[str]:
        return list(reduce(lambda a, b: a.union(b), [set(stack.data.keys()) for stack in self.stacks]))




@dataclass(frozen=True)
class Argument():
    name:str
    auto:Any=MISSING_ARG

class Function():
    def __init__(self, name: str, tokens: List[Union[str, Argument]], func: Callable):
        self.name: str = name
        self.tokens: Set[str] = set([token for token in tokens if type(token) == str])
        self.__args: List[Argument] = [token for token in tokens if type(token) == Argument]
        self.less_args = len([arg for arg in self.__args if arg.auto is MISSING_ARG])
        self.func = func
    def __call__(self, variables: List[Any], _file: str, _stack: "StackSet") -> Any:
        if len(self.__args) != len(variables): pass # error
        if len(variables) < self.less_args: raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        variables += (MISSING_ARG,) * (len(self.__args) - len(variables))
        localDict: Dict[str, Any] = {arg_slot.name:Variable(var if type(var) != Void else arg_slot.auto) for arg_slot, var in zip(self.__args, variables)}
        if MISSING_ARG in localDict.values(): raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        innerStack = (lambda lastStack: Stack(_file, lastStack.line, type=StackType.FUNCTION, col=lastStack.col, master=lastStack, data=localDict))(_stack.last)
        _stack.enter(innerStack)
        v = self.func(localDict, _file, _stack)
        _stack.exit()
        return v

def create_py_function(func_name:str, tokens: List[Union[str, Argument]], func: Callable, auto_return: bool = True):
    def __func (arguments, _, __): 
        __relay = func(**{k:v.value for k, v in arguments.items()})
        if auto_return and __relay is None: return VOID
        return __relay
    return Function(func_name, tokens, __func)

_dict = {}
def __print(text): print(text, end="")
def __println(text): print(text)
def __combine(t1, t2): return str(t1)+str(t2)
def __repeat(text, count): return str(text)*count
class Forever(Iterator):
    def __init__(self) -> None: self.index = 0
    def __next__(self) -> int: self.index += 1; return self.index
_dict["print"]    = create_py_function("print",    ["of", Argument("text")], __print)
_dict["println"]  = create_py_function("println",  ["of", Argument("text")], __println)
_dict["combine"]  = create_py_function("combine",  ["of", Argument("t1"), "and", Argument("t2")], __combine)
_dict["repeat"]   = create_py_function("repeat",   ["of", Argument("text"), "for", Argument("count"), "times"], __repeat)
_dict["range"]    = create_py_function("range",    ["of", Argument("start"), "to", Argument("end"), "by", Argument("sep", auto=1)], lambda start, end, sep: range(start, end, sep))
_dict["asString"] = create_py_function("asString", ["of", Argument("text")], lambda text: str(text))
_dict["asInt"]    = create_py_function("asInt",    ["of", Argument("text")], lambda text: int(text))
_dict["asFloat"]  = create_py_function("asFloat",  ["of", Argument("text")], lambda text: float(text))
_dict["asBool"]   = create_py_function("asBool",   ["of", Argument("text")], lambda text: bool(text))
_dict["asList"]   = create_py_function("asList",   ["of", Argument("text")], lambda text: list(text))
_dict["length"]   = create_py_function("length",   ["of", Argument("text")], lambda text: len(text))
_dict["forever"]  = create_py_function("forever",  [], Forever)





class CodeType(Enum):
    #System (0~)
    EMPTY = 0
    PARENTHESIS = 10

    #Eval (100~)
    #Eval - Datas (100~)
    EVAL_INTEGER = 100
    EVAL_DECIMAL = 101
    EVAL_STRING = 102
    EVAL_BOOLEAN = 103
    EVAL_VARIABLE = 104
    EVAL_RANGE = 105

    EVAL_LOOP_VALUE = 110

    #Eval - Work (120~)
    EVAL_EXECUTE = 120
    EVAL_SET = 121
    EVAL_CALC = 122

    #Exec (200~)
    #Exec - Condition (200~)
    EXEC_CONDITION_IF = 200
    EXEC_CONDITION_ELIF = 201
    EXEC_CONDITION_ELSE = 202

    #Exec - Loop (210~)
    EXEC_LOOP_WHILE = 210
    EXEC_LOOP_LOOP = 211

    #Eval - Loop - Flow Control (215~)
    EXEC_LOOP_CONTINUE = 215
    EXEC_LOOP_BREAK = 216

    #Exec - Function (220~)
    EXEC_FUNCTION_DEFINE = 220
    EXEC_FUNCTION_SHARE = 221
    EXEC_FUNCTION_RETURN = 222

class Code():
    def __init__(self, codeType: CodeType, col: ColRange, **kw: Dict[str, Any]) -> None:
        self.codeType = codeType
        self.col = col
        self.kw = kw
    @property
    def is_eval(self): return 200 > self.codeType.value >= 100
    @property
    def is_exec(self): return 300 > self.codeType.value >= 200
    @property
    def is_dataholder(self): return 300 > self.codeType.value >= 250
    def __str__(self) -> str: return f"Code::{self.codeType} (at {self.col}) [{self.kw}]"

class LineTree():
    def __init__(self, line: int, code: Code, master: Union[Void, "LineTree"] = VOID):
        self.line = line
        self.code = code
        self.master = master
        self.childs: List["LineTree"] = []
    def add_child(self, child: "LineTree") -> None:
        child.master = self
        self.childs.append(child)
    def __str__(self) -> str: return "["+str(self.line)+"]"+"( "+", ".join([str(l) for l in self.childs])+" )"

def create_function(name: str, parameters: List[Union[str, Argument]], code: List[LineTree]) -> Function:
    def __func(arguments: Dict[str, Any], __file: str, __stack: StackSet):
        __relay = _run_tree(code, __file, __stack)
        return __relay
    return Function(name, parameters, __func)

# class Module():
#     def __init__(self, name: str, tree: List[LineTree], colmap: ColMapper) -> None:
#         self.name = name
#         self.tree = tree
#         self.colmap = colmap

class _EvalException(Exception):
    def __init__(self, exception: Exception, col: ColRange) -> None:
        self.exception = exception
        self.col = col

class HandledException(Exception):
    def __init__(self, exception: _EvalException, stack: StackSet) -> None:
        self.exception = exception.exception
        self.stack = stack
        self.col = exception.col
        
        
class SyntaxResultType(Enum):
    Right = 0

    Identition = auto()
    UnexpectedSyntax = auto()
    MixedIdentition = auto()
    MultiTab = auto()
    UnexpectedIndent=auto()
    def toException(self): ...

class ExecuteResultType(Enum):
    Success = 0
    CompileError = 1
    RuntimeError = 2

class ExecuteResult():
    def __init__(self, result: ExecuteResultType, *args: Tuple[Any, ...]) -> None:
        self.result = result
        self.args = args
    def __str__(self) -> str: return "ExecuteResult [%s]%s" % (self.result, "".join(["\n["+str(l)+"] "+str(arg) for l, arg in enumerate(self.args)]))


def match_parenthesis(__source: str):
    if not (__source.startswith("(") and __source.endswith(")")): return False
    __stack = 0
    is_in_string = False
    for __index, __char in enumerate(__source):
        if __char == '"': is_in_string = not is_in_string
        if is_in_string: continue
        if __char == "(": __stack += 1
        if __char == ")": __stack -= 1
        if __stack == 0 and __index != len(__source)-1: return False
    return __stack == 0

def splitArguments(__source: str) -> List[str]:
    split_result: List[List[Union[bool, str]]] = [[True, ""]]
    delta_bracket: int = 0
    isStrMode: bool = False
    isArgumentMode: bool = False
    for char in list(__source):
        if split_result[-1][1] == "": isArgumentMode = char == "("
        split_result[-1][0] = isArgumentMode
        if char == "\"":
            isStrMode = not isStrMode
            split_result[-1][1] += char
        elif isStrMode: split_result[-1][1] += char
        elif isArgumentMode:
            if char == "(": delta_bracket += 1
            if char == ")": delta_bracket -= 1
            split_result[-1][1] += char
            if delta_bracket == 0: split_result.append([True, ""])
        else:
            if char == " ": split_result.append([True, ""])
            else: split_result[-1][1] += char
    return list(map(tuple, filter(lambda v: v[1] != "", split_result)))

ESCAPING_MAPPER = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", "\"": "\""}
ESCAPING_MAPPER_KEYS = tuple(ESCAPING_MAPPER.keys())
def parse_string(__source: str) -> str:
    result = ""
    is_escape = False
    for char in __source[1:-1]:
        if is_escape:
            if char in ESCAPING_MAPPER_KEYS: 
                result += ESCAPING_MAPPER[char]
            else: result += "\\" + char
            is_escape = False
        elif char == "\\": is_escape = True
        else:              result += char
    return result

def checkExpressionSyntax(__source: str) -> bool:
    __source = __source.lstrip()
    if __source == "" or COMPILED_EXPRESSION_EMPTY.fullmatch(__source) != None: return True
    if (match_parenthesis(__source)): 
        return checkExpressionSyntax(__source[1:-1])
    
    

    # Eval
    elif __source.startswith("set"):
        match_define = COMPILED_EXPRESSION_SET.fullmatch(__source)
        return match_define != None and COMPILED_EXPRESSION_VARIABLE.fullmatch(match_define.group(1)) != None and checkExpressionSyntax(match_define.group(2))
    elif COMPILED_EXPRESSION_STRING.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_DECIMAL.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_INTEGER.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_VARIABLE.fullmatch(__source) != None: return True
    
    splits = splitArguments(__source)
    if len(splits) == 5 and splits[1][1] == "to" and splits[3][1] == "by" and checkExpressionSyntax(splits[0][1]) and checkExpressionSyntax(splits[2][1]) and checkExpressionSyntax(splits[4][1]): return True

    # Exec
    if (COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(__source) != None):
        match_blockholder = COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(__source)
        codeTypeString = match_blockholder.group(1)
        textgroup = match_blockholder.group(2).lstrip().rstrip()
        if codeTypeString in ("else"): 
            return len(textgroup) == 0
        elif codeTypeString in ("if", "elif", "while"):
            sa = splitArguments(textgroup)
            return len(sa) == 1 and sa[0][0] and checkExpressionSyntax(sa[0][1][1:-1])
        elif codeTypeString in ("loop"):
            sa = splitArguments(textgroup)
            return len(sa) == 1 and sa[0][0] and sa[0][1].startswith("(") and sa[0][1].endswith(")") and checkExpressionSyntax(sa[0][1][1:-1])
        elif codeTypeString in ("function"):
            sa = splitArguments(textgroup)
            return len(sa) >= 1 and (not sa[0][0]) and not (False in [(
                COMPILED_EXPRESSION_VARIABLE.fullmatch(token[1:-1]) != None if _type else COMPILED_EXPRESSION_WORD.fullmatch(token) != None
            ) for _type, token in sa[1:]])
    if COMPILED_EXPRESSION_LOOP_VALUE.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_LOOP_CONTROL.fullmatch(__source) != None: return True
        
    if COMPILED_EXPRESSION_CALL_FUNC.fullmatch(__source) != None:
        match_function = COMPILED_EXPRESSION_CALL_FUNC.fullmatch(__source)
        return COMPILED_EXPRESSION_WORD.fullmatch(match_function.group(1)) != None \
         and not (False in [(
            checkExpressionSyntax(token[1:-1]) if token.startswith("(") and token.endswith(")") else COMPILED_EXPRESSION_WORD.fullmatch(token) != None
        ) for _, token in splitArguments(match_function.group(2))])
    
    __IsValueList = [_type for _type, _ in splits]
    if (len(splits) >=3 and __IsValueList[:-1] and (not False in [__IsValueList[i] == (i % 2 == 0) for i in range(len(__IsValueList))])):
        return not False in [len(splits[i][1]) >= 2 
                                and splits[i][1][0] == "(" 
                                and splits[i][1][-1] == ")" 
                                and checkExpressionSyntax(splits[i][1][1:-1]) 
                             for i in range(len(splits)) 
                             if i % 2 == 0
                            ]
    return False

def checkSyntax(__source: str) -> Tuple[SyntaxResultType, Tuple[int, ...]]:
    first_not_empty_line = [(i, line) for i, line in enumerate(__source.split("\n")) if line != "" and COMPILED_EXPRESSION_EMPTY.fullmatch(line) == None]
    if len(first_not_empty_line) == 0: return (SyntaxResultType.Right, ())
    if first_not_empty_line[0][1][0] in (" ", "\t"): return (SyntaxResultType.UnexpectedIndent, (0, ))
    del first_not_empty_line

    __wrong_syntax_lines: Tuple[int] = tuple([i for i, line in enumerate(__source.split("\n")) if not checkExpressionSyntax(line)])
    if len(__wrong_syntax_lines) > 0: return (SyntaxResultType.UnexpectedSyntax, __wrong_syntax_lines)
    
    last_ident = 0
    lines = __source.split("\n")
    first_identify: Union[str, Void] = VOID
    for l, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")): first_identify, last_ident = VOID, 0
        else:
            looking_identify:str = ""
            code: str = ""
            for char in list(line):
                if len(code) != 0: code += char
                elif char == " " or char == "\t": looking_identify += char
                else: code = char
            if len(set(looking_identify)) == 2: return (SyntaxResultType.MixedIdentition, (l, ))
            if last_ident == 0:
                first_identify = looking_identify
                if "\t\t" in first_identify: return (SyntaxResultType.MultiTab, (l, ))
                last_ident = 1
            else:
                if set(first_identify) != set(looking_identify): return (SyntaxResultType.DifferentIdent, (l, ))
                last_ident = looking_identify.count(first_identify)
    return (SyntaxResultType.Right, ())

def _compile(expression: str, col: ColRange = None) -> Code:
    if col == None: col = ColRange(0, len(expression)-1)
    try:
        if expression == "": return Code(CodeType.EMPTY, deepcopy(col))
        elif match_parenthesis(expression): return Code(CodeType.PARENTHESIS, col, code=_compile(expression[1:-1], ColRange(col.start+1, col.end-1)))
        elif COMPILED_EXPRESSION_INTEGER     .fullmatch(expression) != None: return Code(CodeType.EVAL_INTEGER , deepcopy(col), value=int  (expression))
        elif COMPILED_EXPRESSION_DECIMAL     .fullmatch(expression) != None: return Code(CodeType.EVAL_DECIMAL , deepcopy(col), value=float(expression))
        elif COMPILED_EXPRESSION_STRING      .fullmatch(expression) != None: return Code(CodeType.EVAL_STRING  , deepcopy(col), value=parse_string (expression))
        elif COMPILED_EXPRESSION_BOOLEAN     .fullmatch(expression) != None: return Code(CodeType.EVAL_BOOLEAN , deepcopy(col), value=bool (expression))
        elif COMPILED_EXPRESSION_LOOP_VALUE  .fullmatch(expression) != None: return Code(CodeType.EVAL_LOOP_VALUE, deepcopy(col), index=int(expression.replace("loop-value-", "")))
        elif COMPILED_EXPRESSION_LOOP_CONTROL.fullmatch(expression) != None: return Code(CodeType.EXEC_LOOP_CONTINUE if expression=="continue" else CodeType.EXEC_LOOP_BREAK, deepcopy(col))
        __splits = splitArguments(expression)
        if len(__splits) == 5 and __splits[1][1] == "to" and __splits[3][1] == "by":
            s = col.start
            value_start = _compile(__splits[0][1], ColRange(s, s+len(__splits[0][1])-1))
            value_end = _compile(__splits[2][1], ColRange(s+len(__splits[0][1])-1+4, s+len(__splits[0][1])-1+4+len(__splits[2][1])-1))
            value_sep = _compile(__splits[4][1], ColRange(s+len(__splits[0][1])-1+4+len(__splits[2][1])-1+4, s+len(__splits[0][1])-1+4+len(__splits[2][1])-1+4+len(__splits[4][1])-1))
            if not CodeType.EVAL_EXECUTE in map(lambda v: v.codeType, (value_start, value_end, value_sep)):
                return Code(CodeType.EVAL_RANGE, deepcopy(col), start=value_start, end=value_end, sep=value_sep)


        if expression.startswith("set"):
            match_define = COMPILED_EXPRESSION_SET.fullmatch(expression)
            if match_define == None: return SyntaxError()

            argName = match_define.group(1)

            value = _compile(match_define.group(2), ColRange(col.start+4+len(argName)+4, col.end))
            if COMPILED_EXPRESSION_VARIABLE.fullmatch(argName) == None: return SyntaxError()
            if type(value) == Void: return SyntaxError()

            return Code(CodeType.EVAL_SET, deepcopy(col), var=argName[1:], value=value)
        elif expression.startswith("$"):
            return Code(CodeType.EVAL_VARIABLE, deepcopy(col), value=expression[1:])
        elif COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression) != None:
            codeTypeString:str = COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(1)
            split_result = splitArguments(COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(2))
            if codeTypeString == "function":
                return Code(CodeType.EXEC_FUNCTION_DEFINE, deepcopy(col), name=split_result[0][1], 
                            tokens=tuple(Array(split_result)[1:].map(lambda v: (ItSelf if not v[0] else lambda k: Argument(k[2:-1]))(v[1])))
                            )
            codeType = {
                "if"      : CodeType.EXEC_CONDITION_IF,
                "elif"    : CodeType.EXEC_CONDITION_ELIF,
                "else"    : CodeType.EXEC_CONDITION_ELSE,
                "while"   : CodeType.EXEC_LOOP_WHILE,
                "loop"    : CodeType.EXEC_LOOP_LOOP,
            }[codeTypeString]
            parse_result: List[Union[str, Code]] = []
            __col_st = col.start + len(codeTypeString) + 1
            for sps in split_result: # split string
                item = sps[1]
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    parse_result.append(_compile(item[1:-1], ColRange(__col_st+1, __col_st+len(item)-1)))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(codeType, deepcopy(col), args=tuple(parse_result))
        


        # 
        elif COMPILED_EXPRESSION_CALL_FUNC.fullmatch(expression) != None:
            match_function = COMPILED_EXPRESSION_CALL_FUNC.fullmatch(expression)
            if match_function == None: return SyntaxError()
            split_result = splitArguments(match_function.group(2))
            parse_result: List[Union[str, Code]] = []
            __col_st = col.start + len(match_function.group(1)) + 1
            for sps in split_result: # split string
                item = sps[1]
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    parse_result.append(_compile(item[1:-1], ColRange(__col_st+1, __col_st+len(item)-1)))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(CodeType.EVAL_EXECUTE, deepcopy(col), target=match_function.group(1), args=tuple(parse_result))
        else:
            __col_st = col.start
            i = 0
            __args = []
            __evalCode = ""
            for __type, __value in __splits:
                if not __type: __evalCode += __value
                else:
                    i += 1
                    __args.append(_compile(__value[1:-1], ColRange(__col_st+1, __col_st+len(__value)-1)))
                    __evalCode += "__args[" + str(i-1) + "]"
                __evalCode += " "
                __col_st += len(__value) + 1
            return Code(CodeType.EVAL_CALC, deepcopy(col), evalCode=__evalCode, args=tuple(__args))
    except Exception: raise
    finally: ...

def _eval(__code: Code, __file: str, __stack: StackSet):
    __looking__stack = __stack.last
    __looking__stack.col = __code.col
    try:
        if __code.codeType == CodeType.EMPTY: return VOID
        if __code.codeType == CodeType.PARENTHESIS: return _eval(__code.kw["code"], __file, __stack)
        if __code.codeType == CodeType.EVAL_INTEGER: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_DECIMAL: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_STRING:  return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_BOOLEAN: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_VARIABLE:
            return __stack.last.getData(__code.kw["value"])
        if __code.codeType == CodeType.EVAL_EXECUTE:
            target: Function = __stack.last.getData(__code.kw["target"])
            arguments = tuple([_eval(arg, __file, __stack) for arg in __code.kw["args"] if isinstance(arg, Code)])
            for value in arguments:
                if value is RELAY_SET_RETURN: raise SyntaxError("Function arguments must not be SET expression")
            result = target(arguments, __file, __stack)
            return result
        if __code.codeType == CodeType.EVAL_SET:
            value = _eval(__code.kw["value"], __file, __stack)
            if value is RELAY_SET_RETURN: raise SyntaxError("SET expression must not be SET expression")
            __stack.last.data[__code.kw["var"]] = Variable(value)
            return RELAY_SET_RETURN
        if __code.codeType == CodeType.EVAL_CALC:
            return eval(__code.kw["evalCode"], {"__args": [_eval(arg, __file, __stack) for arg in __code.kw["args"]]})
        if __code.codeType == CodeType.EVAL_LOOP_VALUE:
            if not __code.kw["index"] in __looking__stack.loop.values:
                raise SyntaxError("Loop value of index '"+str(__code.kw["index"])+"' is overloading. (" + ("no loop detected" if __looking__stack.loop.values == {} else "max loop index: "+str(max(__looking__stack.loop.values.keys()))) + ")")
            return __looking__stack.loop.values[__code.kw["index"]]
        if __code.codeType == CodeType.EVAL_RANGE: return range(_eval(__code.kw["start"], __file, __stack), _eval(__code.kw["end"], __file, __stack), _eval(__code.kw["sep"], __file, __stack))
    except _EvalException: raise
    except HandledException: raise
    except Exception as e: raise _EvalException(e, __code.col)

def _treeMap(__source: str) -> List[LineTree]:
    __source = "\n".join([__l.rstrip() for __l in _remove_comment(__source).split("\n")])

    master: List[LineTree] = []
    last_ident = 0
    lines = __source.split("\n")
    first_identify: Union[str, Void] = VOID
    for i, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")):
            master.append(LineTree(i, _compile(line)))
            first_identify = VOID
            last_ident = 0
        else:
            looking_identify:str = ""
            code: str = ""
            for char in list(line):
                if len(code) != 0: code += char
                elif char == " " or char == "\t": looking_identify += char
                else: code = char
            if last_ident == 0:
                master[-1].add_child(LineTree(i, _compile(code)))
                first_identify = looking_identify
                last_ident = 1
            else:
                last_ident = looking_identify.count(first_identify)
                eval("last" + ".childs[-1]" * (last_ident-1) + ".add_child(new)", {"last": master[-1], "new": LineTree(i, _compile(code))})
    return master

def _run_tree(master: List[LineTree], __file: str, __stack: StackSet = None):
    __IGNORE_IF = False
    __looking__stack = __stack.last
    __result = VOID
    for tree in master:
        __stack.last.line = tree.line
        __stack.last.col = tree.code.col
        try:
            if tree.code.is_eval:
                __result = _eval(tree.code, __file, __stack)
            elif tree.code.is_exec:
                match tree.code.codeType:
                    case CodeType.EXEC_LOOP_CONTINUE: return RELAY_LOOP_CONTINUE
                    case CodeType.EXEC_LOOP_BREAK:    return RELAY_LOOP_BREAK

                __relay = VOID
                if tree.code.codeType == CodeType.EXEC_CONDITION_IF:
                    __IGNORE_IF = False
                    if _eval(tree.code.kw["args"][0], __file, __stack):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __file, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELIF:
                    if (not __IGNORE_IF) and _eval(tree.code.kw["args"][0], __file, __stack):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __file, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELSE:
                    if (not __IGNORE_IF):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __file, __stack)
                else: __IGNORE_IF = False
                if __relay is RELAY_LOOP_CONTINUE or __relay is RELAY_LOOP_BREAK: return __relay
                del __relay

                if tree.code.codeType == CodeType.EXEC_LOOP_WHILE:
                    while _eval(tree.code.kw["args"][0], __file, __stack):
                        while_control = _run_tree(tree.childs, __file, __stack)
                        if while_control is RELAY_LOOP_CONTINUE: continue
                        elif while_control is RELAY_LOOP_BREAK: break
                elif tree.code.codeType == CodeType.EXEC_LOOP_LOOP:
                    __iter = _eval(tree.code.kw["args"][0], __file, __stack)
                    if type(__iter) == int: __iter = range(__iter)
                    __index: int = __looking__stack.loop.next_index
                    __looking__stack.loop.next_index += 1
                    for __item in __iter:
                        __looking__stack.loop.values[__index] = __item
                        loop_control:Void = _run_tree(tree.childs, __file, __stack)
                        if loop_control is RELAY_LOOP_CONTINUE: continue
                        elif loop_control is RELAY_LOOP_BREAK: break
                    __looking__stack.loop.next_index -= 1
                    del __looking__stack.loop.values[__index]
                elif tree.code.codeType == CodeType.EXEC_FUNCTION_DEFINE:
                    __stack.last.data[tree.code.kw["name"]] = Variable(create_function(tree.code.kw["name"], tree.code.kw['tokens'], tree.childs))
        except _EvalException as e: raise HandledException(e, deepcopy(__stack))
    return __result

def _remove_comment(__source: str) -> str:
    result = ""
    for l in __source.split("\n"):
        __is_str = False
        if l == "": result += "\n"; continue
        for char in list(l):
            if char == "\"": __is_str = not __is_str
            if char == "#" and not __is_str: break
            else: result += char
        result += "\n"
    return result[:-1]

def _exec(__source: Union[str, List[LineTree]], __globals: Dict[str, Any] = None,  __file: str = "<string>", __stack: StackSet = None, __ignoreSyntax: bool=False):
    if not __ignoreSyntax and type(__source) == str:
        __Tsource = "\n".join([__l.rstrip() for __l in _remove_comment(__source).split("\n")])
        __syntax, __err_lines = checkSyntax(__Tsource)
        if __syntax != SyntaxResultType.Right:
            note("Found wrong syntax while parsing script: \""+__file+"\"")
            for __err_line in __err_lines: note("[%s] at line %d `%s`"%(__syntax.name, __err_line, __Tsource.split("\n")[__err_line]))
            return ExecuteResult(ExecuteResultType.CompileError, ("SyntaxError", __syntax), __err_lines)
    
    try:
        if type(__source) == str: __sourceTree = _treeMap(__source)
        else: __sourceTree = __source

        __stack = StackSet(__file, __globals)
        __result = _run_tree(__sourceTree, __file, __stack)
        
        return ExecuteResult(ExecuteResultType.Success, __result)
    except HandledException as e:
        if e.exception.__class__ == SystemExit: return ExecuteResult(ExecuteResultType.Success)
        
        def note(*values, sep=" ", end=""):
            e.add_note(sep.join(map(str, values))+end)
        
        note("Traceback:")
        for stack in e.stack.history:
            note("  "+"File \""+stack.loc+"\" line "+str(stack.line+1))
            note("  "+"  "+__source.split("\n")[stack.line].lstrip())
            note("  "+"  "+" "*(stack.col.start) + "^"*(stack.col.end-stack.col.start))
        note(e.exception.__class__.__name__ +": "+ str(e.exception))
        
        #notes
        if type(e.exception) == NameError:
            from thefuzz.process import extract as extract_similar
            note("[Help] Did you mean '"+extract_similar(e.exception.args[0].split("'")[1], e.stack.getKeys())[0][0]+"'?")
                

        raise

if __candy_mode == "RUN":
    __source = ""
    with open(__candy_file, "r", encoding="utf-8") as f:
        __source = f.read()
    if __candy_setting["timer"]:
        time_start = __time()
    try:
        _exec(__source, __file=__candy_file)
    except HandledException as he:
        print("\n".join(he.__notes__))
    if __candy_setting["timer"]:
        print("Time elapsed: %.6f sec"%(__time()-time_start))
elif __candy_mode == "TERMINAL":
    __global = {}
    def __console_input(__prompt: str)->str:
        try: return input(__prompt)
        except KeyboardInterrupt as ki: print(); raise
        except: raise
    while True:
        try:
            __source = [__console_input(">>> ")]
            while True:
                last_spacing = len(__source[-1])-len(__source[-1].lstrip())
                if (not _remove_comment(__source[-1]).rstrip().endswith(":")) or (len(__source) > 1 and last_spacing == 0): break
                __source.append(__console_input("... "))
            if __candy_setting["timer"]:
                time_start = __time()
            __result = _exec("\n".join(__source), __global)
            if __result.result == ExecuteResultType.Success and not (type(__result.args[0]) is Void): print(__result.args[0])
            if __candy_setting["timer"]:
                print("Time elapsed: %.6f sec"%(__time()-time_start))
        except HandledException as he: print("\n".join(he.__notes__))
        except EOFError as eofe: break
        except KeyboardInterrupt as ki: pass
        except Exception as e:
            print("\nAn internal error has occurred.")
            raise