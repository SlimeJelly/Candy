from enum import Enum, auto
from types import NoneType
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Type, TypedDict, Union, Iterator, Literal
from dataclasses import dataclass, field
from copy import deepcopy
from sys import argv as __sys_argv
from os.path import exists, sep
from functools import reduce
from re import compile as regxp_compile
from pathlib import Path
from traceback import format_exc as __format_exc
from thefuzz.process import extract as extract_similar

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

def zip_range(iter: Iterable):
    """
    Make given numbers to tuple of range
    
    example)
    
    Input: [1, 2, 3, 4, 10, 11, 14, 15, 16]
    
    Output: [range(1, 5), range(10, 12), range(14, 17)]
    """
    if len(iter) == 0: return []
    start = min(iter)
    end = min(iter)
    result = []
    for i in iter:
        if end + 1 == i:
            end = i
        else:
            result.append(range(start, end+1))
            start = i
            end = i
    
    return result

def unzip_range(iter: Iterable):
    """
    Make given numbers to tuple of range
    
    example)
    
    Input: [range(1, 5), range(10, 12), range(14, 17)]
    
    Output: [1, 2, 3, 4, 10, 11, 14, 15, 16]
    """
    result = []
    for x in iter:
        result.extend(x)
    return result

sys_argv: Array = Array(__sys_argv)
del __sys_argv



class CandyLangInfo(TypedDict):
    location: str
    folder: str
    interpreter: Literal["PY", "EXE"]
    file: Union[str, NoneType]
    mode: Literal["RUN", "COMPILE", "TERMINAL"]

candy_info: CandyLangInfo = {
    "location": __file__,
    "folder": Path(__file__).parent.absolute().__str__(),
    "interpreter": __file__.split(".")[-1].upper(),
    "file": sys_argv.filter (lambda loc: loc.endswith(".candy"))\
                    .map    (lambda loc: loc.replace("\\", sep).replace("/", sep))\
                    .collect(None, lambda target, loc: loc if target == None else target),
    "mode": None
}

candy_info["mode"] = "TERMINAL" if candy_info["file"] == None else "COMPILE" if "-compile" in sys_argv else "RUN"
if candy_info["mode"] == None: raise Exception("Something went wrong")

del CandyLangInfo




class CandyLangSystemDebugSetting(TypedDict):
    print_info: bool

class CandyLangSetting(TypedDict):
    systemdebug: CandyLangSystemDebugSetting
    ignore_help: bool
    use_timer: bool
    
candy_setting: CandyLangSetting = {
    "systemdebug": {
        "print_info": True #FIXME: debug mode
    },
    "ignore_help": "-ignore-help" in sys_argv,
    "timer": "-timer" in sys_argv
}
if candy_setting["timer"]:
    from time import time as __time

del CandyLangSetting, CandyLangSystemDebugSetting




if type(candy_info["file"]) == str and not exists(candy_info["file"]): 
    print("Unable to find file \""+candy_info["file"]+"\"")
    exit(1)
del exists, sep

EXPRESSION_SPACE = "[ |\t]+"
EXPRESSION_NAME = "[a-z|A-Z|_][\w|\d|_]*"
EXPRESSION_VARIABLE = "("+EXPRESSION_NAME+")[ |\t]*([^\n]*)"
EXPRESSION_INTEGER = "-?[1-9][0-9]*|-?0"
EXPRESSION_DECIMAL = "-?[0-9]*?.[0-9]+"
COMPILED_EXPRESSION_EMPTY                = regxp_compile("[ |\t]+")
COMPILED_EXPRESSION_NAME                 = regxp_compile(EXPRESSION_NAME)
COMPILED_EXPRESSION_VARIABLE_OR_FUNCTION = regxp_compile("("+EXPRESSION_NAME+")[ |\t]*([^\n]*)")
COMPILED_EXPRESSION_VARIABLE_FORCE       = regxp_compile("\$" + EXPRESSION_NAME)
COMPILED_EXPRESSION_VARIABLE             = regxp_compile("\$?(" + EXPRESSION_NAME + ")")
COMPILED_EXPRESSION_STRING               = regxp_compile("\"[^\"\n]*\"")
COMPILED_EXPRESSION_INTEGER              = regxp_compile(EXPRESSION_INTEGER)
COMPILED_EXPRESSION_DECIMAL              = regxp_compile(EXPRESSION_DECIMAL)
COMPILED_EXPRESSION_BOOLEAN              = regxp_compile("(true|false)")
COMPILED_EXPRESSION_SET                  = regxp_compile("set"+EXPRESSION_SPACE+"(\$?"+EXPRESSION_NAME+")"+EXPRESSION_SPACE+"to"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_WORD                 = regxp_compile("\\w+")
COMPILED_EXPRESSION_BLOCKHOLDER          = regxp_compile("(if|elif|else|for|while|loop|function)"+"([^\n]*):")
COMPILED_EXPRESSION_LOOP_VALUE           = regxp_compile("loop-value-\\d+")
COMPILED_EXPRESSION_LOOP_CONTROL         = regxp_compile("(break|continue)")
COMPILED_EXPRESSION_RETURN               = regxp_compile("return [^\n]*")
COMPILED_EXPRESSION_VARIABLE_CONTROL     = regxp_compile("(forget|share)"+EXPRESSION_SPACE+"([^\n]+)")
del EXPRESSION_SPACE, EXPRESSION_NAME, EXPRESSION_VARIABLE, EXPRESSION_INTEGER, EXPRESSION_DECIMAL
del regxp_compile

OPERATOR_TOKENS = ("+", "-", "*", "/", "%", "//", "**", 
                   "==", "!=", "<", ">", "<=", ">=",)

SPACING = (" ", "\t")

class RelayData():
    def __init__(self, name: str) -> None:
        self.name = name
    @staticmethod
    def isRelayData(target): return isinstance(target, RelayData)

    EMPTY: "RelayData"
    MISSING_ARG: "RelayData" = None
    NO_MASTER: "RelayData" = None
    NO_RESULT: "RelayData" = None
    LOOP_BREAK: "RelayData" = None
    LOOP_CONTINUE: "RelayData" = None
    
    ReturnData: Type["ReturnData"] = None

class ReturnData(RelayData):
    def __init__(self, data: Any) -> None:
        super().__init__("RETURN")
        self.data = data 
    def __str__(self) -> str: return "ReturnData("+str(self.data)+")"
    def __repr__(self) -> str: return "ReturnData("+self.data.__repr__()+")"

RelayData.ReturnData = ReturnData
del ReturnData

RelayData.EMPTY = RelayData("EMPTY")   
RelayData.MISSING_ARG = RelayData("MISSING_ARG")
RelayData.NO_MASTER = RelayData("NO_MASTER")
RelayData.NO_RESULT = RelayData("NO_RESULT")
RelayData.LOOP_BREAK = RelayData("LOOP_BREAK")
RelayData.LOOP_CONTINUE = RelayData("LOOP_CONTINUE")


# @dataclass(frozen=True)
# class Void():
#     data: str

# VOID = Void("VOID")
# MISSING_ARG = Void("MISSING_ARG")
# RELAY_SET_RETURN = Void("RELAY_SET_RETURN")
# RELAY_LOOP_BREAK = Void("RELAY_LOOP_BREAK")
# RELAY_LOOP_CONTINUE = Void("RELAY_LOOP_CONTINUE")

class StackType(Enum):
    ROOT = 0
    MODULE = 1
    FUNCTION = 2

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
    loc: "Script" # file
    line: int
    type: StackType
    col: ColRange
    master: Union["Stack", NoneType] = None
    loop: LoopData = field(default_factory=LoopData)
    data: Dict[str, Variable] = field(default_factory=dict)
    def get(self, name: str):
        if name in self.data: return self.data[name].value
        elif self.master != None: return self.master.get(name)
        else: raise NameError("name '"+name+"' is not defined")
    def forget(self, name: str):
        if name in self.data: del self.data[name]
        elif self.master != None: self.master.forget(name)
        else: raise NameError("name '"+name+"' is not defined")

#TODO: 각 file의 마스터 스택을 분리
class StackSet():
    # def __init__(self, loc: "Script", __global: Dict[str, Any] = None) -> None:
    #     self.stacks: List[Stack] = [Stack(loc, 0, type=StackType.ROOT, col=ColRange(0, 0))]
    #     __global = {} if __global == None else __global
    #     for key, value in _dict.items():
    #         if not (key in __global): __global[key] = Variable(value)
    #     self.stacks[-1].data = __global
    def __init__(self, __global: Dict[str, Any] = None) -> None:
        self.stacks: List[Stack] = [_master_stack]
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
    def last(self) -> Stack:
        return self.stacks[-1]
    @property
    def history(self) -> Tuple[Stack, ...]:
        return tuple(filter(lambda s: s.loc.path != "<Candy>", self.stacks[1:])) # FIXME(?) Show only non-internal stacks (beacuse of built-in functions)
    @property
    def file_stack(self) -> Tuple[Stack, ...]:
        result = []
        for stack in self.history:
            if stack.type == StackType.ROOT: result.append(stack)
            if stack.loc == result[-1].loc:
                result[-1] = stack
        return tuple(result[1:])
    def getKeys(self)->List[str]:
        return list(reduce(lambda a, b: a.union(b), [set(stack.data.keys()) for stack in self.stacks]))




@dataclass(frozen=True)
class Argument():
    name:str
    auto:Any=RelayData.MISSING_ARG

class _Candy_Object():
    def __init__(self, _name: Union[str, NoneType], _type: Union[str, NoneType]="Unkown", _internal: bool=False) -> None:
        self.__str = "<"+("internal " if _internal else "")+_type+" "+_name+">"
    def __str__(self) -> str: return self.__str

class Function(_Candy_Object):
    def __init__(self, name: str, tokens: List[Union[str, Argument]], func: Callable, origin: "Script", _internal: bool=False):
        super().__init__(origin.name+"."+name, "function", _internal)
        self.name: str = name
        self.tokens: Set[str] = set([token for token in tokens if type(token) == str])
        self.__args: List[Argument] = [token for token in tokens if type(token) == Argument]
        self.less_args = len([arg for arg in self.__args if arg.auto is RelayData.MISSING_ARG])
        self.func = func
        self.origin = origin
        
    def __call__(self, variables: List[Any], _stack: "StackSet") -> Any:
        if len(self.__args) != len(variables): pass # error
        if len(variables) < self.less_args: raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        variables += (RelayData.MISSING_ARG,) * (len(self.__args) - len(variables))
        localDict: Dict[str, Any] = {arg_slot.name:Variable(var if not RelayData.isRelayData(var) else arg_slot.auto) for arg_slot, var in zip(self.__args, variables)}
        if RelayData.MISSING_ARG in localDict.values(): raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        innerStack = (lambda lastStack: Stack(self.origin, lastStack.line, type=StackType.FUNCTION, col=lastStack.col, master=lastStack, data=localDict))(_stack.last)
        _stack.enter(innerStack)
        v = self.func(localDict, _stack)
        _stack.exit()
        return v

def create_py_function(func_name:str, tokens: List[Union[str, Argument]], func: Callable, auto_return: bool = True):
    def __func (arguments, _): 
        __relay = func(**{k:v.value for k, v in arguments.items()})
        if auto_return and __relay is None: return RelayData.NO_RESULT
        return __relay
    return Function(func_name, tokens, __func, _master_script, _internal=True)





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
    EVAL_AUTO_FVARIABLE_FUNCTION = 105
    EVAL_VARIABLE_GET = 106
    EVAL_RANGE = 107

    EVAL_LOOP_VALUE = 110

    #Eval - Work (120~)
    EVAL_SET = 120
    EVAL_CALC = 121
    
    EVAL_VARIABLE_FORGET = 130
    EVAL_VARIABLE_SHARE = 131

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
    def __init__(self, line: int, code: Code, master: Union[RelayData, "LineTree"] = RelayData.NO_MASTER):
        self.line = line
        self.code = code
        self.master = master
        self.childs: List["LineTree"] = []
    def add_child(self, child: "LineTree") -> None:
        child.master = self
        self.childs.append(child)
    def __str__(self) -> str: return "["+str(self.line)+"]"+"( "+", ".join([str(l) for l in self.childs])+" )"

def create_function(name: str, parameters: List[Union[str, Argument]], code: List[LineTree], __origin: "Script") -> Function:
    def __func(arguments: Dict[str, Any], __stack: StackSet):
        __relay = _run_tree(code, __stack)
        if RelayData.isRelayData(__relay) and __relay.name == "RETURN": return __relay.data
        return __relay
    return Function(name, parameters, __func, __origin)

# class Module():
#     def __init__(self, name: str, tree: List[LineTree], colmap: ColMapper) -> None:
#         self.name = name
#         self.tree = tree
#         self.colmap = colmap

class InternalException(Exception):
    ...

class _EvalException(InternalException):
    def __init__(self, exception: Exception, col: ColRange) -> None:
        self.exception = exception
        self.col = col

class HandledException(InternalException):
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
    
    

    elif __source.startswith("set"):
        match_define = COMPILED_EXPRESSION_SET.fullmatch(__source)
        return match_define != None \
                and (COMPILED_EXPRESSION_NAME.fullmatch(match_define.group(1)) != None
                     or COMPILED_EXPRESSION_VARIABLE_FORCE.fullmatch(match_define.group(1)) != None) \
                and checkExpressionSyntax(match_define.group(2))
    elif COMPILED_EXPRESSION_STRING.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_DECIMAL.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_INTEGER.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_VARIABLE_FORCE.fullmatch(__source) != None: return True
    
    splits = splitArguments(__source)
    if len(splits) == 5 and splits[1][1] == "to" and splits[3][1] == "by" and checkExpressionSyntax(splits[0][1]) and checkExpressionSyntax(splits[2][1]) and checkExpressionSyntax(splits[4][1]): return True
    
    if COMPILED_EXPRESSION_VARIABLE_CONTROL.fullmatch(__source) != None:
        match_variable_control = COMPILED_EXPRESSION_VARIABLE_CONTROL.fullmatch(__source)
        return not (False in Array(match_variable_control.group(2).split(","))\
                                    .filter(lambda v: len(v.strip()) != 0)\
                                    .map(lambda v: v.lstrip().rstrip())\
                                    .map(lambda v: COMPILED_EXPRESSION_VARIABLE.fullmatch(v) != None))
    if __source.count(" ") == 1:
        splits = __source.split(" ")
        if splits[0].endswith("'s") and checkExpressionSyntax(splits[0][:-2]) and COMPILED_EXPRESSION_NAME.fullmatch(splits[1]) != None:
            return True
    if COMPILED_EXPRESSION_LOOP_VALUE.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_LOOP_CONTROL.fullmatch(__source) != None: return True
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
    if COMPILED_EXPRESSION_RETURN.fullmatch(__source) != None:
        return_value = __source[6:].lstrip()
        return checkExpressionSyntax(return_value)
        
    if COMPILED_EXPRESSION_VARIABLE_OR_FUNCTION.fullmatch(__source) != None:
        match_function = COMPILED_EXPRESSION_VARIABLE_OR_FUNCTION.fullmatch(__source)
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
    first_identify: Union[str, NoneType] = None
    for l, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")): first_identify, last_ident = None, 0
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

def _wrap_source(__source: str) -> str:
    code_deleted: List[int] = [] # list of deleted char index
    __looking_index = 0
    __nowline_index = 0
    
    result: str = ""
    __is_line_combine = False # 줄 합치기 " \ " 여부
    __is_str = False # 문자열 내부 여부
    __is_str_escape = False # 문자열 내부 이스케이프 여부
    __bracket_depth = 0 # 괄호 깊이
    
    __last_line_combine = False # 이전 줄이 합쳐진 줄인지 여부
    __current_line_non_space_appeared = False # 현재 줄에 공백을 제외한 문자가 나왔는지 여부
    __last_newLine = False # 이전 문자가 개행문자인지 여부
    for line in __source.split("\n"):
        __last_newLine = False
        __current_line_non_space_appeared = False
        __ignore_space_delete = False
        __nowline_index = 0
        for char in list(line):
            if __is_str_escape: 
                __is_str_escape = False
            elif char == "#":
                code_deleted.extend(range(__looking_index, __looking_index - __nowline_index + len(line)))
                __looking_index += len(line) - __nowline_index
                break
            elif char == "\"":
                __is_str = not __is_str
            elif char == "\\":
                __ignore_space_delete = True
                __is_line_combine = True
                code_deleted.extend(range(__looking_index, __looking_index - __nowline_index + len(line)))
                __looking_index += len(line) - __nowline_index
                break
            elif char == "(": __bracket_depth += 1
            elif char == ")": __bracket_depth -= 1
            elif __last_line_combine and (not __current_line_non_space_appeared) and char in SPACING:
                code_deleted.append(__looking_index)
                __looking_index += 1
                __nowline_index += 1
                continue
            __current_line_non_space_appeared = True
            result += char
            __looking_index += 1
            __nowline_index += 1
        if not __ignore_space_delete:
            result = result.rstrip()
        __last_line_combine = False
        if __bracket_depth < 0:
            raise SyntaxError("Found unmatched bracket at line %d"%(__source.split("\n").index(line)+1))
        elif __bracket_depth > 0:
            __is_line_combine = True
        if not __is_line_combine:
            result += "\n"
            __last_newLine = True
            # print(max(range(__looking_index, __looking_index - __nowline_index + len(line))))
            code_deleted.append(__looking_index)
        else:
            __last_line_combine = True
        __is_line_combine = False
        __looking_index += 1
    if __last_newLine:
        result = result[:-1]
    if len(__source) in code_deleted:
        code_deleted.remove(len(__source))
    code_deleted = tuple(set(code_deleted))
    return result, zip_range(code_deleted)

def _parse(expression: str, col: ColRange = None) -> Code:
    if col == None: col = ColRange(0, len(expression)-1)
    try:
        if expression == "": return Code(CodeType.EMPTY, deepcopy(col))
        elif match_parenthesis(expression): return Code(CodeType.PARENTHESIS, col, code=_parse(expression[1:-1], ColRange(col.start+1, col.end-1)))
        elif COMPILED_EXPRESSION_INTEGER     .fullmatch(expression) != None: return Code(CodeType.EVAL_INTEGER , deepcopy(col), value=int  (expression))
        elif COMPILED_EXPRESSION_DECIMAL     .fullmatch(expression) != None: return Code(CodeType.EVAL_DECIMAL , deepcopy(col), value=float(expression))
        elif COMPILED_EXPRESSION_STRING      .fullmatch(expression) != None: return Code(CodeType.EVAL_STRING  , deepcopy(col), value=parse_string (expression))
        elif COMPILED_EXPRESSION_BOOLEAN     .fullmatch(expression) != None: return Code(CodeType.EVAL_BOOLEAN , deepcopy(col), value=bool (expression))
        elif COMPILED_EXPRESSION_LOOP_VALUE  .fullmatch(expression) != None: return Code(CodeType.EVAL_LOOP_VALUE, deepcopy(col), index=int(expression.replace("loop-value-", "")))
        elif COMPILED_EXPRESSION_LOOP_CONTROL.fullmatch(expression) != None: return Code(CodeType.EXEC_LOOP_CONTINUE if expression=="continue" else CodeType.EXEC_LOOP_BREAK, deepcopy(col))
        __splits = splitArguments(expression)
        if len(__splits) == 5 and __splits[1][1] == "to" and __splits[3][1] == "by":
            s = col.start
            value_start = _parse(__splits[0][1], ColRange(s, s+len(__splits[0][1])-1))
            value_end = _parse(__splits[2][1], ColRange(s+len(__splits[0][1])-1+4, s+len(__splits[0][1])-1+4+len(__splits[2][1])-1))
            value_sep = _parse(__splits[4][1], ColRange(s+len(__splits[0][1])-1+4+len(__splits[2][1])-1+4, s+len(__splits[0][1])-1+4+len(__splits[2][1])-1+4+len(__splits[4][1])-1))
            if set(map(lambda v: v.codeType, (value_start, value_end, value_sep))) <= {CodeType.EVAL_INTEGER, CodeType.EVAL_DECIMAL, CodeType.PARENTHESIS}:
                return Code(CodeType.EVAL_RANGE, deepcopy(col), start=value_start, end=value_end, sep=value_sep)

        if expression.startswith("set"):
            match_define = COMPILED_EXPRESSION_SET.fullmatch(expression)
            if match_define == None: return SyntaxError()

            argName = match_define.group(1)
            if len(argName) == 0: return SyntaxError()
            __is_force_variable = argName[0] == '$'

            value = _parse(match_define.group(2), ColRange(col.start+4+len(argName)+4, col.end))
            if RelayData.isRelayData(value): return SyntaxError()

            if __is_force_variable: argName = argName[1:]
            return Code(CodeType.EVAL_SET, deepcopy(col), var=argName, value=value)
        elif COMPILED_EXPRESSION_VARIABLE_CONTROL.fullmatch(expression) != None:
            match_variable_control = COMPILED_EXPRESSION_VARIABLE_CONTROL.fullmatch(expression)
            return Code({
                            "forget": CodeType.EVAL_VARIABLE_FORGET, 
                            "share" : CodeType.EVAL_VARIABLE_SHARE
                        }[match_variable_control.group(1)], 
                        deepcopy(col), 
                        targets= tuple(Array(match_variable_control.group(2).split(","))
                                    .filter(lambda v: len(v.strip()) != 0)
                                    .map(lambda v: v.lstrip().rstrip().replace("$", "")))
                        )
        elif expression.startswith("$"):
            return Code(CodeType.EVAL_VARIABLE, deepcopy(col), value=expression[1:])
        elif COMPILED_EXPRESSION_RETURN.fullmatch(expression) != None:
            return Code(CodeType.EXEC_FUNCTION_RETURN, deepcopy(col), value=_parse(expression[6:].lstrip(), ColRange(col.start+7, col.end)))
        elif COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression) != None:
            codeTypeString: str = COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(1)
            split_result = splitArguments(COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(2))
            if codeTypeString == "function":
                return Code(CodeType.EXEC_FUNCTION_DEFINE, deepcopy(col), name=split_result[0][1], 
                            tokens=tuple(
                                Array(split_result)[1:]\
                                    .map(lambda v: 
                                        (ItSelf if not v[0] else lambda k: Argument((lambda k: k[1:] if k.startswith("$") else k)(k[1:-1])))(v[1]))
                            ))
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
                    parse_result.append(_parse(item[1:-1], ColRange(__col_st+1, __col_st+len(item)-1)))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(codeType, deepcopy(col), args=tuple(parse_result))
        

        elif expression.split(" ")[0].endswith("'s"):
            return Code(CodeType.EVAL_VARIABLE_GET, deepcopy(col), target=_parse(expression.split(" ")[0][:-2], ColRange(col.start, col.end-2)), value=expression.split(" ")[1])
        # 
        elif COMPILED_EXPRESSION_VARIABLE_OR_FUNCTION.fullmatch(expression) != None:
            match_function = COMPILED_EXPRESSION_VARIABLE_OR_FUNCTION.fullmatch(expression)
            if match_function == None: return SyntaxError()
            
            split_result = splitArguments(match_function.group(2))
            
            parse_result: List[Union[str, Code]] = []
            __col_st = col.start + len(match_function.group(1)) + 1
            for sps in split_result: # split string
                item = sps[1]
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    parse_result.append(_parse(item[1:-1], ColRange(__col_st+1, __col_st+len(item)-1)))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(CodeType.EVAL_AUTO_FVARIABLE_FUNCTION, deepcopy(col), target=match_function.group(1), args=tuple(parse_result))
        else:
            __col_st = col.start
            i = 0
            __args = []
            __evalCode = ""
            for __type, __value in __splits:
                if not __type: __evalCode += __value
                else:
                    i += 1
                    __args.append(_parse(__value[1:-1], ColRange(__col_st+1, __col_st+len(__value)-1)))
                    __evalCode += "__args[" + str(i-1) + "]"
                __evalCode += " "
                __col_st += len(__value) + 1
            return Code(CodeType.EVAL_CALC, deepcopy(col), evalCode=__evalCode, args=tuple(__args))
    except Exception: raise
    finally: ...

def _treeMap(__source: str) -> List[LineTree]:
    __source = "\n".join([__l.rstrip() for __l in __source.split("\n")])

    master: List[LineTree] = []
    last_ident = 0
    lines = __source.split("\n")
    first_identify: Union[str, NoneType] = None
    for i, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")):
            master.append(LineTree(i, _parse(line)))
            first_identify = None
            last_ident = 0
        else:
            looking_identify: str = ""
            code: str = ""
            for char in list(line):
                if len(code) != 0: code += char
                elif char == " " or char == "\t": looking_identify += char
                else: code = char
            if last_ident == 0:
                master[-1].add_child(LineTree(i, _parse(code)))
                first_identify = looking_identify
                last_ident = 1
            else:
                last_ident = looking_identify.count(first_identify)
                __looking__tree = master[-1]
                for _ in range(last_ident-1):
                    __looking__tree = __looking__tree.childs[-1]
                __looking__tree.add_child(LineTree(i, _parse(code)))
                del __looking__tree
    return master

def _compile(__source: str, __file: str, __ignoreSyntax: bool=False):
    __source, zip_deleted_chars = _wrap_source(__source)
    deleted_chars = unzip_range(zip_deleted_chars)
    if not __ignoreSyntax and type(__source) == str:
        __Tsource = "\n".join([__l.rstrip() for __l in __source.split("\n")])
        __syntax, __err_lines = checkSyntax(__Tsource)
        if __syntax != SyntaxResultType.Right:
            se = SyntaxError("Found wrong syntax while parsing script: \""+__file+"\"")
            for __err_line in __err_lines: se.add_note("[%s] at line %d `%s`"%(__syntax.name, __err_line, __Tsource.split("\n")[__err_line]))
            raise se
    
    __sourceTree = _treeMap(__source)
    return __sourceTree, deleted_chars

class ScriptType(Enum):
    SCRIPT = 0
    COMPILED_SCRIPT = 1

class Script():
    def __init__(self, __path: str) -> None:
        self.__path: str = __path
        
    def fake(self, __source: str) -> None:
        self.file: Path = Path(candy_info["location"])
        self.path: str = self.__path
        self.name: str = self.__path
        self.fileType = ScriptType.SCRIPT
        self.source = __source
        
    def load(self) -> None:
        self.file: Path = Path(self.__path).absolute()
        self.path: str = self.file.__str__()
        self.name: str = self.file.name
        if not self.file.exists():
            raise FileNotFoundError(f"File '{self.file}' is not found.")
        elif not self.file.is_file():
            raise FileNotFoundError(f"File '{self.file}' is not a file.")
        elif not self.file.suffix in (".candy", ".ccandy"):
            raise TypeError(f"File '{self.file}' is not a candy script.")
        self.fileType = ScriptType.SCRIPT if self.file.suffix == ".candy" else ScriptType.COMPILED_SCRIPT
        if self.fileType == ScriptType.SCRIPT:
            self.source = self.file.read_text(encoding="utf-8")
    
    def compile(self, __ignoreSyntax: bool = False) -> None:
        if self.fileType == ScriptType.SCRIPT:
            self.tree, self.deleted_chars = _compile(self.source, self.path, __ignoreSyntax)
    def run(self, __globals: Dict[str, Any] = None):
        return _exec(self, __globals)

def _eval(__code: Code, __stack: StackSet):
    __looking__stack = __stack.last
    __last_stack_col = __looking__stack.col
    __looking__stack.col = __code.col
    __is_exception_raised = False
    try:
        if __code.codeType == CodeType.EMPTY: return RelayData.NO_RESULT
        if __code.codeType == CodeType.PARENTHESIS: return _eval(__code.kw["code"], __stack)
        if __code.codeType == CodeType.EVAL_INTEGER: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_DECIMAL: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_STRING:  return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_BOOLEAN: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_VARIABLE: return __stack.last.get(__code.kw["value"])
        if __code.codeType == CodeType.EVAL_VARIABLE_GET:
            _from = _eval(__code.kw["target"], __stack)
            _attr = __code.kw["value"]
            if _attr in object().__dir__(): raise AttributeError("Can't access to internal data '"+str(_from)+"."+_attr+"'")
            result = object.__getattribute__(_from, _attr)
            if type(result) == Variable: return result.value
            return result
        if __code.codeType == CodeType.EVAL_AUTO_FVARIABLE_FUNCTION:
            _attr = __stack.last.get(__code.kw["target"])
            if type(_attr) != Function: return _attr
            arguments = tuple([_eval(arg, __stack) for arg in __code.kw["args"] if isinstance(arg, Code)])
            for value in arguments:
                if value is RelayData.NO_RESULT: raise SyntaxError("Function arguments must not be SET expression")
            result = _attr.__call__(arguments, __stack)
            return result
        if __code.codeType == CodeType.EVAL_SET:
            value = _eval(__code.kw["value"], __stack)
            if value is RelayData.NO_RESULT: raise SyntaxError("The expression of value is non-result.")
            __stack.last.data[__code.kw["var"]] = Variable(value)
            return RelayData.NO_RESULT
        if __code.codeType == CodeType.EVAL_CALC:
            return eval(__code.kw["evalCode"], {"__args": [_eval(arg, __stack) for arg in __code.kw["args"]]})
        if __code.codeType == CodeType.EVAL_LOOP_VALUE:
            if not __code.kw["index"] in __looking__stack.loop.values:
                raise SyntaxError("Loop value of index '"+str(__code.kw["index"])+"' is overloading. (" + ("no loop detected" if __looking__stack.loop.values == {} else "max loop index: "+str(max(__looking__stack.loop.values.keys()))) + ")")
            return __looking__stack.loop.values[__code.kw["index"]]
        if __code.codeType == CodeType.EVAL_RANGE: return range(_eval(__code.kw["start"], __stack), _eval(__code.kw["end"], __stack), _eval(__code.kw["sep"], __stack))
        if __code.codeType == CodeType.EVAL_VARIABLE_FORGET:
            __current_stack = __stack.last
            for _attr in __code.kw["targets"]:
                __current_stack.forget(_attr)
            return RelayData.NO_RESULT
        if __code.codeType == CodeType.EVAL_VARIABLE_SHARE:
            __current_stack = __stack.last
            __upper_stack = __current_stack.master
            for _attr in __code.kw["targets"]:
                __upper_stack.data[_attr] = __current_stack.data[_attr]
            return RelayData.NO_RESULT
    except InternalException: __is_exception_raised = True; raise
    except Exception as e: __is_exception_raised = True; raise _EvalException(e, __code.col)
    finally:
        if not __is_exception_raised:
            __looking__stack.col = __last_stack_col

def _run_tree(master: List[LineTree], __stack: StackSet = None):    
    __IGNORE_IF = False
    __looking__stack = __stack.last
    __result = RelayData.EMPTY
    for tree in master:
        __looking__stack.line = tree.line
        __looking__stack.col = tree.code.col
        try:
            if tree.code.is_eval:
                __result = _eval(tree.code, __stack)
            elif tree.code.is_exec:
                match tree.code.codeType:
                    case CodeType.EXEC_LOOP_CONTINUE:   return RelayData.LOOP_CONTINUE
                    case CodeType.EXEC_LOOP_BREAK:      return RelayData.LOOP_BREAK
                    case CodeType.EXEC_FUNCTION_RETURN: 
                        return RelayData.ReturnData(_eval(tree.code.kw["value"], __stack))

                __relay = RelayData.EMPTY
                if tree.code.codeType == CodeType.EXEC_CONDITION_IF:
                    __IGNORE_IF = False
                    if _eval(tree.code.kw["args"][0], __stack):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELIF:
                    if (not __IGNORE_IF) and _eval(tree.code.kw["args"][0], __stack):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELSE:
                    if (not __IGNORE_IF):
                        __IGNORE_IF = True
                        __relay = _run_tree(tree.childs, __stack)
                else: __IGNORE_IF = False
                if __relay is RelayData.LOOP_CONTINUE or __relay is RelayData.LOOP_BREAK: return __relay
                if RelayData.isRelayData(__relay) and __relay.name == "RETURN": return __relay
                del __relay

                if tree.code.codeType == CodeType.EXEC_LOOP_WHILE:
                    while _eval(tree.code.kw["args"][0], __stack):
                        while_control = _run_tree(tree.childs, __stack)
                        if while_control is RelayData.LOOP_CONTINUE: continue
                        elif while_control is RelayData.LOOP_BREAK: break
                        elif RelayData.isRelayData(while_control) and while_control.name == "RETURN": return while_control
                elif tree.code.codeType == CodeType.EXEC_LOOP_LOOP:
                    __iter = _eval(tree.code.kw["args"][0], __stack)
                    if type(__iter) == int: __iter = range(__iter)
                    __index: int = __looking__stack.loop.next_index
                    __looking__stack.loop.next_index += 1
                    for __item in __iter:
                        __looking__stack.loop.values[__index] = __item
                        loop_control: RelayData = _run_tree(tree.childs, __stack)
                        if loop_control is RelayData.LOOP_CONTINUE: continue
                        elif loop_control is RelayData.LOOP_BREAK: break
                        elif RelayData.isRelayData(loop_control) and loop_control.name == "RETURN": return loop_control
                    __looking__stack.loop.next_index -= 1
                    del __looking__stack.loop.values[__index]
                elif tree.code.codeType == CodeType.EXEC_FUNCTION_DEFINE:
                    __looking__stack.data[tree.code.kw["name"]] = Variable(create_function(tree.code.kw["name"], tree.code.kw['tokens'], tree.childs, __looking__stack.loc))
        except _EvalException as e: raise HandledException(e, deepcopy(__stack))
    return __result

def _exec(__source: Script, __stack: StackSet, __globals: Dict[str, Any] = None):
    try:
        __stack = StackSet(__globals)
        __stack.enter(Stack(__source, 0, StackType.MODULE, ColRange(0, 0), __stack.last))
        __result = _run_tree(__source.tree, __stack)
    
        return __result

    except HandledException as e:
        if e.exception.__class__ == SystemExit: raise SystemExit
        
        def note(*values, sep=" ", end=""):
            e.add_note(sep.join(map(str, values)) + end)
        
        note("Traceback:")
        for stack in e.stack.history:
            note("  "+"File \""+stack.loc.path+"\" line "+str(stack.line+1))
            note("  "+"  "+stack.loc.source.split("\n")[stack.line].lstrip())
            note("  "+"  "+" "*(stack.col.start) + "^"*(stack.col.end-stack.col.start))
        note(e.exception.__class__.__name__ +": "+ str(e.exception))
        
        if not candy_setting["ignore_help"]:
            #notes
            if type(e.exception) == NameError:
                note("[Help] Did you mean '"+extract_similar(e.exception.args[0].split("'")[1], e.stack.getKeys())[0][0]+"'?")
        if candy_setting["systemdebug"]["print_info"]:
            note("\nDebugging Info:")
            
            note("")
            note("[System History]")
            
            history_string = __format_exc().split("\n")
            for exc in history_string:
                if exc == "": break
                note("  "+exc)
            
            note("")
            note("[Stack History]")
            for stack in e.stack.history:
                note("  "+"File \""+stack.loc.path+"\" line "+str(stack.line+1)+" col "+str(stack.col.start+1))
            
            note("")
            note("[Data History]")
            __data_container = []
            for stack in e.stack.history:
                __data_container.append(tuple(map(str, [stack.loc.path, stack.line, str(stack.col.start)+":"+str(stack.col.end), ", ".join(stack.data.keys())])))
            __max_length_loc = max([len(__data[0]) for __data in __data_container])
            __max_length_line = max([len(__data[1]) for __data in __data_container])
            __max_length_col = max([len(__data[2]) for __data in __data_container])
            __data_text = "\n".join(
                map(lambda __data: ("  "
                                    +"File \""+__data[0].ljust(__max_length_loc)+"\" "
                                    +"line "+__data[1].ljust(__max_length_line)+" "
                                    +"col "+__data[2].ljust(__max_length_col)
                                    
                                    +"\n"
                                    +"    "+__data[3]
                                    +"\n"), __data_container)
            )
            note(__data_text)
            
                
        raise
    
_master_script = Script("<Candy>")
_master_script.fake("")
_master_stack = Stack(_master_script, 0, type=StackType.ROOT, col=ColRange(0, 0))

_dict = {}
from time import sleep
def __print(text): print(text, end="")
def __println(text): print(text)
def __combine(t1, t2): return str(t1)+str(t2)
def __repeat(text, count): return str(text)*count
class Forever(_Candy_Object, Iterator):
    def __init__(self) -> None:
        super().__init__("forever", "Forever", _internal=True)
        self.index = 0
    def __next__(self) -> int: self.index += 1; return self.index
_dict["print"]    = create_py_function("print",    ["of", Argument("text")], __print)
_dict["println"]  = create_py_function("println",  ["of", Argument("text")], __println)
_dict["input"]    = create_py_function("input",    [], input)
_dict["combine"]  = create_py_function("combine",  ["of", Argument("t1"), "and", Argument("t2")], __combine)
_dict["repeat"]   = create_py_function("repeat",   ["of", Argument("text"), "for", Argument("count"), "times"], __repeat)
_dict["range"]    = create_py_function("range",    ["of", Argument("start"), "to", Argument("end"), "by", Argument("sep", auto=1)], lambda start, end, sep: range(start, end, sep))
_dict["asString"] = create_py_function("asString", ["of", Argument("text")], lambda text: str(text))
_dict["asInt"]    = create_py_function("asInt",    ["of", Argument("text")], lambda text: int(text))
_dict["asFloat"]  = create_py_function("asFloat",  ["of", Argument("text")], lambda text: float(text))
_dict["asBool"]   = create_py_function("asBool",   ["of", Argument("text")], lambda text: bool(text))
_dict["asList"]   = create_py_function("asList",   ["of", Argument("text")], lambda text: list(text))
_dict["length"]   = create_py_function("length",   ["of", Argument("text")], lambda text: len(text))
_dict["wait"]     = create_py_function("wait",     ["for", Argument("time"), "seconds"], lambda time: sleep(time))
_dict["forever"]  = create_py_function("forever",  [], Forever)

if candy_info["mode"] == "RUN":
    __source = ""
    master_script = Script(candy_info["file"])
    master_script.load()
    master_script.compile()
    if candy_setting["timer"]:
        time_start = __time()
    try:
        master_script.run()
    except HandledException as he:
        print("\n".join(he.__notes__))
    if candy_setting["timer"]:
        print("Time elapsed: %.6f sec"%(__time()-time_start))

elif candy_info["mode"] == "TERMINAL":
    __global = {}
    def __console_input(__prompt: str)->str:
        try: return input(__prompt)
        except KeyboardInterrupt as ki: print("^C"); raise
        except: raise
    
    def __remove_string(__source: str) -> str:
        __is_string = False
        __is_escape = False
        result = ""
        for char in list(__source):
            if __is_string:
                if   __is_escape : __is_escape = False
                elif char == "\\": __is_escape = True
                elif char == "\"": __is_string = False
            else:
                if   char == "\"": __is_string = True
                else:              result += char
        return result
    
    while True:
        try:
            __source = [__console_input(">>> ")]
            bracket_count = 0
            while True:
                last_spacing = len(__source[-1])-len(__source[-1].lstrip())
                
                bracket_count += (lambda text: text.count("(") - text.count(")"))(__remove_string(__source[-1]))
                if bracket_count <= 0 \
                    and (
                        (not __source[-1].replace("\\", "").rstrip().endswith(":")) \
                        or (len(__source) > 1 and last_spacing == 0)
                    ): break
                __source.append(__console_input("... "))
            
            terminal_script = Script("<terminal>")
            terminal_script.fake("\n".join(__source))
            terminal_script.compile()
            if candy_setting["timer"]:
                time_start = __time()
            __result = terminal_script.run(__global)
            if not (RelayData.isRelayData(__result)): print(__result.args[0])
            if candy_setting["timer"]:
                print("Time elapsed: %.6f sec"%(__time()-time_start))
        except HandledException as he: print("\n".join(he.__notes__))
        except EOFError as eofe: break
        except KeyboardInterrupt as ki: pass
        except Exception as e:
            print("\nAn internal error has occurred.")
            raise