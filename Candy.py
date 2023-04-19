from enum import Enum, auto
from types import FunctionType
from typing import Any, Dict, List, NoReturn, Set, Tuple, Union, Iterator
from dataclasses import dataclass
from copy import copy, deepcopy
import re

@dataclass(frozen=True)
class Void():
    data: str
void = Void("EMPTY")
void_do_exp = Void("RETURN_OF_DO_EXP")
void_set_exp = Void("RETURN_OF_SET_EXP")
void_missing_arg = Void("MISSING_ARG")

@dataclass(frozen=True)
class Argument():
    name:str
    auto:Any=void_missing_arg

class Function():
    def __init__(self, name: str, tokens: List[Union[str, Argument]], func:FunctionType):
        self.name: str = name
        self.tokens: Set[str] = set([token for token in tokens if type(token) == str])
        self.__args: List[Argument] = [token for token in tokens if type(token) == Argument]
        self.less_args = len([arg for arg in self.__args if arg.auto is void_missing_arg])
        self.func = func
    def __call__(self, variables: List[Any], _globals: Dict[str, Any], _file: str, _stack: "StackSet") -> Any:
        if len(self.__args) != len(variables): pass # error
        if len(variables) < self.less_args: raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        variables += (void_missing_arg,) * (len(self.__args) - len(variables))
        localDict: Dict[str, Any] = {arg_slot.name:(var if type(var) != Void else arg_slot.auto) for arg_slot, var in zip(self.__args, variables)}
        if void_missing_arg in localDict.values(): raise TypeError(self.name+" expected at least "+str(self.less_args)+" argument, got "+str(len(variables)))
        return self.func(localDict, _globals, _file, _stack)

def get_py_type_check_func(_Type: type) -> bool: return lambda obj: type(obj) == _Type

@dataclass(frozen=True)    
class StatementStack():
    colRange: Tuple[int, int]
    depth: int

class StatementStackSet():
    def __init__(self) -> None: self.stacks: List[StatementStack] = []
    def lookat(self, colRange: Tuple[int, int]) -> None: self.stacks[-1].colRange = colRange
    def enter(self, newStack: StatementStack) -> None: self.stacks.append(newStack)
    def exit(self) -> None: self.stacks.pop()
    @property
    def last(self) -> StatementStack: return self.stacks[-1]
    @property
    def history(self) -> Tuple[StatementStack]: return tuple(self.stacks)

@dataclass(frozen=True)
class Stack():
    loc: str # file
    line: int

class StackSet():
    def __init__(self) -> None: self.stacks: List[Stack] = []
    def enter(self, newStack: Stack) -> None: self.stacks.append(newStack)
    def exit(self) -> None: self.stacks.pop()
    @property
    def last(self) -> Stack: return self.stacks[-1]
    @property
    def history(self) -> Tuple[Stack]: return tuple(self.stacks)


class __ExpressionException(Exception):
    def __init__(self, exception: Exception, stack: StatementStackSet) -> None:
        self.exception = exception
        self.curStack = stack

class HandledException(Exception):
    def __init__(self, exception: "__ExpressionException", stack: StackSet) -> None:
        self.exception = exception.exception
        self.stack = stack
        self.colStack = exception.curStack

class CodeType(Enum):
    #System (0~)
    EMPTY = 0

    #Eval (100~)
    EVAL_INTEGER = 100
    EVAL_DECIMAL = 101
    EVAL_STRING = 102
    EVAL_BOOLEAN = 103
    EVAL_VARIABLE = 104

    EVAL_EXECUTE = 110
    EVAL_SET = 111
    EVAL_CALC = 112

    #Exec (200~)
    #Exec - Block (200~)
    EXEC_CONDITION_IF = 200
    EXEC_CONDITION_ELIF = 201
    EXEC_CONDITION_ELSE = 202

    EXEC_LOOP_FOR = 210
    EXEC_LOOP_WHILE = 220

    #Exec - DataHolder (250~)
    EXEC_DEFINE_FUNCTION = 250

class Code():
    def __init__(self, codeType: CodeType, stateStack: StatementStack, **kw: Dict[str, Any]) -> None:
        self.codeType = codeType
        self.stack = stateStack
        self.kw = kw
    @property
    def is_eval(self): return 200 > self.codeType.value >= 100
    @property
    def is_exec(self): return 300 > self.codeType.value >= 200
    @property
    def is_dataholder(self): return 300 > self.codeType.value >= 250
    def __str__(self) -> str: return f"Code::{self.codeType} (at {self.stack}) [{self.kw}]"

class LineTree():
    def __init__(self, line: int, code: Code, master: Union[Void, "LineTree"] = void):
        self.line = line
        self.code = code
        self.master = master
        self.childs: List["LineTree"] = []
    def add_child(self, child: "LineTree") -> None:
        child.master = self
        self.childs.append(child)
    def __str__(self) -> str: return "["+str(self.line)+"]"+"( "+", ".join([str(l) for l in self.childs])+" )"

class SyntaxResultType(Enum):
    Right = 0

    Identition = auto()
    UnexpectedSyntax = auto()
    MixedIdentition = auto()
    MultiTab = auto()
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

# Inner classes
class Infinity():
    def __init__(self) -> None: self.sign = True
    def __str__(self) -> str: return ("+" if self.sign else "-") + "Infinity"
    def __repr__(self) -> str: return self.__str__()
    def __bool__(self) -> bool: return True
    def __eq__(self, __value: object) -> bool: return type(__value) == Infinity and self.sign == __value.sign
    def __ne__(self, __value: object) -> bool: return not self.__eq__(__value)
    def __hash__(self) -> int: return 0
    def __abs__(self) -> "Infinity": self.sign = True; return self
    def __neg__(self) -> "Infinity": self.sign = not self.sign; return self
    def __lt__(self, __value: object) -> bool: return False if type(__value) != Infinity else (True if (not self.sign) and __value.sign else False)
    def __le__(self, __value: object) -> bool: return False if type(__value) != Infinity else (self < __value or self == __value)
    def __gt__(self, __value: object) -> bool: return False if type(__value) != Infinity else (False if (not self.sign) and __value.sign else True)
    def __ge__(self, __value: object) -> bool: return False if type(__value) != Infinity else (self > __value or self == __value)

def create_function(name: str, parameters: Iterator[Union[str, Argument]], code: Iterator[LineTree]) -> Function:
    def __func(arguments: Dict[str, Any], __globals: Dict[str, Any], __file: str, __stack: StackSet) -> NoReturn:
        __locals = copy(__globals)
        __locals.update(arguments)
        _run_tree(code, __locals, __file, __stack)
    return Function(name, parameters, __func)

def create_py_function(func_name:str, tokens: List[Union[str, Argument]], func: FunctionType): return Function(func_name, tokens, lambda arguments, _1, _2, _3: func(**arguments))

EXPRESSION_SPACE = "[ |\t]+"
EXPRESSION_VARIABLE = "[a-z|A-Z][\w|\d|_]*"
COMPILED_EXPRESSION_EMPTY = re.compile("[ |\t]+")
COMPILED_EXPRESSION_VARIABLE = re.compile(EXPRESSION_VARIABLE)
COMPILED_EXPRESSION_STRING = re.compile("\"[^\"\n]*\"")
COMPILED_EXPRESSION_INTEGER = re.compile("[0-9]+")
COMPILED_EXPRESSION_DECIMAL = re.compile("[0-9]*.[0-9]+")
COMPILED_EXPRESSION_BOOLEAN = re.compile("(true|false)")
COMPILED_EXPRESSION_SET = re.compile("set"+EXPRESSION_SPACE+"("+EXPRESSION_VARIABLE+")"+EXPRESSION_SPACE+"to"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_FUNCTION = re.compile("(do|get)"+EXPRESSION_SPACE+"("+EXPRESSION_VARIABLE+")"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_WORD = re.compile("\\w+")
COMPILED_EXPRESSION_BLOCKHOLDER = re.compile("(if|elif|else|for|while)"+"([^\n]*):")

OPERATOR_TOKENS = ("+", "-", "*", "/", "%", "//", "**", 
                   "==", "!=", "<", ">", "<=", ">=",)

def putDefault(__dict: Dict[str, Any] = None):
    if __dict == None: __dict = {}
    defaultKeys = list(__dict.keys())
    def __print(text): print(text)
    def __combine(t1, t2): return str(t1)+str(t2)
    def __repeat(text, count): return str(text)*count
    if not "print"    in defaultKeys: __dict["print"]    = create_py_function("print",    ["of", Argument("text")], __print)
    if not "combine"  in defaultKeys: __dict["combine"]  = create_py_function("combine",  ["of", Argument("t1"), "and", Argument("t2")], __combine)
    if not "repeat"   in defaultKeys: __dict["repeat"]   = create_py_function("repeat",   ["of", Argument("text"), "for", Argument("count"), "times"], __repeat)
    if not "range"    in defaultKeys: __dict["range"]    = create_py_function("range",    ["of", Argument("start"), "to", Argument("end"), "by", Argument("sep", auto=1)], lambda start, end, sep: range(start, end, sep))
    if not "asString" in defaultKeys: __dict["asString"] = create_py_function("asString", ["of", Argument("text")], lambda text: str(text))
    if not "asInt"    in defaultKeys: __dict["asInt"]    = create_py_function("asInt",    ["of", Argument("text")], lambda text: int(text))
    if not "asFloat"  in defaultKeys: __dict["asFloat"]  = create_py_function("asFloat",  ["of", Argument("text")], lambda text: float(text))
    if not "asBool"   in defaultKeys: __dict["asBool"]   = create_py_function("asBool",   ["of", Argument("text")], lambda text: bool(text))
    if not "asList"   in defaultKeys: __dict["asList"]   = create_py_function("asList",   ["of", Argument("text")], lambda text: list(text))
    if not "infinity" in defaultKeys: __dict["infinity"] = Infinity()

def splitArguments(__source: str) -> List[str]:
    split_result: List[List[bool, str]] = [[True, ""]]
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
    return [tuple(v) for v in split_result if v[1] != ""]

def checkExpressionSyntax(__source: str) -> bool:
    __source = __source.lstrip()
    if __source == "" or COMPILED_EXPRESSION_EMPTY.fullmatch(__source) != None: return True

    # Eval
    elif __source.startswith("set"):
        match_define = COMPILED_EXPRESSION_SET.fullmatch(__source)
        return match_define != None and COMPILED_EXPRESSION_VARIABLE.fullmatch(match_define.group(1)) != None and checkExpressionSyntax(match_define.group(2))
    elif __source.startswith("do") or __source.startswith("get"):
        match_function = COMPILED_EXPRESSION_FUNCTION.fullmatch(__source)
        return match_function != None and COMPILED_EXPRESSION_VARIABLE.fullmatch(match_function.group(2)) != None \
         and not False in [(
            checkExpressionSyntax(token[1:-1]) if token.startswith("(") and token.endswith(")") else COMPILED_EXPRESSION_WORD.fullmatch(token) != None
        ) for _, token in splitArguments(match_function.group(3))]
    elif COMPILED_EXPRESSION_STRING.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_DECIMAL.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_INTEGER.fullmatch(__source) != None \
        or COMPILED_EXPRESSION_VARIABLE.fullmatch(__source) != None: return True
    splits = splitArguments(__source)
    __IsValueList = [_type for _type, _ in splits]
    if (len(splits) >=3 and __IsValueList[:-1] and (not False in [__IsValueList[i] == (i % 2 == 0) for i in range(len(__IsValueList))])):
        return not False in [len(splits[i][1]) >= 2 
                                and splits[i][1][0] == "(" 
                                and splits[i][1][-1] == ")" 
                                and checkExpressionSyntax(splits[i][1][1:-1]) 
                             for i in range(len(splits)) 
                             if i % 2 == 0
                            ]

    # Exec
    if (__source.startswith("if") or __source.startswith("elif") or __source.startswith("else")
        or __source.startswith("while") or __source.startswith("for")
    ):
        match_blockholder = COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(__source)
        if match_blockholder == None: return False
        textgroup = match_blockholder.group(2).lstrip().rstrip()
        codeTypeString = match_blockholder.group(1)
        if codeTypeString in ("else"): 
            return len(textgroup) == 0
        elif codeTypeString in ("if", "elif", "while"):
            sa = splitArguments(textgroup)
            return len(sa) == 1 and sa[0][0] and checkExpressionSyntax(sa[0][1][1:-1])
        elif codeTypeString in ("for"):
            sa = splitArguments(textgroup)
            return len(sa) == 3 and (not sa[0][0]) and (not sa[1][0]) and sa[2][0] and \
                COMPILED_EXPRESSION_VARIABLE.fullmatch(sa[0][1]) \
                and sa[1][1] == "in" and checkExpressionSyntax(sa[2][1][1:-1])
        #return len(textgroup) == 0 or (textgroup.startswith("(") and textgroup.endswith(")") and checkExpressionSyntax(textgroup))

    return COMPILED_EXPRESSION_VARIABLE.fullmatch(__source) != None

def checkSyntax(__source: str) -> Tuple[SyntaxResultType, Tuple[int, ...]]:
    first_not_empty_line = [(i, line) for i, line in enumerate(__source.split("\n")) if line != "" and COMPILED_EXPRESSION_EMPTY.fullmatch(line) == None]
    if len(first_not_empty_line) == 0: return (SyntaxResultType.Right, ())
    if first_not_empty_line[0][1][0] in (" ", "\t"): return (SyntaxResultType.UnexpectedIndent, (0, ))
    del first_not_empty_line

    __wrong_syntax_lines: Tuple[int] = tuple([i for i, line in enumerate(__source.split("\n")) if not checkExpressionSyntax(line)])
    if len(__wrong_syntax_lines) > 0: return (SyntaxResultType.UnexpectedSyntax, __wrong_syntax_lines)
    
    last_ident = 0
    lines = __source.split("\n")
    first_identify: Union[str, Void] = void
    for l, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")): first_identify, last_ident = void, 0
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

def _compile(expression: str, __stack: StatementStackSet = None) -> Code:
    if __stack == None: __stack = StatementStackSet()
    if __stack.history == (): __stack.enter(StatementStack((0, len(expression)), 0))
    try:
        if expression == "": return Code(CodeType.EMPTY, deepcopy(__stack))
        elif COMPILED_EXPRESSION_INTEGER .fullmatch(expression) != None: return Code(CodeType.EVAL_INTEGER , deepcopy(__stack), value=int  (expression))
        elif COMPILED_EXPRESSION_DECIMAL .fullmatch(expression) != None: return Code(CodeType.EVAL_DECIMAL , deepcopy(__stack), value=float(expression))
        elif COMPILED_EXPRESSION_STRING  .fullmatch(expression) != None: return Code(CodeType.EVAL_STRING  , deepcopy(__stack), value=eval (expression))
        elif COMPILED_EXPRESSION_BOOLEAN .fullmatch(expression) != None: return Code(CodeType.EVAL_BOOLEAN , deepcopy(__stack), value=bool (expression))
        elif expression.startswith("set"):
            match_define = COMPILED_EXPRESSION_SET.fullmatch(expression)
            if match_define == None: return SyntaxError()

            argName = match_define.group(1)

            newRange = __stack.last.colRange
            newRange = (newRange[0]+4+len(argName)+4, newRange[1])
            __stack .enter(StatementStack(newRange, __stack.last.depth+1))
            value = _compile(match_define.group(2), __stack)
            if COMPILED_EXPRESSION_VARIABLE.fullmatch(argName) == None: return SyntaxError()
            if type(value) == Void: return SyntaxError()
            
            return Code(CodeType.EVAL_SET, deepcopy(__stack), var=argName, value=value)
        elif expression.startswith("do") or expression.startswith("get"):
            match_function = COMPILED_EXPRESSION_FUNCTION.fullmatch(expression)
            if match_function == None: return SyntaxError()
            split_result = splitArguments(match_function.group(3))
            parse_result: List[Union[str, Code]] = []
            __col_st, _ = __stack.last.colRange
            __col_st += (3 if expression.startswith("do") else 4) + len(match_function.group(2)) + 1
            for sps in split_result: # split string
                item = sps[1]
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    __stack.enter(StatementStack((__col_st+1, __col_st+len(item)-1), __stack.last.depth+1))
                    parse_result.append(_compile(item[1:-1], __stack))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(CodeType.EVAL_EXECUTE, deepcopy(__stack), target=match_function.group(2), args=tuple(parse_result), return_result=match_function.group(1) == "get")
        elif COMPILED_EXPRESSION_VARIABLE.fullmatch(expression) != None: return Code(CodeType.EVAL_VARIABLE, deepcopy(__stack), value=expression)
        elif COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression) != None:
            codeTypeString = COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(1)
            codeType = {
                "if"   : CodeType.EXEC_CONDITION_IF,
                "elif" : CodeType.EXEC_CONDITION_ELIF,
                "else" : CodeType.EXEC_CONDITION_ELSE,
                "for"  : CodeType.EXEC_LOOP_FOR,
                "while": CodeType.EXEC_LOOP_WHILE
            }[codeTypeString]
            split_result = splitArguments(COMPILED_EXPRESSION_BLOCKHOLDER.fullmatch(expression).group(2))
            parse_result: List[Union[str, Code]] = []
            __col_st, _ = __stack.last.colRange
            __col_st += len(codeTypeString) + 1
            for sps in split_result: # split string
                item = sps[1]
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    __stack.enter(StatementStack((__col_st+1, __col_st+len(item)-1), __stack.last.depth+1))
                    parse_result.append(_compile(item[1:-1], __stack))
                else: parse_result.append(item)
                __col_st += len(item) + 1
            return Code(codeType, deepcopy(__stack), args=tuple(parse_result))
        else:
            __splits = splitArguments(expression)
            __col_st, _ = __stack.last.colRange
            i = 0
            __args = []
            __evalCode = ""
            for __type, __value in __splits:
                if not __type: __evalCode += __value
                else:
                    i += 1
                    __stack.enter(StatementStack((__col_st+1, __col_st+len(__value)-1), __stack.last.depth+1))
                    __args.append(_compile(__value[1:-1], __stack))
                    __evalCode += "__args[" + str(i-1) + "]"
                __col_st += len(__value) + 1
            return Code(CodeType.EVAL_CALC, deepcopy(__stack), evalCode=__evalCode, args=tuple(__args))
    except Exception: raise
    finally: __stack.exit()

def _eval(__code: Code, __globals: Dict[str, Any], __file: str, __stack: StackSet):
    try:
        if __code.codeType == CodeType.EMPTY: return void
        if __code.codeType == CodeType.EVAL_INTEGER: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_DECIMAL: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_STRING:  return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_BOOLEAN: return __code.kw["value"]
        if __code.codeType == CodeType.EVAL_VARIABLE:
            if __code.kw["value"] in __globals: return __globals[__code.kw["value"]]
            else: raise NameError("name '"+__code.kw["value"]+"' is not defined")
        if __code.codeType == CodeType.EVAL_EXECUTE:
            if not (__code.kw["target"] in __globals):
                raise NameError("name '"+__code.kw["target"]+"' is not defined")
            target: Function = __globals[__code.kw["target"]]
            arguments = tuple([_eval(arg, __globals, __file, __stack) for arg in __code.kw["args"] if isinstance(arg, Code)])
            for value in arguments:
                if value is void_do_exp: raise SyntaxError("Function arguments must not be DO expression")
                if value is void_set_exp: raise SyntaxError("Function arguments must not be SET expression")
            result = target(arguments, __globals, __file, __stack)
            return result if __code.kw["return_result"] else void_do_exp
        if __code.codeType == CodeType.EVAL_SET:
            value = _eval(__code.kw["value"], __globals, __file, __stack)
            if value is void_do_exp: raise SyntaxError("SET expression must not be DO expression")
            if value is void_set_exp: raise SyntaxError("SET expression must not be SET expression")
            __globals[__code.kw["var"]] = value
            return void_set_exp
        if __code.codeType == CodeType.EVAL_CALC: return eval(__code.kw["evalCode"], {"__args": [_eval(arg, __globals, __file, __stack) for arg in __code.kw["args"]]})
    except __ExpressionException: raise
    except Exception as e: raise __ExpressionException(e, __code.stack)

def _treeMap(__source: str) -> List[LineTree]:
    master: List[LineTree] = []
    last_ident = 0
    lines = __source.split("\n")
    first_identify: Union[str, Void] = void
    for i, line in enumerate(lines):
        if line == "": continue
        elif not (line[0] in (" ", "\t")):
            master.append(LineTree(i, _compile(line)))
            first_identify = void
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

def _run_tree(master: List[LineTree], __globals: Dict[str, Any], __file: str, __stack: StackSet = None):
    if __stack == None: __stack = StackSet()
    __IGNORE_IF = False
    for tree in master:
        __stack.enter(Stack(__file, tree.line))
        try:
            if tree.code.is_eval: _eval(tree.code, __globals, __file, __stack)
            elif tree.code.is_exec:
                if tree.code.is_dataholder: original_keys = __globals.keys()


                # Execute Eval_~~~ code
                if tree.code.codeType == CodeType.EXEC_CONDITION_IF:
                    __IGNORE_IF = False
                    if _eval(tree.code.kw["args"][0], __globals, __file, __stack):
                        __IGNORE_IF = True
                        _run_tree(tree.childs, __globals, __file, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELIF:
                    if (not __IGNORE_IF) and _eval(tree.code.kw["args"][0], __globals, __file, __stack):
                        __IGNORE_IF = True
                        _run_tree(tree.childs, __globals, __file, __stack)
                elif tree.code.codeType == CodeType.EXEC_CONDITION_ELSE:
                    if (not __IGNORE_IF):
                        __IGNORE_IF = True
                        _run_tree(tree.childs, __globals, __file, __stack)
                else: __IGNORE_IF = False

                if tree.code.codeType == CodeType.EXEC_LOOP_WHILE:
                    while _eval(tree.code.kw["args"][0], __globals, __file, __stack):
                        _run_tree(tree.childs, __globals, __file, __stack)
                elif tree.code.codeType == CodeType.EXEC_LOOP_FOR:
                    __iter = _eval(tree.code.kw["args"][2], __globals, __file, __stack)
                    __var = tree.code.kw["args"][0]
                    for __item in __iter:
                        __globals[__var] = __item
                        _run_tree(tree.childs, __globals, __file, __stack)
                
                if tree.code.is_dataholder: __globals = {key: __globals[key] for key in original_keys}
        except __ExpressionException as e:
            __last_stack = __stack.last
            __copied_stack = deepcopy(__stack)
            __copied_stack.exit()
            __copied_stack.enter(Stack(__last_stack.loc, __last_stack.line))
            raise HandledException(e, __copied_stack)
        __stack.exit()
    
def _exec(__source: str, __globals: Dict[str, Any] = None,  __file: str = "<string>"):
    __syntax, __err_lines = checkSyntax(__source)
    if __syntax != SyntaxResultType.Right:
        print("Found wrong syntax while parsing script: \""+__file+"\"")
        for __err_line in __err_lines: print("[%s] at line %d `%s`"%(__syntax.name, __err_line, __source.split("\n")[__err_line]))
        return ExecuteResult(ExecuteResultType.CompileError, ("SyntaxError", __syntax), __err_lines)
    
    try:
        
        master = _treeMap(__source)

        if __globals == None: __globals = {}
        putDefault(__globals)

        _run_tree(master, __globals, __file)
        return ExecuteResult(ExecuteResultType.Success)
    except HandledException as e:
        stack_statement = e.colStack.last
        print("Traceback:")
        for stack in e.stack.history:
            print("  "+"File \""+stack.loc+"\", line "+str(stack.line+1))
            print("  "+"  "+__source.split("\n")[stack.line])
            print("  "+"  "+" "*stack_statement.colRange[0] + "^"*(stack_statement.colRange[1]-stack_statement.colRange[0]))
        print(e.exception.__class__.__name__ +": "+ str(e.exception))
        return ExecuteResult(ExecuteResultType.RuntimeError, e.exception, e.stack.history, stack_statement)