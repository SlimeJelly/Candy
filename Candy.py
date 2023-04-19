import re
from types import FunctionType
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass
from copy import deepcopy

@dataclass(frozen=True)
class Argument():
    name:str

FunctionList: Dict[str, "Function"] = {}

class Function():
    def __init__(self, name: str, tokens: List[Union[str, Argument]], func:FunctionType):
        self.name = name
        self.tokens = tokens
        self.__args: List[Argument] = [token for token in tokens if type(token) == Argument]
        self.func = func
        FunctionList[name] = self
    def __call__(self, *variables: List[Any]) -> Any:
        if len(self.__args) != len(variables): pass # error
        localDict: Dict[str, Any] = {arg_slot.name:var for arg_slot, var in zip(self.__args, variables)}
        return self.func(**localDict)

def get_py_type_check_func(_Type: type) -> bool: return lambda obj: type(obj) == _Type

Void = type("Void", (), {})
void = Void()
del Void

@dataclass(frozen=True)    
class StatementStack():
    colRange: Tuple[int, int]
    depth: int

class StatementStackSet():
    def __init__(self) -> None:
        self.stacks: List[StatementStack] = [] # (C:\\Users\\user\\Desktop\\SpookyLang.py, 1, 45, 2)
    def lookat(self, colRange: Tuple[int, int]) -> None:
        self.stacks[-1].colRange = colRange
    def enter(self, newStack: StatementStack) -> None:
        self.stacks.append(newStack)
    def exit(self) -> None:
        self.stacks.pop()
    @property
    def last(self) -> StatementStack: return self.stacks[-1]
    @property
    def history(self) -> Tuple[StatementStack]:
        # result = [self.stacks[0]]
        # for stack in self.stacks[1:]:
        #     if stack.line == result[-1].line:
        #         if stack.depth > result[-1].depth: result[-1] = stack
        #     else: result.append(stack)
        # return tuple(result)
        return tuple(self.stacks)

@dataclass(frozen=True)
class Stack():
    loc: str # file
    line: int
    statement: StatementStackSet

class StackSet():
    def __init__(self) -> None:
        self.stacks: List[Stack] = []
    def enter(self, newStack: Stack) -> None:
        self.stacks.append(newStack)
    def exit(self) -> None:
        self.stacks.pop()
    @property
    def last(self) -> Stack: return self.stacks[-1]
    @property
    def history(self) -> Tuple[Stack]:
        return tuple(self.stacks)


class __ExpressionException(Exception):
    def __init__(self, exception: Exception, stack: StatementStackSet) -> None:
        self.exception = exception
        self.stack = stack

class HandledException(Exception):
    def __init__(self, exception: Exception, stack: StackSet) -> None:
        self.exception = exception
        self.stack = stack

EXPRESSION_SPACE = "[ |\t]+"
EXPRESSION_VARIABLE = "[a-z|A-Z][\w]*"
COMPILED_EXPRESSION_VARIABLE = re.compile(EXPRESSION_VARIABLE)
COMPILED_EXPRESSION_STRING = re.compile("\"[^\"\n]*\"")
COMPILED_EXPRESSION_INTEGER = re.compile("[0-9]+")
COMPILED_EXPRESSION_DECIMAL = re.compile("[0-9]*.[0-9]+")
COMPILED_EXPRESSION_SET = re.compile("set"+EXPRESSION_SPACE+"("+EXPRESSION_VARIABLE+")"+EXPRESSION_SPACE+"to"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_FUNCTION = re.compile("(do|get)"+EXPRESSION_SPACE+"("+EXPRESSION_VARIABLE+")"+EXPRESSION_SPACE+"([^\n]+)")
COMPILED_EXPRESSION_WORD = re.compile("\\w+")

def putDefault(__dict: Dict[str, Any] = None):
    if __dict == None: __dict = {}
    defaultKeys = list(__dict.keys())
    def __print(text): print(text)
    def __combine(t1, t2): return str(t1)+str(t2)
    def __repeat(text, count): return str(text)*count
    if not "print"   in defaultKeys: __dict["print"]   = Function("print",   ["of", Argument("text")], __print)
    if not "combine" in defaultKeys: __dict["combine"] = Function("combine", ["of", Argument("t1"), "and", Argument("t2")], __combine)
    if not "repeat"  in defaultKeys: __dict["repeat"]  = Function("repeat",  [Argument("text"), "for", Argument("count"), "times"], __repeat)

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
    return split_result

def checkSyntax(__source: str) -> bool:
    if __source == "" or re.match("[ |\t]+", __source) != None: return True
    elif __source.startswith("set"):
        match_define = COMPILED_EXPRESSION_SET.match(__source)
        return match_define != None and COMPILED_EXPRESSION_VARIABLE.match(match_define.group(1)) != None and checkSyntax(match_define.group(2))
    elif __source.startswith("do") or __source.startswith("get"):
        match_function = COMPILED_EXPRESSION_FUNCTION.match(__source)
        return match_function != None and COMPILED_EXPRESSION_VARIABLE.match(match_function.group(2)) != None \
         and not False in [(
            checkSyntax(token[1:-1]) if token.startswith("(") and token.endswith(")") else COMPILED_EXPRESSION_WORD.match(token) != None
        ) for _, token in splitArguments(match_function.group(3)) if token != ""]
    return COMPILED_EXPRESSION_STRING.match(__source) != None \
            or COMPILED_EXPRESSION_DECIMAL.match(__source) != None \
            or COMPILED_EXPRESSION_INTEGER.match(__source) != None \
            or COMPILED_EXPRESSION_VARIABLE.match(__source) != None

def _eval(expression: str, __globals: Dict[str, Any] = None, __stack: StatementStackSet = None) -> Any:
    #print(expression, script.split("\n")[__stack.last.line][__stack.last.colRange[0]:__stack.last.colRange[1]], __stack.last, sep="\n", end="\n\n")
    if __stack.history == (): __stack.enter(StatementStack((0, len(expression)), 0))
    if __globals == None: __globals = {}
    putDefault(__globals)
    try:
        if expression == "": return ""
        if expression.startswith("set"):
            match_define = COMPILED_EXPRESSION_SET.match(expression)
            if match_define == None: return SyntaxError()
            #if isinstance(value, Exception): return value

            argName = match_define.group(1)

            newRange = __stack.last.colRange
            newRange = (newRange[0]+4+len(argName)+4, newRange[1])
            __stack .enter(StatementStack(newRange, __stack.last.depth+1))
            value = _eval(match_define.group(2), __globals, __stack)
            if COMPILED_EXPRESSION_VARIABLE.match(argName) == None: return SyntaxError()
            if value is void: return SyntaxError()

            __globals[argName] = value
            return void
        if expression.startswith("do") or expression.startswith("get"):
            match_function = COMPILED_EXPRESSION_FUNCTION.match(expression)
            if match_function == None: return SyntaxError()
            split_result = splitArguments(match_function.group(3))
            parse_result: List[str] = []
            __col_st, _ = __stack.last.colRange
            __col_st += (3 if expression.startswith("do") else 4) + len(match_function.group(2)) + 1
            for sps in split_result: # split string
                item = sps[1]
                if item == "": continue
                if sps[0] and item.startswith("(") and item.endswith(")"):
                    __stack.enter(StatementStack((__col_st+1, __col_st+len(sps[1])-1), __stack.last.depth+1))
                    parse_result.append(_eval(item[1:-1], __globals, __stack))
                __col_st += len(item) + 1
            print(split_result, parse_result)
            result:Any = eval(match_function.group(2)+"("+",".join(["__v"+str(i) for i in range(len(parse_result))])+")", __globals, {"__v"+str(i):v for i, v in enumerate(parse_result)})
            return result if match_function.group(1) == "get" else void
        if COMPILED_EXPRESSION_STRING  .match(expression) != None: return eval(expression)
        if COMPILED_EXPRESSION_DECIMAL .match(expression) != None: return float(expression)
        if COMPILED_EXPRESSION_INTEGER .match(expression) != None: return int(expression)
        if COMPILED_EXPRESSION_VARIABLE.match(expression) != None: return eval(expression, __globals)
        raise SyntaxError()
    except __ExpressionException: raise
    except Exception as e: raise __ExpressionException(e, deepcopy(__stack))
    finally: __stack.exit()

def _exec(__source: str, __globals: Dict[str, Any] = None,  __file: str = "<string>"):
    try:
        __wrong_syntax_lines: Tuple[int] = tuple([i for i, line in enumerate(__source.split("\n")) if not checkSyntax(line)])
        if len(__wrong_syntax_lines) > 0:
            print("SyntaxError while parsing script: \""+__file+"\"")
            line_splitted_script = __source.split("\n")
            for line in __wrong_syntax_lines:
                print("  line %d :: "%(line+1)+line_splitted_script[line])
            return

        if __globals == None: __globals = {}
        putDefault(__globals)

        lines = __source.split("\n")
        __stack = StackSet()

        for i, line in enumerate(lines):
            __stack.enter(Stack(__file, i, StatementStackSet()))
            # if __stack == None:
            #     __stack = StatementStackSet()
            # __stack.enter(StatementStack((0, len(lines[i])), 0))
            #print("==============", __stack.last)
            try: _eval(line, __globals, __stack.last.statement)
            except __ExpressionException as e:
                __last_stack = __stack.last
                __copied_stack = deepcopy(__stack)
                __copied_stack.exit()
                __copied_stack.enter(Stack(__last_stack.loc, __last_stack.line, e.stack))
                raise HandledException(e, __copied_stack)
            __stack.exit()
    except HandledException as e:
        for stack in e.stack.history:
            stack_statement = stack.statement.last
            print("Traceback:")
            print("  "+"File \""+stack.loc+"\", line "+str(stack.line+1))
            print("  "+"  "+lines[stack.line])
            print("  "+"  "+" "*stack_statement.colRange[0] + "^"*(stack_statement.colRange[1]-stack_statement.colRange[0]))
        print(type(e.exception.exception).__name__ +": "+ str(e.exception.exception))