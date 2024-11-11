"""
This module holds all the libraries and functions I use
"""
## TODO: Need to write some unit tests especially to check against different python versions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from plotnine import * ## For using R's ggplot in python ##
from io import StringIO
import json
from bs4 import BeautifulSoup
import requests
from IPython.display import display,Markdown,HTML,Javascript,clear_output,Code
from IPython import get_ipython
#import html5lib
import subprocess
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#import datetime
import os
from threading import Thread,RLock
lock=RLock()
from time import time,sleep
from types import ModuleType,BuiltinFunctionType,FrameType,FunctionType,MethodType
from typing import Any,Callable,NoReturn,Union,Iterable,Generator
import tempfile
from importlib.util import module_from_spec,spec_from_loader
########### for installing editable local libraries and figuring out what modules python scripts use
import re
import glob
import shutil
from inspect import getfile,getsource,signature,_empty,currentframe,stack,isclass
import sys
from functools import partial,wraps,reduce
from keyword import iskeyword
import pickle
import webbrowser
import pyautogui
from tkinter import Tk
import secrets
import string
from operator import itemgetter
from itertools import combinations,chain as iter_chain # since I have a class called 'chain'
import IPython
from warnings import simplefilter,warn
#import traceback
import ast
import dis
import importlib
import psutil
import dill
import ctypes
from copy import deepcopy
import urllib
import readline
#from collections.abc import Iterable
from contextlib import contextmanager
from collections import deque
import pickletools

module_dir=os.path.dirname(__file__)

@lambda x: x()
class wrap:
    """
    Decorator for wrapping functions with other functions
    i.e.
    How to use:
    
    def do2(string):
        return str(string)

    @wrap(do2)
    def do(string):
        return int(string)

    do(3.0)
    # should print
    # '3'
    """
    __inplace=False
    @property
    def inplace(self) -> object:
        self=copy(self) ## otherwise __inplace stays set to True since we're not instantiating e.g. reusing the same instance for every use
        self.__inplace=True
        return self
    
    def __call__(self,FUNC: Callable,*wrap_args,**wrap_kwargs) -> Callable:
        def wrap_wrapper(func: Callable) -> Callable: # function to wrap
            @wraps(func) # transfers the docstring since the functions redefined as the wrapper
            def wrapper(*args,**kwargs) -> Any: # its args
                inner=func(*args,**kwargs)
                if self.__inplace:
                    FUNC(inner,*wrap_args,**wrap_kwargs)
                    return inner
                return FUNC(inner,*wrap_args,**wrap_kwargs)
            return wrapper
        return wrap_wrapper

def to_pickle(obj: object,filename: str,force: bool=False) -> None:
    """Convenience function for pickling objects in python with context management"""
    if filename[-4:]!='.pkl': filename+='.pkl'
    with open(filename,'wb') as file:
        if force: return dill.dump(obj, file)
        pickle.dump(obj, file)

def read_pickle(filename: "str",force: bool=False) -> Any:
    """Convenience function for reading pickled objects in python with context management"""
    if filename[-4:]!='.pkl': filename+='.pkl'
    with open(filename, 'rb') as file:
        if force: return dill.load(file)
        return pickle.load(file)

## needs more work done ##
class pickle_stack:
    """Stack for deserializing pickle opcodes/names and args into readable python code"""
    #### use single underscore for names for access and separation (i.e. on dir usage) ####
    # for determining opcode type
    _opcode_map={opcode.name:index for index,opcode in enumerate(pickletools.opcodes)}
    ## for determining push opcodes value
    _push_map=dict(zip((opcode.name for index,opcode in enumerate(pickletools.opcodes) if index < 26 or index in (29,34,38)),
                       (*(int,)*7,*(str,)*3,*(bytes,)*3,bytearray,...,...,None,True,False,*(str,)*4,float,float,[],(),{},set()),
                       strict=True)) ## both tuples must be the same length

    @classmethod
    def read(cls,filename: str) -> object:
        """Convenience method to initialize the stack with opcodes"""
        return cls(analyze_pickle(filename))
    
    def __init__(self,opcodes: Generator) -> None:
        """
        stack: the pickle stack
        marks: position of all current marks in the stack
        memo: memoization cache
        code: the final python code in string form
        """
        self.stack,self.marks,self.memo,self.code=*(deque(),)*2,{},""
        for opcode, arg, pos in opcodes: self.stack_operation(opcode,arg)

    def __repr__(self) -> str: return self.code
    
    @property
    def pop(self) -> Generator:
        """pops the stack until reaching the mark position"""
        return (self.stack.pop() for i in range(len(self.stack) - (self.mark.pop() if len(self.mark) else 0)))
    
    def stack_operation(self,obj,arg: str) -> None:
        """
        Figures out if the opcode is stacking or stack manipulating
        then performs the relevant operation to the pickle stack
        """
        index=self._opcode_map[obj.name]
        if 62 < index < 66: ## special opcodes e.g. PROTO,STOP,FRAME
            return
        if index < 26 or index in (29, 34, 38, 43): ## push opcodes (pushes a value onto the stack)
            if obj.name=="MARK": return self.marks.append(len(self.stack))
            mapping=self._push_map[obj.name]
            if type(mapping)==type: return self.stack.append(mapping(arg))
            else: return self.stack.append(mapping)
        if 44 < index < 52: ## memo opcodes
            return
            if index < 48: ## memo get
                return self.stack.append(self.memo[arg]) ## ---------------------- what are the setters?? Needs implementing ##
            else: ## memo put
                self.memo[arg]=arg ## supposed to be the top of the stack apparently
            return
        #######################################################
        ### class instantiation
        #######################################################
        if 58 < index < 63: ## class opcodes
            return ## --------------------------------------------- needs implementing ##
        if 51 < index < 57: ## GET PUSH values opcodes
            if 54 < index: ## global attrs
                ## GLOBAL is proto 3 and text based apparently - likely global classes
                ## STACK_GLOBAL is proto 4 and binary based apparently - global variables
                value=... ## ------------------------------------ needs implementing ##
            else: ## registry attrs
                value=... ## ------------------------------------ needs implementing ##
            # push onto stack
            return self.stack.append(value)
        if 65 < index: ## GET PUSH values opcodes for persistent attrs
            value=... ## --------------------------------------------- needs implementing ##
            return self.stack.append(value)
        match index: ## object wrapping, object mutating, and stack manipulation opcodes
            case 26: # APPEND (list 1 item)
                item=self.stack.pop()
                value=self.stack.pop().append(item)
            case 27: # APPENDS (list > 1 item)
                items=self.pop
                value=self.stack.pop().append(items)
            case 28: # LIST --------------------------------------------- (comes before items)
                value=list(self.pop)
            case 30 | 31 | 32 | 33: # TUPLE # tuple with > 3 items # TUPLE1 # tuple with 1 item # TUPLE2 # tuple with 2 items # tuple # TUPLE with 3 items
                value=tuple(self.pop)
            case 35: # DICT
                value=dict(self.pop)
            case 36: # SETITEM (dict 1 item)
                dct=dict((self.pop,))
                value=self.stack.pop().update(dct)
                ((1,2),)
            case 37: # SETITEMS (dict > 1 item)
                
                dict(
                    (self.pop,)
                )
                
                dct=self.pop
                value=self.stack.pop().update(dct)
            case 39: # ADDITEMS # set() --------------------------------------------- (comes before items)
                value=set(self.pop)
            case 40: # FROZENSET ------------------------------------- check if it comes before or not
                value=frozenset(self.pop) # ------------------------------------------------- does MARK get popped??
            case 41: # POP (pops the stack)
                return self.stack.pop()
            case 42: # DUP (appends a duplication of the top item onto the stack)
                return self.stack.append(self.stack[-1])
            case 44: # POP_MARK (pops the last MARK and everything above it on the stack)
                tuple(self.pop)
                return self.stack.pop() # pop the MARK opcode
            case 57: # REDUCE () ---------------------- Needs checking
                value=arg(*self.pop)
            case 58: # BUILD () ---------------------- Needs checking
                args=self.pop
                self.stack.pop(args)
                value=...
        return self.stack.append(value)

def get_dunders() -> pd.DataFrame:
    """
    Retrieves a dataframe holding information about all dunder methods in python
    
    # reference: Hunner,T., (2024). Every dunder method in Python, https://www.pythonmorsels.com/every-dunder-method/, python morsels
    """
    __dunders__=read_pickle(os.path.join(module_dir,"dunders"))
    __dunders__.__version__="3.9-3.11"
    return __dunders__

## might remove __init__. seems unecessary but needs testing ##
class Named:
    """
    Named instance. Any object that has a name assigned to it is of this instance
    a=[1,2,3]
    isinstance(a,Named()) # True
    isinstance([1,2,3],Named()) # False
    """
    def __init__(self,depth: int=1) -> None: self.depth=depth
    def __instancecheck__(self,obj: Any) -> bool: return len(name(depth=self.depth)["args"][:-1]) > 0
    def __or__(self,type: type|tuple[type]) -> Union[type]: self.depth+=2; return Union[self,type]
    def __ror__(self,type: type|tuple[type]) -> Union[type]: self.depth+=2; return Union[self,type]

class BuiltinInstance:
    """
    Used for checking instances of types specific to builtin types
    
    How to use:
    
    i.e.
    BuiltinClassType: isinstance(int,BuiltinInstance(type))
    BuiltinCallableType: isinstance(int,BuiltinInstance(Callable))
            .                            .
            .                            .
            .                            .
    
    """
    def __init__(self,type: type|tuple[type]=object) -> None:
        isinstance(None,type) ## it shouldn't raise an error if it's a valid arguement for isinstance
        self.type,self.builtins=type,__builtins__.__dict__.values()
    
    def __instancecheck__(self,instance: Any) -> bool: return isinstance(instance,self.type) and instance in self.builtins
    def __subclasscheck__(self,subclass: type) -> bool: return issubclass(subclass,self.type) and subclass in self.builtins
    def __or__(self,type: type|tuple[type]) -> Union[type]: return Union[self.type,type]
    def __ror__(self,type: type|tuple[type]) -> Union[type]: return Union[self.type,type]

class readonly:
    """allows readonly attributes"""
    def __init__(self, fget) -> None: self.fget,self.__doc__=fget,fget.__doc__
    def __get__(self, obj, objtype) -> Any: return self.fget(obj)
    def __set__(self, obj, value) -> NoReturn: raise AttributeError("readonly attribute")
    def __delete__(self, obj) -> NoReturn: raise AttributeError("readonly attribute")

def iter_empty(iterable: Iterable) -> bool:
    """Checks if an iterable is empty"""
    try:
        next(copy(iterable))
        return False
    except StopIteration: return True

def iter_parts(iterable: Iterable,number_of_subsets: int) -> Iterable:
    """iterable version of partition"""
    def parts(cls: Callable) -> Generator:
        nonlocal iterable
        iterable=iter(iterable)
        while True:
            yield cls(value for _,value in zip(range(number_of_subsets),iterable)) # make sure it's range,iterable otherwise it skips values
            if iter_empty(iterable): break
    cls=type(iterable) if isinstance(iterable,list|tuple|set|dict) else (lambda x:x)
    return cls(parts(cls))

class attrdict(dict):
    """
    Allows attrs to be accessible as a dictionary
    
    Note: if a value is in either dictionary or class
    instance it will go with what's already available
    and use or modify. If this happens and you need to
    perform operations on a dictionary then go back to
    the regular methods that were intended.
    
    How to use:
    
    dct=attrdict({'a':0,'b':1,'c':3})
    dct['a']
    dct.a
    dct.a=3
    del dct.a
    dct.a=3 # setattr
    dct['a']=3
    """
    def __init__(self,*args,**kwargs) -> None: super().__init__(*args,**kwargs)
    def __getattr__(self,key: str) -> Any: return self[key] if key in self else super().__getattr__(key)

    def __setattr__(self,key: str,value: Any) -> None:
        if key in self: self[key]=value
        else: super().__setattr__(key,value)

    def __delattr__(self,key: str) -> None:
        if key in self: del self[key]
        else: super().__delattr__(key)

def pd_df_map(method: Callable) -> Callable:
    """Allows mapping str methods to pd.DataFrame.map"""
    @wraps(method)
    def wrapper(self,*args,**kwargs) -> pd.DataFrame:
        """for mapping the string methods"""
        return self.df.map(lambda x:method(x,*args,**kwargs))
    return wrapper

def StringAccessor(cls):
    not_allowed=["__init__","__new__","__repr__","__dict__","__class__","__str__","__getattribute__","__doc__"]
    for key,value in str.__dict__.items():
        if key in not_allowed: not_allowed.remove(key)
        else: setattr(cls,key,pd_df_map(value))
    return cls

@property
@StringAccessor
class str_df:
    """String accessor extension for pd.DataFrame"""
    def __init__(self,df: pd.DataFrame) -> None: self.df = df # save df

pd.DataFrame.str=str_df

def get_js(string: str,timeout: int="") -> str:
    """
    Allows communication between jupyter notebooks IPython and javascript

    Note: some amount of timeout is needed though generally it suffices to leave it as is as it seems to work that way
    
    # code reference: erpuntbakker (2019) https://stackoverflow.com/questions/58881349/cannot-get-jupyter-notebook-to-access-javascript-variables?rq=3, CC BY-SA 4.0
    # changes made: condensed into a function for reuse, allowed timeout to be optional, removed unnecessary code
    """
    if timeout: timeout=f",{timeout*10**3}"
    display(Javascript("""
      const CodeCell = window.IPython.CodeCell;
      CodeCell.prototype.native_handle_input_request = CodeCell.prototype.native_handle_input_request || CodeCell.prototype._handle_input_request
      CodeCell.prototype._handle_input_request = function(msg) {
          setTimeout(() => { IPython.notebook.kernel.send_input_reply("""+string+""") """+timeout+"""})
      }
    """))
    return input({})

@lambda x: x()
class cut:
    """
    Allows setting multiple values as copies rather than pointers 
    effectively cutting each reference from the original id if the
    value being set to is mutable

    How to use:

    cut.a.b.c=[] # will set a,b,c each a copy of []

    if you go:
    
    cut.a.b.c

    it will record the variables and therefore you can set 
    cut by choosing one of the variables:
    cut.a=[] # will also set b,c since getattr has been used on them

    to prevent this if desired you can reset the variables
    recorded by using a negation on cut before using it:

    -cut
    cut.a=[] # only gives a copy to 'a'
    or 
    (-cut).a=[]

    if you need to see what variables are recorded you can use
    cut._cut__temp or ~cut to display them

    Also, if you want, you can use lambda expressions to set
    the name of the input variable as the result of the expression:
    i.e.
    cut(lambda x: x+1)(lambda y: y+2).c=3 # x: 4,y: 5, c: 3 
    """
    __temp=set()
    def __call__(self,FUNC) -> object:
        self.__temp|={FUNC}
        return self
    
    def __setattr__(self,key,value) -> None:
        if key=="_cut__temp": return super().__setattr__(key,value)
        current=scope(1).locals
        for key in self.__temp|{key}:
            if isinstance(key,Callable): current[list(signature(key).parameters)[0]]=key(copy(value))
            else: current[key]=copy(value)
        self.__temp=set()
    
    def __getattr__(self,key) -> object:
        self.__temp|={key}
        return self

    def __invert__(self) -> set: return self.__temp
    
    def __neg__(self) -> object:
        self.__temp=set()
        return self

def comp(*FUNCS: tuple[Callable,...]) -> Callable:
    """
    wraps multiple functions together around one object
    
    How to use:

    comp(classmethod,property)(*some value or object*)
    
    You should be able to use it as a decorator as well
    
    @comp(classmethod,property)
    def test():
        return 3
    """
    return reduce(lambda outer, inner: lambda obj: outer(inner(obj)), FUNCS)

@contextmanager
def decorate(*FUNC: Callable) -> None:
    """
    Used to decorate multiple function definitions at once
    
    How to use:

    with decorate(property):
        def f1(): pass
        def f2(): pass
        def f3(): pass
        ...
    # should make all functions have property decorators
    """
    current=scope(2).locals.copy()
    yield
    new=scope(2).locals.copy()
    for key,value in new.items():
        if key not in current: scope(2)[key]=comp(*FUNC)(value)

def import_module(path: str,asname: str=None,attrs: str|Iterable[str]=None) -> ModuleType|Iterable:
    """
    Allows dynamic importing of a module or its attributes from a file
    
    How to use:
    i.e.
    import_module(c:\\Users\\Me\\A\\B\\C\\test.py) ## will import module test as name (reference/pointer) 'test'
    import_module(c:\\Users\\Me\\A\\B\\C\\test.py,asname="module") ## will import module test as name 'module'
    import_module(c:\\Users\\Me\\A\\B\\C\\test.py,attrs=["a","b","c"]) ## will import 'a','b','c' from module 'test'
    """
    path=section_path(path)
    with Path(path.dir): ## context manager needed in case of imports in the paths directory
        module_name=asname if asname else path.name
        spec=importlib.util.spec_from_file_location(module_name, path.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if attrs:
            attrs=as_list(attrs)
            return getattr(module,attrs[0]) if len(attrs)==1 else (getattr(module,attr) for attr in as_list(attrs))
        return module

def position(obj: Any) -> tuple[int,...]:
    """
    extracts the position of an object

    Assuming your object (typically) represents i.e. a location in the source code
    and has attrs 'lineno', 'end_lineno','col_offset', 'end_col_offset' then this
    function is used for the purpose of extracting the positions in this order
    
    How to use:

    i.e. if got an ast node or instruction object
    position(node) # will return its position in the source code
    """
    return tuple(getattr(obj,attr) for attr in ('lineno', 'end_lineno','col_offset', 'end_col_offset'))

## needs testing ##
def section_source(pos: tuple[int,...],source: str) -> str:
    """
    gets the section of source code from a string assuming you have a position of the form:
    ('lineno', 'end_lineno','col_offset', 'end_col_offset') # where each entry is an int type

    This will allow you to extract the particular source code of interest in accordance with its 
    line/col position

    How to use:

    section_source((0,0,5,5),"True;1+1;not 3") # should return 1+1;
    """
    line="\n".join(source.split("\n")[pos[0]-1:pos[1]])
    return line#[pos[2]:-pos[3]] ## need to check this

## needs testing ##
def shallow_trace(obj: Named|str,show_source: bool=False,depth: int=1,source: str="") -> list[str]:
    """
    Does a general search for the last known import, assignment or 
    definition of an object from an ast of its source code.
    
    Generally, this should suffice for most use cases, however, 
    if objects are being created from isolated frames or other 
    'hidden' objects (anything that needs to be directly checked
    for to see if it modifies variables at any frames from its
    scope) then these need to be checked as well to give a more
    in depth trace but will also take longer to process. - (will develop later)

    How to use:
    s='''
    from test import t

    t=1
    t=2
    t=3
    '''
    shallow_trace(t,True,source=s) # should return ['from test import t','t=1','t=2','t=3']
    shallow_trace(t,source=s) # should return a list of ast nodes i.e. [<ast.ImportFrom>,<ast.Assign>,...]
    """
    obj_name=obj if isinstance(obj,str) else name(depth=depth)["args"][0]
    trace,source=[],source if source else history(True)
    ## traverse the code recording imports, assignments, and definitions
    for node in ast.parse(source).body:
        if isinstance(node,ast.Import):
            for obj in node.names:
                if obj.name==obj_name: trace+=[node]
                elif hasattr(obj,"asname") and obj.asname==obj_name: trace+=[node]
        elif isinstance(node,ast.ImportFrom):
            for obj in node.names:
                if obj.name==obj_name: trace+=[node]
                elif hasattr(obj,"asname") and obj.asname==obj_name: trace+=[node]
        elif isinstance(node,ast.FunctionDef|ast.ClassDef):
            if node.name==obj_name: trace+=[node]
        elif isinstance(node,ast.Assign):
            if isinstance((targets:=node.targets[0]),ast.Tuple):
                for target in targets.elts:
                    if target.id==obj_name: trace+=[node]
            elif node.targets[0].id==obj_name: trace+=[node]
        elif isinstance(node,ast.Expr):
            pass ## needs implementing
    return [section_source(position(node),source) for node in trace] if show_source else trace

def patch_exception(FUNC: Callable) -> None:
    """
    Monkey patches the exception handler in python to allow
    your own custom exception trace back functionality
    
    Note: your function typically should be formatted the following way:
    
    from typing import NoReturn
    from types import TracebackType
    
    def my_exception(exception_type: type,exception: BaseException, traceback: TracebackType) -> NoReturn:
        ...
    
    This is assuming you want to override the traceback display and not just 
    arbitarily monkey patch the method to a value; though you can utilize the
    arguements to do other things globally by using the global key word to i.e. 
    log to a file or pass into another process.
    
    Example of How to use:
    
    import traceback

    def my_exception(exception_type: type,exception: BaseException, trace_back: TracebackType) -> NoReturn:
        print(traceback.format_exception(exception_type,exception,trace_back))
        
    patch_exception(my_exception) # can be patched again if desired

    1/0 # won't raise the usual error but will instead utilize your function

    Try not to use this function unless you have a preference for traceback formatting
    or other logging behavior as the program will still raise an error but what it
    displays will be modified
    """
    if has_IPython():
        def showtraceback(*args,**kwargs) -> None: FUNC(*sys.exc_info())
        IPython.core.interactiveshell.InteractiveShell.showtraceback=showtraceback
    else: sys.excepthook=FUNC

@contextmanager
def Path(path: str) -> None:
    """
    Context manager to temporarily move into different directories and then go back to the original directory

    How to use:

    with Path(*some directory*):
        *do something*
        os.getcwd() # should be equal to the *some directory* provided to Path as an arguement
    os.getcwd() # should be equal to the previous directory that you were in prior to the context manager being used
    """
    if path:
        try:
            current=os.getcwd()
            if not os.path.isdir(path): path=os.path.dirname(path)
            os.chdir(path)
            yield    
            os.chdir(current)
        except Exception as e:
            os.chdir(current)
            raise e
    else: yield

def module_file(module: str,relative: str|Iterable[str]="",extensions: Iterable[str]=[".py",".pyc",".so",".pyd"],show_type: bool=False) -> str|tuple[str,bool]:
    """
    Gets the full file path to a module without executing it
    
    if wanting relative imports you must supply relative with a directory to be relative to

    How to use:

    module_file('unittest') # should return something like: 'C:\\Users\\Me\\...\\Lib\\unittest\\__init__.py'
    module_file('unittest',show_type=True) # should return something like: ('C:\\Users\\Me\\...\\Lib\\unittest\\__init__.py','absolute')
    module_file('builtins') # should throw a FileNotFound exception/error (even if you have a library called the builtins (unless you intend to do a relative or dynamic import))
    module_file('builtins','C:\\Users\\Me\\a\\b\\c') # assuming C:\\Users\\Me\\a\\b\\c\\builtins.py exists it should return it else it will throw an exception
    """
    paths=as_list(relative) if relative else as_list(sys.path) # sys.path contains the directories python uses to look for modules
    modules=module.split(".")
    submodules=len(modules)
    for module in modules:
        module="\\"+module
        for path in paths:
            if path=="": path=os.getcwd()
            ## modules take preceedence over packages
            for ext in extensions:
                if os.path.isfile((location:=path+module+ext)) and submodules==1:
                    if show_type: return location,"relative" if path not in sys.path else "absolute"
                    return location
            if os.path.isdir((location:=path+module)):
                location+="\\__init__.py" # if it's a package it will always have an __init__.py file
                if os.path.isfile(location):
                    if submodules > 1:
                        submodules-=1
                        paths=[os.path.dirname(location)]
                        break
                    if show_type: return location,"relative" if path not in sys.path else "absolute"
                    return location
    raise FileNotFoundError("module is not on path or does not exist. If module is a relative import try giving a directory to reference from")

def dir_back(depth: int=0,location: str="") -> str:
    """
    Gets the specified parent directory path

    How to use:

    if os.getcwd() is: C:\\Users\\Me\\a\\b\\c
    dir_back() will return: C:\\Users\\Me\\a\\b\\c
    dir_back(1) will return: C:\\Users\\Me\\a\\b
    dir_back(2) will return: C:\\Users\\Me\\a
    ...
    """
    current=location if location else os.getcwd()
    for i in range(depth): current=os.path.dirname(current)
    return current

## might remove ##
def as_dir(location: str) -> str: return location if os.path.isdir(location) else os.path.dirname(location)

def source(module: str,location: str="") -> tuple[str,str,bool]:
    """
    Gets the source code for a module
    
    Designed to integrate with the get_imports function but can be used to get the source code
    of a module e.g. source(*module*,*its location (if doing relative imports)*)[0]
    """
    # is it a relative import or a site-package
    try:
        with Path(location): file,relative=module_file(module,extensions=[".py"],show_type=True)
        return open(file,encoding="utf-8").read(),file,relative=="relative"
    except FileNotFoundError: return (None,)*3

def foundations(code: str) -> tuple[dict,dict]:
    """
    records all imported objects and their modules from the given code
    
    For every module it will go recursively however many modules deep to
    form a foundation of what code your program used to execute on

    How to use:

    foundations("from test import t")

    # should return a tuple of two dictionaries representing absolute 
    # and relative imports respectively with the following formatting

    absolute imports are recorded as:

    {module1:{attr1,attr2,...},
     module2:{attr1,attr2,...},
     ...}
    
    whereas relative imports have their locations noted then the attributes imported
    
    {module1:{location1:{attr1,attr2,...},location2:{attr1,attr2,...},...},
     module2:{location1:{attr1,attr2,...},location2:{attr1,attr2,...},...},
     ...}
     
    relative imports are of the form from . import y or from .. import y, from ... import y,... etc. or are in the
    same directory or below as part of a package/project directory
    
    relative imports with the dot notation basically shifts its current directory back one and then does the imports 
    so it's essentially running: import ... but from the parent directory above. The number of dots in the from statement 
    is how many levels (directories) to shift up relative to the current path
    """
    locations_searched,current_imports,relative_imports=set(),{},{}
    def import_name(node: ast.ImportFrom|str,location: str="",import_from: str=None) -> None:
        """
        used when node is: from x import y or import z
        
        When importing modules we either add the module as is (import)
        or add attributes to it (import from)
        """
        nonlocal current_imports,relative_imports
        module=import_from if import_from else node
        ## get the next round of modules
        module_source,new_location,relative=source(module,location)
        if module_source:
            if relative: # add to the relative imports
                if module not in relative_imports: relative_imports[module]={new_location:set()}
                elif new_location not in relative_imports[module]: relative_imports[module][new_location]=set()
                if import_from: relative_imports[module][new_location]|={attr.name for attr in node.names}
            else:
                if module not in current_imports: current_imports[module]=set()
                if import_from: current_imports[module]|={attr.name for attr in node.names}
                current_imports=get_imports(module_source,new_location)
    
    def import_names(node: ast.alias,location: str) -> None:
        """handles nodes of the form: import ..."""
        for module in as_list(node): import_name(module.names[0].name,location)
    
    def get_imports(code: str,location: str="") -> dict:
        """
        Parses the code into an ast (abstract syntax tree) that is traversed 
        to uncover what attributes are being imported and where from
        """
        nonlocal locations_searched,current_imports
        ## prevents duplicate recursions
        if location in locations_searched: return current_imports
        else: locations_searched|={location}
        ## use ast.walk to get full coverage of the source
        for node in ast.walk(ast.parse(code)):
            if isinstance(node,ast.Import): import_names(node,location)
            elif isinstance(node,ast.ImportFrom):
                # adjust the location if necessary for singular use
                temp_location=dir_back(node.level,location)
                import_name(node,temp_location,node.module) if node.module else import_names(node,temp_location)
        return current_imports
    
    return get_imports(code),relative_imports

def as_list(obj: Any) -> list:
    """
    Makes sure an object is contained in a list

    How to use:

    as_list([1,2,3]) # should return [1,2,3]
    as_list(1) # should return [1]
    """
    return obj if isinstance(obj,list) else [obj]

def extract_callables(True_name: str,True_module: str) -> tuple[str,list[dict]]:
    """
    Gets all callables from a string

    Typically you should use this to extract all callables from a source file but
    you can also use this to extract any python valid function/class definitions
    from a string
    
    How to use:
    # assuming this is the module 'test'
    s='''
    a=1
    def hi():pass
    b=2
    class test: pass
    c=3
    def hi_there():pass
    d=4
    hi="hello"
    e=5
    class hi:pass
    '''
    extract_callables("hi","test") # should return 3 sectioned ast nodes representing the function definition,assignment, and class definition
    """
    source=history(True) if True_module=="__main__" else open(module_file(True_module),encoding="utf-8").read()
    ## TODO: will need to also add in the parameters e.g. to allow knowledge on how many there are etc.
    return source,[section_ast(obj) for obj in ast.parse(source).body if isinstance(obj,ast.FunctionDef|ast.ClassDef) and obj.name==True_name]

def section_ast(obj: ast.FunctionDef|ast.ClassDef) -> dict:
    """
    Allows source code to be broken into parts according with the ast object
    
    How to use:

    Assuming an ast.FunctionDef|ast.ClassDef is passed in:

    it will return a dictionary of:
    |   key    |       value       |
    ---------- - -------------------
    decorators | a dictionary of decorator names mapping to their source code position
    docstring  | source code position of the docstring
    full       | source code position of the full function
    signature  | source code position of the function signature
    body       | source code position of the body
    """
    record={"decorators":{decorator.id:(decorator.lineno-1,decorator.end_lineno) for decorator in obj.decorator_list[::-1]},
            "docstring":ast.get_docstring(obj),"full":[obj.lineno-1,obj.end_lineno,obj.col_offset,obj.end_col_offset]}
    code_start=obj.body[bool(record["docstring"] and len(obj.body) > 1)]
    record["body"]=(code_start.lineno-1,obj.body[-1].end_lineno,code_start.col_offset,None)
    start=obj.body[0]
    if code_start.lineno==start.lineno: record["signature"]=(start.lineno-1,start.end_lineno,0,start.col_offset)
    else:
        record["signature"]=[obj.lineno-1,None,obj.col_offset,None]
        if record["docstring"]: record["signature"][1]=-len(record["docstring"])
    if record["decorators"]: record["full"][0]=list(record["decorators"].values())[-1][0]
    return record
    
def source_code(True_name: Callable|str,True_module: str="__main__",join: bool=True,check_cache: bool=False,key: str="original",depth: int=1) -> tuple[str,str,str,str]|str:
    """
    Gets the source code of a function manually, including for decorated functions/classes.

    Note: only works if the True_name is the actual name of the function/class and 
    True_module is the actual module name otherwise these presumably have been changed
    and will mess with the results.

    The main advantage of using this function over the regular getsource from inspect 
    is allowing you to break up the source code into its distinct parts and retrieval
    of source code for functions that would have otherwise not been retrieved due to
    how getsource works. 
    
    For example, getsource won't work on class decorated functions.
    
    This is likely because class decorated functions as opposed to function decorators 
    don't have a __closure__ attribute and thus there is no way of unwraping the class
    decorator to know what the original function was and where its source code is. My
    function will trace back to the source manually by traversing an abstract syntax tree
    of the source code allowing retrieval.
    
    set join=False to break up the source code

    key="original","new", or other custom specified key available for the undecorate function

    How to use:

    i.e. a class decorated function
    # test2.py
    class a:pass
    class b(a):pass
    class c(a):pass
    # test.py
    from test2 import a,b,c
    @a
    @b
    @c
    def t():
        '''hi'''
        return 3
    # __main__
    from test import t

    source_code(t) # or source_code('t','test')
    # should return:
    @a
    @b
    @c
    def t():
        '''hi'''
        return 3
    source_code(t,join=False) # or source_code('t','test',join=False)
    # should return:
    ['@a\n@b\n@c',def t():,'hi','    return 3']
    """
    if not isinstance(True_name,str):
        if not (hasattr(True_name,"__name__") and hasattr(True_name,"__module__")):
            obj_name=name(depth=depth)["args"][0]
            if not hasattr(True_name,"__module__"):
                if "." in obj_name:
                    obj_name=obj_name.split(".")
                    True_module,True_name=".".join(obj_name[:-1]),obj_name[-1]
                else: ## trace back where the object originates from
                    trace=shallow_trace(obj_name)
                    for nodes in trace:
                        if isinstance(nodes,ast.ImportFrom):
                            for node in nodes.names:
                                if node.name==obj_name or (hasattr(node,"asname") and node.asname==obj_name):
                                    True_name,True_module=node.name,nodes.module
                                    break
                            else: continue
                            break
                    else: raise Exception("object couldn't be traced")
            else: True_name,True_module=obj_name,True_name.__module__
        else: True_name,True_module=True_name.__name__,True_name.__module__
    if check_cache:
        global SOURCE_CODES
        try: source=SOURCE_CODES[True_name]
        except: raise Exception("source code not found")
        try: source=source[key]
        except: raise Exception(f"source code not found at key '{key}' but the original source code may exist i.e. try 'original' as key value")
    ## need to do something about the history function in case of exceptions because they get recorded otherwise it should work fine
    source,record=extract_callables(True_name,True_module)
    if record:
        record=record[-1]
        lines=source.split("\n")
        if join:
            pos=record["full"]
            lines=lines[slice(*pos[:2])]
            lines[-1]=lines[-1][slice(*pos[2:])]
            lines=[line[pos[2]:] for line in lines[:-1]]+[lines[-1]]
            return "\n".join(lines)
        pos=record["body"]
        body="\n".join(lines[slice(*pos[:2])])[slice(*pos[3:])]
        pos=record["signature"]
        signature=lines[slice(*pos[:2])]
        signature[0]=signature[0][pos[2]:]
        signature[-1]=signature[-1][:pos[3]]
        pos=list(record["decorators"].values())
        decorators="\n".join(lines[slice(pos[-1][0],pos[0][-1])]) if pos else ""
        docstring=record["docstring"]
        if docstring==None: docstring=""
        return decorators,signature[0],docstring,body
    raise Exception(f"Source code for callable '{True_name}' not found in module '{True_module}'")

def history(join: bool=False) -> list[str]:
    """
    Gets the entire code execution history for the main program

    Note: if on a CLI you might want to clear the history before use
    otherwise it will include history from other sessions.
    
    e.g.
    import readline
    readline.clear_history()

    How to use:

    # __main__
    1+1
    2
    range(5)
    class a: pass
    a()
    history(True) # will return '1+1\n2\nrange(5)\nclass a: pass\na()'
    history() # will return the results in a list e.g. in their execution order
    """
    join_up=lambda string,code: string.join(code) if join else code
    if has_IPython():
        section=copy(scope()["In"])
        line_number=scope().global_frame.f_lineno
        section[-1]="\n".join(section[-1].split("\n")[:line_number])
        return join_up("\n",section)
    file_name=scope().scope.get("__file__",None)
    if file_name:
        line_number=scope().global_frame.f_lineno
        return join_up("",open(file_name).readlines()[:line_number])
    return join_up("\n",[readline.get_history_item(i+1) for i in range(readline.get_current_history_length())])

def bracket_removal(code: str) -> str:
    """
    Removes brackets and their enclosing code from a string 
    (assumes strings have been removed i.e. by using extract(code))
    """
    ## get bracket locations for the uppermost encapsulation level
    brackets=bracket_up(code)
    brackets=brackets[brackets["encapsulation_number"]==0]
    starts,ends=iter(brackets["start"].tolist()),iter(brackets["end"].tolist())
    ## remove the brackets
    new_string,start,end,removing="",next(starts),next(ends),False
    for index,char in enumerate(code):
        if index==start: start,removing=next(starts),True
        elif index==end: end,removing=next(ends),False
        if not removing: new_string+=char
    return new_string

def notebook_url() -> str:
    """
    Retrieves the notebooks url

    Note: if using jupyter lab apparently this works:
    # reference: https://github.com/jtpio/ipylab/discussions/139
    import os
    os.environ.get('JPY_SESSION_NAME')
    """
    return get_js("window.location.href")

def IPython__file__() -> str:
    """Gets the full file path if using jupyter notebook"""
    return os.path.join(os.getcwd(),urllib.parse.unquote(urllib.parse.urlparse(notebook_url()).path.split("/")[-1]))

def unwrap(FUNC: Callable,depth: int=1,trace: bool=False) -> tuple[Callable,...]:
    """Extracts the function and all its wrapper functions in execution order"""
    if isclass(FUNC) or trace: # trace - in case decorator functions return functions e.g. as a factory like the class decorators
        func_name=name(depth=depth)["args"]
        return tuple(source_code(FUNC,join=False,depth=depth+1)[0][1:].split("\n@")+func_name)
    functions=(FUNC,)
    while True:
        try:
            FUNC=FUNC.__closure__[0].cell_contents
            functions+=(FUNC,)
        except: break
    return functions

def copy(*args) -> Any|tuple[Any]:
    """general purpose function for copying python objects"""
    new_args=tuple()
    for arg in args:    
        if hasattr(arg,"copy"): new_args+=(arg.copy(),)
        elif isinstance(arg,FunctionType): new_args+=(func_copy(arg),)
        elif isclass(arg): new_args+=(class_copy(arg),)
        elif isinstance(arg,ModuleType): new_args+=(module_copy(arg),)
        else:
            try: new_args+=(deepcopy(arg),)
            except:
                extend="" if mutable(arg) else " but is likely an immutable type"
                warn(f"\n\nwarning: arguement '{arg}' was not copied{extend}\n",stacklevel=2)
                new_args+=(arg,)
    return new_args[0] if len(new_args)==1 else new_args

def func_copy(FUNC: Callable) -> Callable:
    """for copying a function"""
    return FunctionType(FUNC.__code__,locals())

def module_copy(module: ModuleType) -> ModuleType:
    """Creates a copy of a module"""
    module_copied = ModuleType(module.__name__, module.__doc__)
    module_copied.__dict__.update(module.__dict__)
    return module_copied

## needs testing ##
def ispatched(obj: Callable|ModuleType,attr: str) -> bool:
    """Checks if an object is monkey patched or if the value has changed since initialization"""
    if isinstance(obj,ModuleType): initial_obj=new_module(obj.__name__)
    else:
        try: initial_obj=getattr(new_module(obj.__module__),obj.__name__)
        except: return True
        ## not robust for functions
        if obj!=initial_obj: return True
    return getattr(initial_obj,attr)!=getattr(obj,attr) if hasattr(initial_obj,attr) else hasattr(obj,attr)

def new_module(module: str,name: str="") -> ModuleType:
    """creates a new version of a module"""
    spec=importlib.util.find_spec(module)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def mutable(obj: Any) -> bool:
    """
    determines if an object is mutable.

    Note: this function will always work assuming the creation of it's state (it's __new__
    method) does not vary unpredictably e.g. a general test for mutability is not possible
    because it's unknowable as to all the different ways in which a state is formed if there's 
    unpredicatability. So something like the following or using a random number generator will 
    always be part of a family of edge cases:
    
    count=0
    class EdgeCase(object):
        def __new__(cls, *args, **kwargs):
            global count
            count+=1
            if count % 2:
                return [1,2,3]
            else:
                return tuple()
    
    print(mutable(EdgeCase())) # True
    print(mutable(EdgeCase())) # False
                .                .
                .                .
                .                .

    Generally these don't occur but are possible. Hence why mutability tests
    are not possible unless somehow the source code were taken into consideration
    or a docstring mentions this kind of behavior.
    """
    cls=obj.__class__
    if cls==type: ## types can be mutable if you can set attrs to them and the memory location stays the same
        previous_id=id(obj)
        if hasattr(obj,"__slots__"):
            try:
                slot=obj.__slots__[0]
                temp=getattr(obj,slot,3)
                setattr(obj,slot,3)
                setattr(obj,slot,temp)
                if previous_id==id(obj): return True
            except: pass
            return False
        try:
            obj.__test=3
            del obj.__test
            if previous_id==id(obj): return True
        except: pass
        return False
    return cls.__new__(cls) is not cls.__new__(cls)

OVERLOADS={}
def overload(func: Callable) -> Callable:
    """
    overloader for overloading of simple functions i.e. no 
    self,*,a,b or self,/,a,b support has been implemented
    
    Also, this function probably doesn't work for *args,**kwargs
    potentially
    
    The overloader overloads based on two criteria:
    1. number of args given
    2. type of args given
    
    Note: the second criteria at the moment is very strict e.g. 
    you couldn't implement an Any type and have it work though
    generally many things are of an Any type. Will implement 
    something for it later to allow wider support and flexibility. 
    However, if you're not going to type annotate the function 
    definitions inputs then only the first criteria needs to be 
    met but it will use the earliest definition.
    
    How to use:
    
    @overload
    def t() -> None: # all functions must be overloaded otherwise they get overwritten
        return 1
    @overload
    def t(a: int):
        return 2
    @overload
    def t(*args: int,**kwargs: int):
        return 3
    """
    global OVERLOADS
    ## store
    args,annotations=get_arg_count(func)[-1],func.__annotations__
    if "return" in annotations:
        del annotations["return"]
    annotations=tuple(annotations.values())
    if func.__name__ in OVERLOADS: ## replace
        for dct in OVERLOADS[func.__name__]:
            if (dct["args"],dct["annotations"])==(args,annotations): dct["func"]=func
        else: OVERLOADS[func.__name__]+=(dict(args=args,annotations=annotations,func=func),)
    else: OVERLOADS[func.__name__]=(dict(args=args,annotations=annotations,func=func),)
    ## pass back function that first does a lookup
    @wraps(func)
    def wrapper(*Args,**kwargs) -> Callable:
        """Since all the functions are the same name we have to look them up before executing"""
        annotations=tuple()
        args=(*Args,*kwargs.values())
        if args:
            for args,item in enumerate(args): annotations+=(item.__class__,)
            args+=1 # since it was used as an indexer
        else: args,annotations=0,tuple()
        for dct in OVERLOADS[func.__name__]:
            if dct["args"]==args:
                temp=dct["annotations"]
                if temp:
                    if temp==annotations: return dct["func"](*Args,**kwargs)
                else: return dct["func"](*Args,**kwargs)
        else: raise TypeError(f"No functions with {args} args and annotations: {annotations}")
    return wrapper

## needs more testing ##
class mute:
    """
    Turns mutable objects immutable or immutable objects to mutable

    How to use:

    a=mute(type) # is mutable
    a(int)
    a.a=3
    mute(a) # is immutable
    #a.a=3  ## should raise a TypeError
    """
    _mute__immute_flag=0
    def __init__(self,obj: Any,immute: bool=False) -> None:
        self.__obj,self.__immute_flag=(obj,(obj.__immute_flag+1) % 2) if obj.__class__.__name__=="mute" else (obj,immute)
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute on the wrapped object or the mute object itself."""
        if self._mute__immute_flag: raise TypeError(f"cannot set attribute '{key}' to an immutable type. To create a mutable object use mute(obj)")
        super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        try: return super().__getattr__(key)
        except AttributeError: return getattr(self.__obj, key)
    
    def __repr__(self) -> str: return repr(self.__obj)
    def __call__(self, *args: Any, **kwargs: Any) -> Any: return self.__obj(*args, **kwargs)

## needs testing ## - might re-write the code to allow starting with multiple .pkl files so that the processes are not locking up on one file if it's causing slow downs
def multi_process(number_of_processes: int,interval_length: int,FUNC: Callable,combine: str|None=None,wait: bool=True) -> dict|Any|list[subprocess.Popen]:
    """
    multi processor for python code via strings and subprocesses.

    This multiprocessor works by partitioning the interval of a for loop for 
    each process to work on separately and serializing the code, partitions used,
    and scope used for a function to be transferable over to a separate
    and independent process. 
    
    This process then deserializes or reads in
    the objects from the pickle file (.pkl) and executes it on the partition
    desired saving the result in a temporary pickle file that gets retrieved
    by the main program and combined if desired into a full result. 
    
    On satisfactory retrieval of the results the .pkl files are then deleted
    where the initial .pkl file is deleted last.

    How to use:

    from my_pack import multi_process

    def test(parts):
        return [1,2,3,4][parts[0]:parts[1]]
    
    multi_process(4,4,test) # should return {'result_0': [1], 'result_1': [2], 'result_2': [3], 'result_3': [4]}
    
    multi_process(4,4,test,"+") # should return [1,2,3,4]
    """
    # serialize what is required e.g. the function code,partitions, and the scope used
    directory=os.getcwd()
    get_name=lambda :tempfile.NamedTemporaryFile(dir=directory,suffix=".pkl").name # back slashes get reduced
    obj_store=get_name()
    to_pickle({"FUNC":FUNC.__code__,"part":tuple((part[0],part[1]) for part in 
               partition(number_of_processes,interval_length)),"scope":scope(1).scope.items()},obj_store,force=True)
    # loading the python object
    ## read in the code object to set the scope, function, and which partition it's set to
    remove_backslash=lambda string: string.replace("\\","\\\\")
    process=lambda index,store_name: f"""import dill
from types import FunctionType

with open('{remove_backslash(obj_store)}', 'rb') as file:
    code=dill.load(file)

for key,value in code["scope"]: globals()[key]=value
FUNC=FunctionType(code["FUNC"], globals(), "temp_process")

with open('{remove_backslash(store_name)}','wb') as file:
    dill.dump(
                FUNC(code["part"][{index}]),
                file)
"""
    store_names=[get_name() for i in range(number_of_processes)]
    def retrieve() -> dict:
        """Used to wait for the processes to finish to retrieve and then combine the results after if desired"""
        nonlocal store_names,process
        processes=[Process(process(index,store_name)) for index,store_name in enumerate(store_names)]
        results={}
        for count,(process,file_name) in enumerate(zip(processes,store_names)):
            while process.poll()==None: pass
            if process.poll()!=0: print(process.communicate())
            else:
                results[f"result_{count}"]=read_pickle(file_name)
                os.remove(file_name)
        return results

    if wait or combine:
        result=retrieve()
        os.remove(obj_store)
        if combine: return biop(result.values(),combine)
        return result
    # start the processes
    return [Process(process(index,store_name)) for index,store_name in enumerate(store_names)],store_names

def Process(code: str,save: str="") -> subprocess.Popen:
    """
    Runs python code from a string on a separate process
    
    Note: indentation is two spaces for the code in your string
    else use multi-line strings for ease of use
    
    How to use:
    # create code
    code='''
    import time
    while True:
        print('a')
        time.sleep(3)
        print('b')
        break
    '''
    # run in a new process
    process = Process(code)
    
    Things you can do with it
    ## you can then check on it
    process.info
    
    ## you can wait for it
    while process.poll()==None: pass ## .poll returns the status of whether it's currently running or not
    
    ## show the stdout and stderr as a tuple
    print(process.communicate())
    
    Will add more code later that will pickle the results of processes
    to files that can be retrieved and then combined as one result
    
    After that, multiprocessing should be easily possible so long as code
    that the processes will be using can be retrieved and that the data
    structures wanting to be saved can be pickled.
    """
    ## sys.executable is the directory of python.exe; '-c' allows running your code; subprocess.PIPE is used to communicate between processes over a pipe
    return subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
@property
def process_info(process: subprocess.Popen) -> None:
    """Shows common info regarding a subprocess"""
    pid=process.pid
    print("Process id:",pid)
    try:
        process=psutil.Process(pid)
        print("Status:", process.status(),
              "\nCPU usage:", process.cpu_percent(),
              "\nMemory usage:", process.memory_info(), # might format this better
              "\nCreated:",process.create_time())
    except psutil.NoSuchProcess: print("Process terminated")

if hasattr(subprocess.Popen,"info")==False: subprocess.Popen.info=process_info
else: warn("An attempt to monkey patch 'info' to subprocess.Popen failed")

def load(modules: dict,instantiate: list=[]) -> None:
    """
    Allows dynamic importing of many libraries with or without 'as' usage and instantiation
    Use 'instantiate' to instantiate any classes with the call passed i.e.
    
    load({"a":"hi as k,hello,A as b"},["b(3,6)"]) # wil instantiate b=b(3,6)
    # by default you don't need to pass a call if no args/kwargs ar to be passed i.e.
    load({"a":"hi as k,hello,A as b"},["b"]) # wil instantiate b=b()
    """
    for module,attrs in modules.items(): exec("from "+module+" import "+attrs if attrs else "import "+module,globals())
    if type(instantiate)!=list: raise TypeError("'instantiate' must be of type 'list'")
    for attr in instantiate:
        new_attr,call=slice_occ(attr,"("),"()"
        if type(new_attr)!=str: attr,call=new_attr
        exec(attr+"="+attr+call,globals())

def reload(module: str) -> None:
    """For reloading imports i.e. if you make changes to your library code"""
    # reload, and get imports reloaded if there was any
    importlib.reload(__import__(module))
    # globals() has to be a tuple else it's a pointer to globals which can change during iteration
    imports=set(i.__name__ for i in tuple(globals().values()) if hasattr(i,"__module__") if i.__module__==module)
    if imports:
        imports=",".join(imports)
        exec("from "+module+" import "+imports,globals())
        return print(imports+" from "+module,"reloaded")
    exec("import "+module,globals())
    print(module,"reloaded")

class nonlocals:
    """
    Equivalent of nonlocals()
    
    # code reference: jsbueno (2023) https://stackoverflow.com/questions/8968407/where-is-nonlocals
    # changes made: condensed the core concept of using a stackframe with getting the keys from the 
    # locals dict since every nonlocal should be local as well and made a class
    """
    def __init__(self,frame: FrameType|None=None) -> None:
        self.frame=frame if frame else currentframe().f_back
        self.locals=self.frame.f_locals
    
    def __repr__(self) -> str: return repr(self.nonlocals)
    @property
    def nonlocals(self) -> dict:
        names=self.frame.f_code.co_freevars
        return dct_ext(self.locals)[names] if len(names) else {}

    def check(self,key: Any) -> None:
        if key not in self.nonlocals: raise KeyError(key)

    def __getitem__(self,key: Any) -> Any: return self.nonlocals[key]
    def update(self,dct) -> None:
        for key,value in dct.items(): self[key]=value
    def get(self,key,default=None) -> Any: return self.nonlocals.get(key,default=default)
    def __setitem__(self,key: Any,value: Any) -> None:
        self.check(key)
        self.locals[key]=value
        # code reference: MariusSiuram (2020). https://stackoverflow.com/questions/34650744/modify-existing-variable-in-locals-or-frame-f-locals,CC BY-SA 4.0
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(0))

    def __delitem__(self,key: Any) -> None:
        self.check(key)
        del self.locals[key]
        # code reference: https://stackoverflow.com/questions/76995970/explicitly-delete-variables-within-a-function-if-the-function-raised-an-error,CC BY-SA 4.0
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(1))

def proxy(dunder_method: Callable) -> Callable:
    """Creates a proxy wrapper for the method"""
    @wraps(dunder_method)
    def wrapper(self,*args,**kwargs) -> Any: return getattr(self.func(),dunder_method)(*args,**kwargs)
    return wrapper

def use_all_dunders(cls: type) -> type:
    """
    adds all relevant instance based dunder methods to a class without 
    overriding important methods such as __getattr__,__class__, etc.
    """
    __dunders__=get_dunders()
    not_allowed=["__class__","__getattribute__","__getattr__","__dir__","__set_name__","__init_subclass__","__mro_entries__",
            "__prepare__","__sizeof__","__fspath__","__init__","__new__","__setattr__","__delattr__","__get__","__set__",
            "__delete__","__dict__","__doc__","__name__","__qualname__","__module__","__abstractmethods__"] ## need to check this
    for dunder in __dunders__["call"]:
        if dunder in not_allowed: not_allowed.remove(dunder);continue
        setattr(cls,dunder,proxy(dunder))
    return cls

@use_all_dunders
class staticproperty:
    """
    Allows a function to be called as a variable and will
    synchronize with changes to it's internals from i.e. 
    global variables
    
    Note: staticproperties are generally not advised because
    you cannot pass in arguements to the function or access 
    it's attributes which makes the use cases limited to the 
    use of global or nonlocal variables generally.

    Also, if wanting a fixed staticproperty e.g. one that is
    going to return the same output every call then you can
    use a lambda decorator like so:

    @lambda x: x()
    def t():
        return 3
    
    t # should return '3'

    if wanting to use a staticproperty it will still work if
    not wanting to use a lambda expression for fixed returns.
    
    How to use:
    
    @staticproperty
    def test():
        global a
        return a

    a=3
    test # should return '3'
    a=5
    test # should return '5'
    
    class test:
        @staticproperty
        def test():
            global a
            return a
    a=3
    test.test # should return '3'
    a=5
    test.test # should return '5'
    """
    def __init__(self, func: Callable) -> None: self.func = func
    ## basically how it works is because it's instantiated
    ## it appears as a variable so all that needs to be 
    ## modified is the interactions it has with other objects
    def __getattr__(self, key: str):
        if key in object.__getattribute__(self, "__dict__"):
            return object.__getattribute__(self, key)
        elif key not in ["_ipython_canary_method_should_not_exist_","_ipython_display_","_repr_mimebundle_","_repr_html_","_repr_markdown_","_repr_svg_","_repr_png_","_repr_pdf_","_repr_jpeg_","_repr_latex_","_repr_json_","_repr_javascript_"]:
            return object.__getattribute__(self.func(),key)

@lambda x: x()
class share:
    """
    Stores given variables from __main__ allowing to be used in-module

    How to use:

    from my_pack import share

    share({"globals":globals()})
    # you can now use globals from main in the module
    To use in-module globals use 
    
    share.globals

    Note: Another way of obtaining global variables from
    main is to instantiate a scope class which will
    enable you to visit the module level frame where the
    global scope in the main program will be. Therefore,
    you should be able to also use scope().scope
    to view global variables from the main program
    """
    stored,globals={},globals()
    def __call__(self,*args: tuple[dict]) -> object: self.stored|=biop(args,"|");return self
    def __repr__(self) -> str: return repr(self.stored)
    def __getitem__(self,index: slice|int) -> dict: return self.stored[index]
    def __setitem__(self,index: slice|int,value: Any) -> None: self.stored[index]=value
    def __delitem__(self,index: int) -> None: del self.stored[index]
    def update(self,dct) -> None:
        for key,value in dct.items(): self[key]=value
    def get(self,key,default=None) -> Any: return self.stored.get(key,default=default)
    
def map_set(obj: Any,dct: dict,override: bool=True) -> None:
    """Sets mutliple attributes or keys to an object from a dict"""
    if override:
        for key,value in dct.items(): setattr(obj,key,value)
        return
    for key,value in dct.items():
        if hasattr(obj,key)==False: setattr(obj,key,value)
        else: raise Exception(f"object {obj} already has the key '{key}'")

def module_from_dict(module_name: str,dct: dict) -> ModuleType:
    """Creates a module object from a dictionary"""
    module=ModuleType(module_name)
    map_set(module,dct)
    return module

def get_builtins(form: dict|list|str=list) -> dict|list|ModuleType:
    """
    Gets a dict or list or ModuleType of the builtin functions.
    There seems to be differences on some applications/platforms/versions
    """
    if form not in [dict,list,"module"]: raise ValueError("Accepted inputs: dict|list|'module'")
    builtins_type=type(__builtins__) # either a module or dict
    if builtins_type==ModuleType:
        if form==dict: return __builtins__.__dict__
        if form==list: return dir(__builtins__)
    elif builtins_type==dict:
        if form==list: return list(__builtins__.keys())
        if form=="module": return module_from_dict("__builtins__",__builtins__)
    else: raise TypeError(f"__builtins__ is of an unexpected type: {builtins_type}")
    return __builtins__

## make sure the __builtins__ are a module type ##
if isinstance(__builtins__,ModuleType)==False: __builtins__=get_builtins("module")

def name(*args,depth: int=0,show_codes: bool=False) -> dict:
    """
    names the arguements passed into it. Note: doesn't do the kwargs.
    
    This function is only meant to name args passed in relative from the stack 
    frame depth as desired. Kwargs can be evaluated via kwargs.keys() assignment
    kwargs should be easily traceable within a function else pass them as kwargs
    if necessary.

    How to use:

    a,b,c=range(3)
    name(a,b,c)["args"] # should return ['a','b','c']

    print(name(a,b,c)) # should return {'func': 'name', 'args': ['a', 'b', 'c']}

    def test(*args,**kwargs):
        print(name(depth=1))
    test(a,c,{"a":{"a":3},"b":3}) # should print {'func':'test','args':['a','c']}
    test(a,c,{"a":"\n","b":3}) # should print {'func':'test','args':['a','c']}
    test(a,b,c) # should print {'func':'test','args':['a','b','c']}

    def test1(*args1):
        def test2(*args2):
            print(name(depth=1))
            return name(depth=2)
        return test2(*args1)
    test1(a,b,c) # should print {'func': 'test2', 'args': ['args1']}
                 # and return   {'func': 'test1', 'args': ['a', 'b', 'c']}
    """
    # get the frame and the start/end position in the stack
    frame=stack()[depth+1]
    CALL_position=tuple(frame.positions._asdict().values()) # position of the CALL opcode.opname # the end of the function
    string=frame.code_context[0][CALL_position[2]:] # the original line of code reduced by its offset
    PUSH_NULL_position=*(CALL_position[0],)*2,CALL_position[2],CALL_position[2]+len(slice_occ(string,"(")[0]) # the start of the function
    # get the op_codes
    frame=frame[0]
    op_codes=dis.get_instructions(frame.f_code)
    # skip to the starting position in the stack
    for op_code in op_codes:
        if tuple(op_code.positions._asdict().values())==PUSH_NULL_position:
            break
    else:
        print("\n\n")
        print("PREDICTED: ",PUSH_NULL_position)
        for i in dis.get_instructions(frame.f_code): print(i)
        print("\n\n")
        raise ValueError("PUSH_NULL_position was not found in the stack")
    
    # in case PUSH_NULL is not present i.e. when used inside functions
    if op_code.opname!="PUSH_NULL": op_codes=iter_chain((op_code,),op_codes)
    # get the function and args loaded
    call=[]
    for op_code in op_codes:
        if show_codes: print(op_code)
        if (op_code.opname=="LOAD_NAME" or 
            op_code.opname=="LOAD_FAST" or 
            op_code.opname=="LOAD_GLOBAL"): call+=[op_code.argval]
        elif op_code.opname=="LOAD_ATTR": call[-1]+="."+op_code.argval
        elif op_code.opname=="CALL": break
    
    return {"func":call[0],"args":call[1:]}

class scope:
    """
    gets the function name at frame-depth and the current scope that's within the main program
    
    Note: if using in jupyter notebook scope.scope will remove jupyter notebook specific attributes
    that record in program inputs and outputs. These attributes will still be available just not via 
    scope.scope because it causes a recursion error from some of the attributes changing while in use

    How to use:

    def a():
        c=3
        def b():
            c
            y=4
            print(scope(1).locals)
            print(scope().locals)
            print(scope().nonlocals)
            scope(1)["c"]=7
            print(scope(1).locals)
            print(scope().locals)
            print(scope().nonlocals)
        b()
        print(c)
    a()
    ## i.e. should print:
    {'b': <function a.<locals>.b at 0x000001DD9DFEDB20>, 'c': 3}
    {'y': 4, 'c': 3}
    {'c': 3}
    {'b': <function a.<locals>.b at 0x000001DD9DFEDB20>, 'c': 7}
    {'y': 4, 'c': 7}
    {'c': 7}
    7
    
    This allows us to change variables at any stack frame so long as it's on the stack
    """
    def __init__(self,depth: int=0) -> None:
        ## get the global_frame, local_frame, and name of the call in the stack
        global_frame,local_frame,name=currentframe(),{},[]
        while global_frame.f_code.co_name!="<module>":
            name+=[global_frame.f_code.co_name]
            global_frame=global_frame.f_back
            if len(name)==depth+1: local_frame=(global_frame,) # to create a copy otherwise it's a pointer
        ## instantiate
        if depth > (temp:=(len(name)-1)): raise ValueError(f"the value of 'depth' exceeds the maximum stack frame depth allowed. Max depth allowed is {temp}")
        name=["__main__"]+name[::-1][:-(1+depth)]
        self.depth=len(name)-1
        self.name=".".join(name)
        self.local_frame,self.global_frame=local_frame[0],global_frame
        self.locals,self.globals,self.nonlocals=local_frame[0].f_locals,global_frame.f_locals,nonlocals(local_frame[0])

    def __repr__(self) -> str:
        """displays the current frames scope"""
        return repr(self.scope)
    @property
    def scope(self) -> dict:
        """The full current scope"""
        if has_IPython():
            ## certain attributes needs to be removed since it's causing recursion errors e.g. it'll be the notebook trying to record inputs and outputs most likely ##
            not_allowed,current_scope=["_ih","_oh","_dh","In","Out","_","__","___"],{}
            local_keys,global_keys=list(self.locals),list(self.globals)
            for key in set(local_keys+global_keys):
                if (re.match(r"^_i+$",key) or re.match(r"^_(\d+|i\d+)$",key))==None:
                    if key in not_allowed: not_allowed.remove(key)
                    elif key in local_keys:
                        current_scope[key]=self.locals[key]
                        local_keys.remove(key)
                    else: current_scope[key]=self.globals[key]
            return current_scope
        current_scope=self.globals.copy()
        current_scope.update(self.locals)
        return current_scope
    
    def __getitem__(self,key: Any) -> Any: return self.locals[key] if key in self.locals else self.globals[key]
    def update(self,dct) -> None:
        for key,value in dct.items(): self[key]=value
    def get(self,key,default=None) -> Any: return self.scope.get(key,default=default)
    def __setitem__(self,key: Any,value: Any) -> None:
        if key in self.locals:
            self.locals[key]=value
            # code reference: MariusSiuram (2020). https://stackoverflow.com/questions/34650744/modify-existing-variable-in-locals-or-frame-f-locals,CC BY-SA 4.0
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.local_frame), ctypes.c_int(0))
        else: self.globals[key]=value

    def __delitem__(self,key: Any) -> None:
        if key in self.locals:
            del self.locals[key]
            # code reference: https://stackoverflow.com/questions/76995970/explicitly-delete-variables-within-a-function-if-the-function-raised-an-error,CC BY-SA 4.0
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(1))
        else: del self.globals[key]

def id_dct(*args) -> dict:
    """Creates a dictionary of values with the values names as keys (ideally)"""
    names=name(depth=1)["args"]
    return dict(zip(names,args))

def refs(*args,scope_used: dict=None) -> list[list[str]]:
    """Returns all variable names that are also assigned to the same memory location within a desired scope"""
    if scope_used==None: scope_used=scope(1).scope # depth set to '1' to get passed the the refs stack frame
    return [[key for key,value in scope_used.items() if value is arg] for arg in args]

def list_join(ls1: list[str],ls2: list[str]) -> str:
    """
    joins a list by another list to produce a joined string (usually)
    Inspiration for this function was from "".join(some_string)
    but for custom joining at each point that can be made to join up
    """
    return biop(tuple(ls1_item+ls2_item for ls1_item,ls2_item in zip(ls1[:-1],ls2)),"+")+ls1[-1]

## needs testing ##
def use_numbers(string: str,chain: bool=False) -> str:
    """Replaces all numbers in a string expressed as words to numbers. Optionally allows reading numbers off as chained expressions"""
    ## prep for zip
    numbers=dict(
        base={0:("zero","one","two","three","four","five","six","seven","eight","nine"),1:tuple(range(10))},
        synonym={0:("no","none","single","couple","pair","few","dozen","bakersdozen"),1:(0,0,1,2,2,3,12,13)},
        special={0:("ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen",
                    "nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"),1:(*range(10,20),*range(20,100,10))},
        scale={0:("hundred","thousand","million","billion","trillion","quadrillion",
                  "quintillion","sextillion","septillion","octillion","nonillion",
                  "decillion","undecillion","duodecillion","tredecillion","quattuordecillion",
                  "quindecillion","sexdecillion","septendecillion","octodecillion","novemdecillion",
                  " vigintillion "),1:tuple(10**i for i in (2,)+tuple(range(3,64,3)))}
    )
    numbers=dct_join(*numbers.values())
    numbers=tuple(biop(numbers[i],"+") for i in range(2))
    ## get the words
    words,new_word=iter(string.lower().split()),[]
    def try_form(word: str) -> str:
        """for dealing with combining strings"""
        if "*" in word:
            if word[0]=="*":
                word=word[1:]
            else:
                return str(eval(re.sub("r[a-zA-Z]+","",word)))
        try:
            return str(eval(re.sub(r"\D+","",word)))+re.sub(r"\d+","",word)
        except Exception:
            return str(eval(re.sub(r"\D+","",word)))

    ## replace the words with numbers
    for word in words:
        flag=False
        for key,value in zip(*numbers):
            if key in word:
                flag=True
                if value % 10 == 0 and value > 9:
                    word=word.replace(key,"*"+str(value))
                else:
                    word=word.replace(key,str(value))

        ## for the use of hyphens '-'
        if flag:
            if "-" in word:
                word=word.replace("-","+")
            word=try_form(word)
        new_word+=[word]
    new_word=" ".join(new_word)
    ## fix the joining of words if applicable
    if chain:
        match=lambda string: re.match("\d+",string)!=None
        joins=[]
        for join in [" "," and "," point "]:
            temp_word=new_word.split(join)
            for i in range(len(temp_word)-1):
                if match(num1:=temp_word[i].split()[-1]) and match(num2:=temp_word[i+1].split()[0]):
                    num1,string_part,num2=re.sub(r"\D+","",num1),re.sub(r"\d+","",num2),re.sub(r"\D+","",num2)
                    if join==" " or join==" and ":
                        joins+=[str(eval(num1+"+"+num2))+string_part]
                    else:
                        joins+=[num1+"."+num2]
            if joins:
                #### needs testing ####
                try:
                    new_word=list_join(re.split(f"\d+{join}\d+",new_word),joins)
                except:
                    new_word=list_join(re.split(f"\d+\D+{join.strip()}\D+\d+",new_word),joins)
                joins=[]
    return new_word

def get_arg_count(attr: Any,values: tuple[Any]=(
    [None]*999,iter([None]*999),[0]*999
                 )) -> list:
    """returns the number of accepted args. Note: doesn't do the kwargs"""
    ## either it's an invalid type
    ## or any of the numbers less than or equal to the length
    if isinstance(attr,Callable)==False: raise TypeError("attr must be Callable")
    if attr.__name__ not in dir(__builtins__):
        try: ## see if it has a signature
            return [len(signature(attr).parameters)]
        except: ## see if it uses an initialization
            try: return [len(signature(attr.__init__).parameters)]
            except: pass
    ## this next section of code shouldn't be used since builtins like 'type' use __init__ rather than __call__ for the signature which was a previous oversight ##
    ## but if for whatever reason it has no __init__ signature then it defaults to a brute force like approach ##
    length=len(values[0])
    if attr==print or attr==display: return [length]
    original_message=""
    ## Intentionally runs tests for errors to infer the number of args allowed
    for value in values:
        try:
            attr(*value)
            return [length]    
        except TypeError as e:
            original_message=str(e)
            message=" ".join(original_message.split(" ")[1:])
            if " takes no arguements " in message: return [0]
            if " exactly one  " in message: return [1] ## may re-write
            if "at least two arguments" in message: return [2]  ## may re-write
            arg_numbers=re.findall(r"\d+",message)
            if len(arg_numbers): return [num for i in arg_numbers if (num:=int(i)) < length]
    raise ValueError(f"the value of 'value' should be reconsidered for the attribute '{attr}'.\n\n Original message: "+original_message)

def find_args(obj: ModuleType|object,use_attr: bool=True,value: Any=[None]*999) -> list:
    """For figuring out how many args functions use"""
    messages=[]
    toggle_print()
    for attr in dir(obj):
         ## these can be done individually
        if attr=="print" or attr=="display": continue
        try:
            attribute=getattr(obj,attr)
            if isinstance(attribute,Callable):
                if use_attr==False: attr=""
                try: attribute(*value)
                except TypeError as e: messages+=[attr+" "+" ".join(str(e).split(" ")[1:])]
                else: messages+=[attr+" "+"any"]
        except: pass
    toggle_print()
    return messages

def toggle_print() -> None:
    """Toggles between enabling and disabling printing to display"""
    global print,display
    print,display=(__builtins__.print,IPython.core.display_functions.display) if print("",end="") and display() else (lambda *args,**kwargs: True,)*2

def class_dict(obj: Any,warn: bool=False) -> dict:
    """For obtaining a class dictionary (not all objects have a '__dict__' attribute)"""
    if warn==False: simplefilter("ignore") ## some attributes are deprecated and they may throw warnings
    keys,attrs=dir(obj),[]
    for key in keys:
        try: attrs+=[getattr(obj,key)]  ## some attrs cannot be retrieved i.e. ,"__abstractmethods__" when passing in the 'type' builtin method
        except: pass
    return dict(zip(keys,attrs))

class classproperty:
    """
    In python version 3.13 class properties are disallowed
    This would be the equivalent of:

    classmethod(property(obj))
    or 
    @classmethod
    @property
    def some_function():
    """
    def __init__(self, fget: Callable=None) -> None: self.fget,self.__doc__=fget,fget.__doc__
    def __get__(self, obj: None, objtype: type|None=None) -> Any: return self.class_method(self.fget)(obj)
    
    def class_method(self,FUNC: Callable) -> Callable:
        """Creates a classmethod wrapper"""
        if hasattr(self,"wrapper"): return self.wrapper
        if hasattr(self,"cls")==False:
            name=FUNC.__qualname__.split(".")[0] # get the classes name
            self.cls=scope()[name] # it should be in the global scope
        number_of_args=get_arg_count(FUNC)[0]
        if number_of_args==0: raise TypeError(f"method '{FUNC.__name__}' must have at least one arguement as the class for it to be a classproperty")
        if number_of_args==1:
            @wraps(FUNC)
            def wrapper(*args,**kwargs): return FUNC(self.cls)
        else:
            @wraps(FUNC)
            def wrapper(*args,**kwargs): return FUNC(self.cls,*args,**kwargs)
        self.wrapper=wrapper
        return wrapper

def class_copy(cls: type) -> type:
    """
    copies a class since somtimes using copy.deepcopy can sometimes return a pointer for types
    
    creates a new class that is identical to the original class
    """
    return type(cls.__name__,(),dict(cls.__dict__))

def create_separate_class(func: Callable) -> Callable:
    """
    defines a wrapper function that can be repeatedly called allowing new classes at different ids to be created.
    We cannot use only the wrapper as say a static property otherwise we cannot repeatdly call new classes because
    being used as a decorator will call it upon definition which therefore is the same class
    """
    @wraps(func)
    def wrapper(*args,**kwargs) -> Callable:
        """creates a new class seperate from the original class"""
        return class_copy(func)(*args,**kwargs)
    return wrapper

#### needs testing #### - need to fix the wrapper
@create_separate_class
class chain:
    """
    if wanting to apply to the object and keep a chain going
    Examples of how to use:
    
    def testing():
        print("hello")
    chain().testing() # global method added
    chain([1,2,3]).sum() # builtin method added
    a=[1,2,3]
    chain().a # attribute added
    chain(pd.Series([1,2,3]))._.explode() # objects methods
    chain(pd.Series([1,2,3]))._.explode()._.testing() # switching between local and global scope
    
    It should also be able to inherit methods i.e.
    
    j=1
    chain(j)**3 # should return 1
    chain(j)+j  # should return 2
    chain(j)*3  # should return 3
    ## full chain example:
    
    a=[1,2,3]
    b=pd.Series
    c=pd.DataFrame
    chain().a.sorted(reverse=True).b().c().tuple()[0].BREAK
    # or
    chain()([1,2,3]).sorted(reverse=True)(pd.Series)(pd.DataFrame).tuple()[0].BREAK

    ## overriding builtins (not recommended; but is possible):
    def sum():
        print("hi")
    chain([1,2,3])._g.sum() # should print 'hi'
    # using a Callable as data rather than a method:
    chain()([1,2,3]).__(pd.Series)
    
    Note: all data and methods (except special methods) have been made private in this class
    to allow for more commonly named attributes to be added.
    
    In python private data and methods strictly don't exist (in what I know currently) i.e.
    in the chain class we have __cache as a private variable but this is accessible via:
    
    chain._chain__cache
    and can be assigned new values or overwritten
    """
    _chain__obj,_chain__use_locals,_chain__use_builtin=0,False,True
    def __init__(self,obj: Any) -> None:
        self.__obj=obj
        self.__update_bases
    @property
    def __update_bases(self) -> object:
        """Finds the new dunder methods to be added to the class"""
        # all dunder methods not allowed to be shared (else the chain classes attributes needed for it to work will get overwritten)
        not_allowed=["__class__","__getattribute__","__getattr__","__dir__","__set_name__","__init_subclass__","__mro_entries__",
                   "__prepare__","__instancecheck__","__subclasscheck__","__sizeof__","__fspath__","__subclasses__","__subclasshook__",
                   "__init__","__new__","__setattr__","__delattr__","__get__","__set__","__delete__","__dict__","__doc__","__call__",
                   "__name__","__qualname__","__module__","__abstractmethods__","__repr__"]
        for key,value in class_dict(self.__obj).items():
            if re.match("__.*__",key)!=None:
                if key in not_allowed: not_allowed.remove(key)
                elif isinstance(value,Callable): ## we're only wanting instance based methods for operations such as +,-,*,... etc.
                    self.__class_support(key,self.__wrap(value)) ## class methods # we need to link it to a new class
        return self

    def __wrap(self,method: Callable) -> Callable:
        """
        wrapper function to ensure methods assigned are instance based 
        and that the dunder methods return values are wrapped in a chain object
        if i.e. used in a binary operation or that these are left as is if 
        type casting a chain object e.g. float(chain(1)) should return 1.0
        and its type should be float and not my_pack.chain or __main__.chain if 
        defined in program
        """
        @wraps(method) ## retains the docstring
        def wrapper(*args): return method(*args[1:])
        return wrapper
    @classmethod
    def __class_support(cls,key: str,value: Any) -> None: setattr(cls,key,value)
    def __repr__(self) -> str: return repr(self.__obj)

    def __call__(self,*args, **kwargs) -> object:
        """Modified to update the object in the chain to keep the chain going"""
        self.__obj=self.__obj(*args, **kwargs) if hasattr(self.__obj,"__init__") or hasattr(self.__obj,"__call__") else \
        args[0](self.__obj,*args[1:],**kwargs) if isinstance(args[0],Callable) else args[0]
        return self.__update_bases

    def __get_attr(self,key: str) -> Any:
        ## the class instance always comes first
        if key in self.__dict__: return super().__getattr__(key)
        ## then the object 
        if key in dir(self.__obj) and self.__use_locals: return getattr(self.__obj, key)
        # attribute does not exist in either class or object, therefore, it must be either a builtin or in the current scope
        if hasattr(__builtins__,key) and self.__use_builtin: return getattr(__builtins__,key)
        else:
            try: return scope(2)[key]
            except KeyError: raise AttributeError(f"Attribute '{key}' does not exist in the current scope or in the current objects configuration")

    def __check(self,attribute: Any) -> Any:
        if isinstance(attribute,Callable):
            ## pass in the object stored to the Callable
            if sum(get_arg_count(attribute)) > 0: return attribute if type(attribute).__name__=="method" else partial(attribute,self.__obj)
            ## if the Callable has no params then it has to be a staticmethod
            ## (because it's set to an instance of a class which means it will expect 'self' as the first arg)
            return attribute if isinstance(attribute,staticmethod) else staticmethod(attribute)
        return attribute
    
    def __getattr__(self,key: str) -> object:
        """Modified to update the object in the chain to keep the chain going"""
        # check the class first then the object
        attribute=self.__get_attr(key)
        self.__obj=self.__check(attribute)
        return self.__update_bases
    @property
    def _(self) -> object:
        """Changes scope from global to local or local to global"""
        self.__use_locals=not self.__use_locals ## since it's a bool
        return self

    def __(self,obj: Any) -> object:
        """
        Allows methods to be passed in to be set as an attribute e.g. not to be called.
        Note: this method is basically just self.__chain but for short hand global use which 
        assumes it's more likely to get overriden even though it shouldn't.
        """
        self.__obj=obj
        return self.__update_bases
    @property
    def _g(self) -> object:
        """
        Allows overriding of builtins. Shouldn't be needed but 
        if for whatever reason monkey patching builtins seems 
        appropriate then this allows use of them in the chain
        """
        self.__use_builtin=not self.__use_builtin
        return self

    @property
    def BREAK(self) -> Any:
        """Breaks the chain e.g. returns the final object"""
        return self.__obj

class tup_ext:
    """Extensions for tuples"""
    def __init__(self,tup: tuple) -> None: self.tup=tup
    def __repr__(self) -> str: return str(self.tup)
    def __len__(self) -> int: return len(self.tup)
    def __getitem__(self,index: int|tuple|list|slice) -> tuple: return self.__getter(index,self.tup)
    def __getter(self,index: int|tuple|list|slice,indexes: tuple|list) -> Any: return itemgetter(*index)(indexes) if type(index)==tuple or type(index)==list else itemgetter(index)(indexes)
    
    def __setitem__(self,indexes: int|tuple|list|slice,values: tuple[Any]) -> None:
        temp=list(self.tup)
        if not isinstance(indexes,list|tuple): temp[indexes]=values
        else: map_set(temp,dict(zip(indexes,values)))
        self.tup=tuple(temp)
        return self

    def __delitem__(self,index: int|slice|tuple) -> None:
        indexes=list(range(len(self.tup)))
        remove=self.__getter(index,indexes)
        if type(remove)!=list: remove=[*remove]
        new_tup=tuple()
        for index,value in enumerate(self.tup):
            if index not in remove: new_tup+=(value,)
            else: remove.remove(index)
        self.tup=new_tup
        return self

class Print:
    """In-time display"""
    def __init__(self,initial: int=0) -> None: self.prev=initial
    def __call__(self,message: Any) -> None:
        self.clear
        print(message,end="\r")
        self.prev=len(message)
    @property
    def clear(self) -> None: print(" "*self.prev,end="\r")

def import_sklearn_models(kind: str) -> None:
    """Convenience function for importing lots of sklearn models"""
    if kind=="classifiers":
        models=zip(["tree","neighbors","ensemble","linear_model","naive_bayes","dummy","neural_network","svm"],
                ["DecisionTreeClassifier","KNeighborsClassifier","RandomForestClassifier,GradientBoostingClassifier",
                    "LogisticRegression","GaussianNB","DummyClassifier","MLPClassifier","SVC"])
    elif kind=="regressors":
        models=zip([],[])
    else: raise ValueError("'kind' must be in [\"classifiers\",\"regressors\"]")
    for directory,model in models: exec("from sklearn."+directory+" import "+model,globals())

def all_multi_combos_dict(arr: dict) -> list:
    """Same as all_multi_combos but for retaining key/column placing"""
    def all_multi_combos(current_combo: list=[]) -> list:
        nonlocal arr
        """Returns a list of all combinations of multiple lists"""
        all_combos,index=tuple(),len(current_combo)
        if index==len(arr): return [current_combo]
        for i in list(dct_ext(arr)[index].values())[0]: all_combos=(*all_combos,*all_multi_combos(current_combo+[i]))
        return all_combos
    return all_multi_combos()

def all_multi_combos(arr: list[list],current_combo: list=[]) -> list:
    """Returns a list of all combinations of multiple lists"""
    all_combos,index=tuple(),len(current_combo)
    if index==len(arr): return [current_combo]
    for i in arr[index]: all_combos=(*all_combos,*all_multi_combos(arr,current_combo+[i]))
    return all_combos
@property
def reset_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns=range(len(df.columns))
    return df

pd.DataFrame.reset_cols=reset_cols

def all_combinations(ls: list,start: int=1,stop: int=0) -> iter:
    """Returns all the unique combinations"""
    stop=len(ls)+1-stop
    return (combo for i in range(start,stop) for combo in combinations(ls,i))

SKLEARN_IMPORTED=dict.fromkeys(["classifier","regressor"],False)
def get_classifier(mod: str="",show: bool=False,plot: bool=False,**kwargs) -> tuple[Callable,str]|Callable:
    """Convenience function for obtaining common sklearn and other classifier models"""
    global SKLEARN_IMPORTED
    if show: return print("Available models: tree, knn, forest, nb, dummy, nnet, svm, gb, log")
    if SKLEARN_IMPORTED["classifier"]==False:
        import_sklearn_models("classifier")
        SKLEARN_IMPORTED["classifier"]=True
    get=lambda _mod: _mod if len(kwargs)==0 else _mod(**kwargs)
    match mod:
        case "tree": FUNC,depth = get(DecisionTreeClassifier),"Depth"
        case "knn": FUNC,depth = get(KNeighborsClassifier),"neighbors"
        case "forest": FUNC,depth = get(RandomForestClassifier),"estimators"
        case "nb": FUNC,depth = get(GaussianNB),"iteration"
        case "dummy": FUNC,depth = get(DummyClassifier),"iteration"
        case "nnet": FUNC,depth = get(MLPClassifier),"iteration"
        case "svm": FUNC,depth = get(SVC),"iteration"
        case "gb": FUNC,depth = get(GradientBoostingClassifier),"iteration"
        case "log": FUNC,depth = get(LogisticRegression),"iteration"
        case _: return print(f"model '{mod}' doesn't exist")
    return (FUNC,depth) if plot else FUNC

## needs regressors added
def get_regressor(mod: str="",show: bool=False,plot: bool=False,**kwargs) -> Callable:
    """Convenience function for obtaining common sklearn and other regressor models"""
    raise NotImplementedError("No regressors added yet")
    global SKLEARN_IMPORTED
    if show: return print("Available models: ")
    if SKLEARN_IMPORTED["regressor"]==False:
        import_sklearn_models("regressor")
        SKLEARN_IMPORTED["regressor"]=True
    get=lambda _mod: _mod if len(kwargs)==0 else _mod(**kwargs)
    match mod:
        case _: return print(f"model '{mod}' doesn't exist")
    return FUNC

def dct_join(*dcts: tuple[dict]) -> dict:
    """For concatenating multiple dicts together where all have the same keys"""
    keys=biop(map(lambda x:x.keys(),dcts),"|")
    return {key: [dct[key] if key in dct else None for dct in dcts] for key in keys}

def biop(data: Any,op: str) -> Any:
    """applies a reduce using a binary operator"""
    return reduce(lambda x,y: eval("x"+op+"y"),data)

FILE_SAVE_INDEX={}
def unidir(path_name: str="",ext: str="") -> None:
    """Finds a unique file name in the specified directory"""
    global FILE_SAVE_INDEX
    file_name=path_name+"."+ext
    if file_name in FILE_SAVE_INDEX: index=FILE_SAVE_INDEX[file_name]
    else: index=FILE_SAVE_INDEX[file_name]=0
    while True:
        file_name=f"{path_name}_{index}.{ext}"
        if os.path.isfile(file_name)==False: return file_name
        index+=1

def pd_read(*args,method: Callable=pd.read_csv,**kwargs) -> tuple[pd.DataFrame,...]:
    """Convenience function For reading multiple csvs into pandas DataFrames"""
    return (method(filename,engine="pyarrow",**kwargs) for filename in args)

def unique(ls: list,order: bool|str="retain") -> list:
    """Returns the unique values of a list with specified ordering"""
    if order:
        if order=="retain":
            temp=[]
            for i in ls:
                if temp.count(i)==0:
                    temp+=[i]
            return temp
        return np.unique(ls).tolist()
    return list(set(ls))

def create_password(length: int=12,selection: None|str=None,char_range: int=range(127)) -> str:
    """
    creates a password using cryptographic pseudo-random numbers
    ## code reference: https://stackoverflow.com/questions/3854692/generate-password-in-python,CC BY-SA 4.0
    # changes made: allowed for a wider range of characters
    """
    if selection==None:
        if isinstance(char_range,int): char_range=range(char_range)
        selection="".join(char for i in char_range if "\\" not in repr((char:=chr(i))))
    return "".join(secrets.choice(selection) for _ in range(length))

def random_shuffle(arr: list|str) -> list|str:
    """Pseudo-randomly shuffles a list or string"""
    flag,index=0,[]
    if isinstance(arr,str): flag,arr=1,list(arr)
    for i in range(len(arr)):
        temp=secrets.choice(range(len(arr)))
        index+=[arr[temp]]
        del arr[temp]
    return "".join(index) if flag else index

def format_password(upper: int=0,lower: int=0,punc: int=0,num: int=0,other: int=0,char_range: int=range(127)) -> str:
    """Creates a new password formatted to password rules"""
    selection=""
    if upper: selection+=create_password(upper,string.ascii_uppercase)
    if lower: selection+=create_password(lower,string.ascii_lowercase)
    if punc: selection+=create_password(punc,string.punctuation)
    if num: selection+=create_password(num,string.digits)
    if other: selection+=create_password(other,char_range=char_range)
    return random_shuffle(selection)

def save_tabs(filename: str,*args,**kwargs) -> None:
    """save urls to .txt file"""
    with open(filename,"w") as file: file.write("\n".join(get_tabs(*args,**kwargs)))

def browse_from_file(filename: str) -> None:
    """Browses urls from .txt file"""
    with open(filename, 'r') as file: urls=[url for line in file if (url:=line.strip())]
    browse(urls) if urls else print("No links found in the file.")

def browse(urls: list[str]) -> None:
    """Uses Google Chrome to browse a selection of links"""
    for url in urls: webbrowser.open(url, new=2, autoraise=True)

def get_url() -> str:
    """get url from google chrome tab"""
    pyautogui.hotkey('ctrl', 'l')
    pyautogui.hotkey('ctrl', 'c')
    tk_obj=Tk()
    result=tk_obj.clipboard_get()
    tk_obj.destroy() ## pops up when using a CLI ## there's no option to turn off the popup on __init__ so it has to be saved then removed ##
    return result

def get_tabs(delay: int|float=1.5,close_init: bool=False) -> list[str]:
    """
    Gets all links from currently open tabs in google chrome
    (stops when there's a repeat e.g. only accepts a unique set of tabs)
    """
    webbrowser.open("https://www.google.com", new=2, autoraise=True)
    sleep(delay)
    pyautogui.hotkey('ctrl', 'tab')
    links,temp=[],get_url()
    while temp not in links:
        pyautogui.hotkey('ctrl', 'tab')
        links+=[temp]
        temp=get_url()
    ## in case for whatever reason something should happen it's possible that
    ##  you might interfere with it and it may close some other application
    if close_init: pyautogui.hotkey('ctrl', 'w')
    return links

def cwd() -> None:
    """convenience function for openning file explorer at the cwd"""
    os.startfile(os.getcwd())

## needs testing (probably can't handle additional *args and **kwargs annotations and needs some exception handling for length mismatches)
def type_check(FUNC: Callable,inputs: bool=True,**kwargs) -> None:
    """For validating types against their type annotations"""
    def try_check(arg: Any,annotation: type,key: Any,message: str) -> None:
        """For handling the checking of each arguements type"""
        def checker(annotation: type,arg: Any) -> None:
            nonlocal message
            if isinstance(arg,annotation)==False: raise TypeError(message)
        if type(annotation) in (tuple,list,set):
            if type(arg) not in (tuple,list,set): raise TypeError(f"arguement '{key}' must be of type {annotation}. Instead recieved: {arg}")
            if len(arg)!=len(annotation): raise Exception(f"length mismatch between arguement '{key}' and annotation {annotation}")
            for type_annotation,arguement in zip(annotation,arg): checker(type_annotation,arguement)
        else: checker(annotation,arg)

    args,kwargs,annotations=kwargs["args"],kwargs["kwargs"],FUNC.__annotations__
    try: annotations["return"]
    except: raise KeyError("Key 'return' does not exist in .__annotations__. You must annotate a return type")
    ## assuming it has a return type
    params=signature(FUNC).parameters ## will use this later to fix the *args **kwargs problem ##
    if len(params)!=len(annotations)-1:
        ## which are missing ##
        missed=[key for key in params.keys() if params[key].annotation==_empty]
        raise Exception(f"The following parameters have no type annotations: {missed}")
    if inputs:
        ## do all the kwargs first
        for key,value in kwargs.items():
            try:
                try_check(*(value,annotations[key],key,f"arguement '{key}' must be of type {annotations[key]}"))
                del annotations[key]
            except: pass
        ## then the args
        args=iter(args)
        for key,annotation in annotations.items():
            if key=="return": continue
            try: arg=next(args)
            except StopIteration: break
            try_check(*(arg,annotation,key,f"arguement '{key}' must be of type {annotation}"))
        ## check for kwargs, args specific types
    else: try_check(*(arg,annotations["return"],"return",f"return arguement must be of type {annotations['return']}"))

class sanitize:
    """
    My class for sanitizing functions inputs 
    and/or outputs and other attributes
    i.e. can be used as decorator to always first 
    validate the input types (needs testing at the moment)
    
    @sanitize
    def do(x: int|float) -> None:
        print("hi")
    do("hello") # should raise a TypeError

    ## you can also add your own checks that the functions 
    ## arguements must pass by adding them i.e.:
    sanitize.add(*desired function*)
    ## Note: this will make all sanitize class instances
    ## make use of the functions added
    
    ## if you only want particular checks applied
    ## to the instances specified and not to all instnaces
    ## you can use:
    @sanitize.use(*desired function*,defaults=False) ## defaults=False won't use the default type checker
    def do(x: int|float) -> None:
        print("hi")
    """
    checks=(type_check,) ## default check used ##
    
    def __init__(self,FUNC: Callable,args: tuple[Callable,...]=tuple(),defaults: bool=True) -> None:
        """retains the function and common metadata or attributes"""
        self.FUNC,self.__doc__,self.__annotations__=FUNC,FUNC.__doc__,FUNC.__annotations__
        self.checks=(*self.checks,*args) if defaults else args

    def __repr__(self) -> str: return str(self.FUNC)
    def __call__(self,*args,**kwargs) -> Callable:
        """Runs through all the checks before calling using the functions arguements"""
        for __check in self.checks:
            __check(self.FUNC,args=args,kwargs=kwargs)
        return self.FUNC(*args,**kwargs)
    @classmethod  ## if we want global class manipulation before instantiation we have to use @classmethod ##
    def add(cls,*args: Callable) -> type:
        """adds additional custom checks to inputs (globally e.g. for all instances)"""
        cls.checks=(*cls.checks,*args)
        return cls
    @classmethod
    def remove(cls,*args: Callable) -> type:
        """removes checks from all instances of the class"""
        cls.checks=tuple(__check for __check in cls.checks if __check not in args)
        return cls
    @classmethod
    def use(cls,*args: Callable,defaults: bool=True) -> object:
        """adds additional custom checks to inputs (locally e.g. only on the instance used)"""
        return partial(cls,args=args,defaults=defaults) ## make an instance of the class with the new args ##

class Sub:
    """shorthand version of re.sub"""
    def __init__(self,code: str) -> None: self.code=code
    def __repr__(self) -> str: return self.code
    def __call__(self,regex: str,repl: str="",flags=re.DOTALL) -> object:
        """For string substitution with regex"""
        self.code=re.sub(regex,repl,self.code,flags=flags)
        return self

    @property
    def get(self) -> str: return self.code

def extract_code(code: str,repl: str=" str ") -> Sub:
    """Removes all strings and comments from code"""
    sub=Sub(code)
    ## remove all double backslashes (i.e. "\\" is possible for a string) and distinguish strings ##
    sub(r"\\\\")
    sub(r"\\\"|\\\'")
    ## remove all strings
    sub.code=remove_docstrings(sub.code)
    sub.code=remove_strings(sub.code)
    # remove all comments
    sub.code+="\n"
    return sub(r"#(.+?)\n","\n")

## needs testing ## - might use an ast instead
## TODO: 1. fix for when source == None
## TODO: 2. fix for when definitions are overrided
## TODO: 3. fix for functools.wraps, functools.partial, and other class wrappers
## TODO: 4. need to account for what happens when variables are not available; in the get_code_requirements function after collecting all the variables
def export(section: str | Callable,source: str | None=None,to: str | None=None,option: str="w",show: bool=False,recursion_limit: int=10) -> str | None:
    """
    Exports code to a string along with any required code that can then be used to write to a file or for use elsewhere
    Example: (save the following into a file called test.py)
    -------------------------------------------
    from my_pack import preprocess

    def something():
        return "asdf;kas \""
    
    class func:pass
    
    c="asdf;kas \"";func
    d='\''
    
    def hi(b=2):
        a=3
        can()
        so()
        return
    def f():
        return so()
    def can():
        f()
        print(5)
    def so():
        print(5)
    -------------------------------------------
    Then run:
    -------------------------------------------
    from test import hi
    from my_pack import export

    export(hi,show=True,to="new.py",option="a")
    -------------------------------------------
    It will write to a file called new.py
    and should print:
    
    initial section:
    --------------------
    def hi(b=2):
        a=3
        can()
        so()
        return
    
    --------------------
    Recursion: 0:
    --------------------
    def hi(b=2):
        a=3
        can()
        so()
        return
    
    def can():
        f()
        print(5)
    
    def so():
        print(5)
    
    --------------------
    Recursion: 1:
    --------------------
    def hi(b=2):
        a=3
        can()
        so()
        return
    
    def can():
        f()
        print(5)
    
    def so():
        print(5)
    
    def f():
        return so()

    --------------------
    """
    # get source_code if Callable
    if isinstance(section,Callable):
        section,callables,source=source_code(section),[section.__name__],section.__module__
    else:
        ## since it's a section of code with no source file it belongs to
        if source==None:
            ## TODO: make this work for ipynb
            # scope().scope['__name__'] # ? # use the current process ## (not sure how for .ipynb)
            raise NotImplementedError("Not yet implemented the source == None case")
        ## check for functions + classes ## raw strings only
        callables=extract_code(section).get
        callables={i[3:-1].strip() for i in re.findall(r"def\s*\w*\s*\(",callables)}|\
                  {i[5:-1].strip() for i in re.findall(r"class\s*\w*\s*\(|\:",callables)}#|\
                  #{i[5:-1].strip() for i in re.findall(r"class\s*\w*\s*\(|\:",callables)} # from ... import ...
    # prep section
    variables,code_export=get_variables(section),section
    if len(variables) > 0:
        # why all the callables and not all the attrs???
        # gather all functions and classes available to the source file of interest
        new_callables,source=all_callables(source,True)
        callables_in_source=set(new_callables).difference(callables)
        # start exporting code
        if show: print("initial section:\n"+"-"*20+"\n"+section+"\n"+"-"*20)
        code_export,modules,module_names=get_code_requirements(*(section,callables_in_source,variables,variables,source,show),limit=recursion_limit)
        ## add the modules ##
        if len(modules) > 0:
            header="\n" if option=="a" else ""
            for key,module in modules.items():
                if len(module)==0:
                    header+="import "+key
                    name=module_names[module_names[0]==key][2]
                    if len(name) > 0:
                        header+=" as "+list(name)[0]
                else:
                    header+=f"from {key} import "+", ".join(module)
                header+="\n"
            # append all modules to the top of the section
            code_export=header+"\n"+code_export
    if to==None: return code_export
    with open(to,option) as file: file.write(code_export)


def get_code_requirements(section: str,callables: list[str],variables_to_export: list[str],variables_present,source: str,
    show: bool=False,modules: dict={},current_modules: pd.DataFrame=pd.DataFrame(),recursions: int=0,limit: int=20) -> tuple[str,dict,pd.DataFrame]:
    """Gets the required code in order to export a section of code from a .py file maintainably"""
    # separate variables and attributes
    attrs,variables=split_list(variables_to_export,lambda variable:True if "." in variable else False)
    ## search attrs for callables and modules (and ideally monkey patches) ##
    callable_exports,definitions,callables,module_names=get_attr_exports(*(attrs,source,callables))
    ## do the same but for the variables and then combine ##
    new_exports,callables=split_list(callables,lambda func:True if (func.__name__ in variables)==True else False)
    new_exports+=callable_exports
    if len(new_exports) > 0:
        ## add the new code ##
        for func in set(new_exports):# a list of functions from the module
            try:
                exec(f'temp=__import__("{source}").{func.__name__}')
            except Exception as e:
                name=module_names[module_names[0]==func].dropna()
                if len(name) > 0:
                    exec(f'temp=__import__("{source}").{list(name[2])[0]}')
                    name[0]=list(name[0])[0].__name__
                    current_modules=pd.concat([current_modules,name])
                else: ## attribute doesn't exist
                    continue
            section,modules=add_code(*(section,modules,locals()["temp"],source))
        section+=definitions
        ## print the current Recursion ##
        if show: print(f"Recursion: {recursions}"+":\n"+"-"*20+"\n"+section+"\n"+"-"*20)
        ## limit the amount of variables needing to be used ##
        ## ! this also allows potential run-time errors (because of overwriting definitions) ##
        new_variables_present=get_variables(section)
        ## get the variables that are not present in the current code section ##
        variables_to_export=new_variables_present.difference(variables_present)
        if len(variables_to_export)==0:
            return section,modules,current_modules
        ## make sure there's some safety in case errors occur ##
        if recursions==limit:
            warn(f"recursion limit of {limit} reached\n\nNote: the function did not complete the exportion. To avoid this and run to completion adjust the recursion limit and/or enter in the current code section")
            return section,modules
        recursions+=1
        ## you have to return the recursion else it won't work properly ##
        return get_code_requirements(*(section,callables,variables_to_export,new_variables_present,source,show,modules,current_modules,recursions))
    return section+definitions,modules,current_modules

def add_code(section: str,modules: dict,local_temp: Callable,source: str) -> tuple[str,dict]:
    """For retrieving and appending necessary code the section depends on"""
    try:
        ## assume it's a function or class ##
        if local_temp.__module__ == source:
            section+="\n"+source_code(local_temp)
        elif local_temp.__module__ not in modules:
            modules[local_temp.__module__]=[local_temp.__name__]
        else:
            modules[local_temp.__module__]+=[local_temp.__name__]
    except:
        ## is it a module ##
        if type(local_temp)==ModuleType and local_temp.__name__ not in modules:
            modules[local_temp.__name__]=[]
        else:
            raise TypeError(f"Variable '{local_temp}' from new_exports is not a Callable or module type")
    return section,modules

def get_attr_exports(attrs: list[str],source: str,callables: list[Callable]) -> tuple[list[str],str,list[Callable],pd.DataFrame]:
    """
    Retrieves: modules, callables, and definitions (e.g. references or things monkey patched) from an attribute chain
    
    Everything in the chain must be made available. If it's a module, function, class, or definition it gets imported
    """
    # go through all the attrs; columns are object,definitions,module_name
    new_exports=pd.DataFrame(search_attr_chain(attr_chain,source) for attr_chain in attrs).drop_duplicates()
    if len(new_exports) > 0: # if there are any
        ## check that this is okay for callables that are just the names ################################################
        allowed_exports,callables=split_list(callables,lambda callable:True if (callable in list(new_exports[0]))==True else False)
        definitions=new_exports[new_exports[0].isin(allowed_exports)][1].dropna().sum()
        # in case pd.Series([]).sum() which returns 0
        definitions="" if type(definitions)!=str else "\n"+definitions
        module_names=new_exports.iloc[:,[0,2]][new_exports.isnull()==False]
        return allowed_exports,definitions,callables,module_names
    return [],"",callables,[]

def configure_export(new_export: list[Callable],obj: Callable,index: int,source: str,attr: str,previous_obj: Callable) -> list[Callable]:
    """For configuring exports in terms of their source and definitions"""
    ## check for different module
    if obj.__module__ != source:
        new_export[2]=obj.__module__
    ## check for referencing or monkey patching
    if obj.__name__ != attr:
        new_export[1]=f"setattr({previous_obj.__name__},{attr},{obj.__name__})\n" if index > 0 and ispatched(previous_obj,attr) else f"{attr}={obj.__name__}\n"
    ## finally, append to the new exports and save the previous object
    return [new_export]

def search_attr_chain(attr_chain: str,source: str) -> list[Callable]:
    """Traverses an attribute chain to uncover where each of the individual attribute came from"""
    new_exports,previous_obj=[],__import__(source)
    for index,attr in enumerate(attr_chain.split(".")): # go up starting with the first one
        previous_obj=copy(obj) # set here because it's possible that we can skip an iteration
        obj=getattr(obj,attr) ## so long as we can retrieve the object then for the most part it should work fine
        ## separate into callables, and modules
        if isinstance(obj,Callable) and not isinstance(obj,BuiltinInstance(object)|MethodType):
            # ! need to figure out how to work this for i.e. functools.wraps and functools.partial etc. ! #
            new_export=[configure_export([function,None,None],obj,index,source,attr,previous_obj) for function in unwrap(obj)]
        elif isinstance(obj,ModuleType):
            new_export=configure_export([obj,None,None],obj,index,source,attr,previous_obj)
        else: ## must be a variable
            new_export=configure_export([obj,None,None],obj,index,source,attr,previous_obj)
            if new_export[1]==None and new_export[2]==None: continue
        new_exports+=new_export
    return new_exports

def split_list(reference: list[str],condition: Callable) -> tuple[list,list]:
    """For splitting one list into two based on a condition function"""
    new,remaining=[],[]
    for item in reference:
        try:
            if condition(item): new+=[item]
            else: remaining+=[item]
        except:
            pass
    return new,remaining

def all_callables(module: str,return_module: bool=False) -> list[str] | tuple[list[str],str]:
    """Returns a list of all callables available in a module"""
    try:
        source=__import__(module)
    except:
        try:
            current=os.getcwd()
            module=slice_occ(module.split(".")[0][::-1],"\\")
            module=[i[::-1] for i in module]
            os.chdir(module[1])
            source=__import__(module[0])
            os.chdir(current)
        except Exception as e:
            os.chdir(current)
            raise e
    callables=[temp for attr in dir(source) if isinstance((temp:=getattr(source,attr)),Callable) or (return_module and isinstance(temp,ModuleType))]
    return (callables,module) if return_module else callables

def side_display(dfs:pd.DataFrame | list[pd.DataFrame], captions: str | list=[], spacing: int=0) -> None:
    """
    # code reference: Lysakowski, R., Aristide, (2021) https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side,CC BY-SA 4.0
    # changes made: Added flexibility to user input and exception handling, made minor adjustments to type annotations and code style preferences, implemented my comments
    Display pd.DataFrames side by side
    
    dfs: list of pandas.DataFrame
    captions: list of table captions
    spacing: int number of spaces
    """
    # for flexibility and exception handling
    dfs,captions=as_list(dfs),as_list(captions)
    length_captions,length_dfs=len(captions),len(dfs)
    if length_captions > length_dfs: raise Exception(f"The number of catpions '{length_captions}' exceeds the number of data frames '{length_dfs}'")
    elif length_captions < length_dfs: captions+=[""]*(length_dfs-length_captions)
    # create tables
    tables=""
    for caption,df in zip(captions, dfs):
        tables+=df.style.set_table_attributes("style='display:inline'")\
        .set_caption(caption)._repr_html_()+spacing * "\xa0"
    display(HTML(tables))

def git_info(repo: str) -> dict:
    """General info from a github repo"""
    api_end_point=re.sub("https://github.com","https://api.github.com/repos",repo)
    return scrape(api_end_point,form="json")

def key_slice(ls: list | dict,slce: slice) -> slice:
    """Converts letter based slicing to numeric based"""
    if type(slce.step) == str:
        raise Exception("slicing step cannot be a string")
    start,stop=slce.start,slce.stop # these will be strings
    # add one to stop to be inclusive for string slicing
    flag=isinstance(stop,str)
    for index,key in enumerate(ls):
        if start==key and type(start) != int: start=index
        if stop==key and type(stop) != int: stop=index
        if type(start) != str and type(stop) != str: break
    # check for errors
    if isinstance(start,str) or isinstance(stop,str):
        # try to be helpful with the error message
        if isinstance(start,str) and isinstance(stop,str):
            raise KeyError("key_slice failed: Both start and end slice arguements are not in the dictionaries key values")
        for which,value in [("Starting",start),("Ending",stop)]:
            if isinstance(value,str):
                raise KeyError(f"in function key_slice: {which} slice '{value}' is not in the dictionaries key values")
    if flag and stop > 0: stop+=1
    return slice(start,stop,slce.step)

class dct_ext:
    """
    Extension to the dict class whereby you can now slice and assign similarly to 
    how you would with strings list or pd.DataFrame objects etc.
    i.e.
    test={"a":3,"b":2}
    test=dct_ext(test)
    test["a","b"]=4,5
    print(test)
    # should print
    # {'a': 4, 'b': 5}
    # you can also slice using ["a":"b"] etc.
    # it also allows multiple assignments i.e.
    from pylab import rcParams
    dct_ext(rcParams)['figure.figsize','font.size','figure.dpi','lines.linewidth','axes.facecolor','patch.edgecolor','font.family']=\
    (0,2.7),3.1,58,42,'colour1','colour2','style1'
    print(rcParams['figure.figsize'])
    # should print
    [0.0, 2.7]
    # for multiple assignments you can use
    dct={"cat":3,"dog":5,"mouse":7}
    *dct_ext(dct),*dct_ext(dct)=1,2,3,4
    """
    def __init__(self,dct: dict) -> None: self.dct=dct
    def __repr__(self) -> str: return str(self.dct)
    def __len__(self) -> int: return len(self.dct)
    def __getitem__(self,index: int|list|tuple|slice) -> dict:
        if isinstance(index,int):
            temp_dct=list(self.dct.items())[index]
            return dict.fromkeys([temp_dct[0]],temp_dct[1])
        if isinstance(index,str): return {index:self.dct[index]}
        if type(index) == slice:
            # get keys as ints if not already
            if type(index.start) == str or type(index.stop) == str:
                index=key_slice(self.dct,index)
            elif type(index.start) == float or type(index.stop) == float or type(index.step) == float:
                raise TypeError("slices must be of the the type slice[int|None,int|None,int|None]")
            return dict(itemgetter(index)(self.items))
        if isinstance(index,list|tuple):
            index=tuple(index[0]) if isinstance(index[0],list|tuple) else tuple(index)
            keys=self.keys
            condition=lambda i:i if type(i)==str else keys[i]
            return {condition(i): self.dct[condition(i)] for i in unique(index)}
        raise TypeError("indexes or slices can only be of type int,list,tuple,or slice[int|None,int|None,int|None]")
    
    def __setitem__(self,index,args) -> None:
        if not isinstance(args,list|tuple): args=[args]
        # make sure keys exist
        for key in index:
            if key not in self.dct:
                self.dct[key]=None
        # get keys and set them
        dct=self.__getitem__(index)
        keys=list(dct.keys())
        # catch errors
        if len(keys)-len(args)!=0: raise Exception(f"in dct_ext.__setitem__: Mismatch between number of keys to set and arguements to be assigned\nkeys: {keys}\nargs: {args}")
        for key,arg in zip(keys,args): self.dct[key]=arg

    def __delitem__(self,index: int|str|slice) -> None:
        if type(index)==slice: index=self.__getitem__(index).keys()
        for i in index: del self.dct[i]    
    @property
    def keys(self) -> list: return list(self.dct.keys())
    @property
    def values(self) -> list: return list(self.dct.values())
    @property
    def items(self) -> list: return list(self.dct.items())

def digit_format(number: str | int | float) -> str:
    """Formats numbers using commas"""
    number=str(number)
    parts=number.split(".")
    number="."+parts[1] if len(parts) > 1 else ""
    for count,i in enumerate(parts[0][::-1]):
        number=i+","+number if abs(count) % 3 == 0 else i+number
    return number[:-1]

def file_size(filename: str,decimals: int=2) -> str:
    """Returns the file size formatted according to its size"""
    unit,size=" KMGT",os.path.getsize(filename) # in bytes
    power=len(str(size))-1
    with open(filename) as file:
        return f"file size: {size/10**(int(power/3)*3):.{decimals}f} {unit[int(power/3)].strip()}B\nlines: {digit_format(sum(1 for _ in file))}"

class Timer:
    """
    How to use
    from time import sleep
    timer=Timer()
    sleep(2)
    timer.get
    # should print approximately
    # 2.0
    # if instead you replaced timer.get with:
    timer.get_format
    # then it should print
    # 0.0 hour/s 0.0 minute/s 2.0 second/s
    if instead .log is used you can set the threshold however you
    like to log only the important events
    """
    def __init__(self,important: int | float=0) -> None:
        self.time=time()
        self.important=important

    def reset(self,important: int | float=0) -> None:
        """for resetting the timer and allowing adjusting the threshold of importance"""
        self.__init__(important)
    @property
    def get(self) -> None:
        """Displays the time difference in seconds"""
        current=time()
        print(current-self.time)
        self.time=current
    @property
    def get_format(self) -> None:
        """Displays the time difference formatted"""
        current=time()
        print(time_formatted(current-self.time))
        self.time=current
    @property
    def log(self) -> None:
        """for logging any time differences of importance i.e. checking for bottlenecks"""
        current=time()
        diff=current-self.time
        if diff > self.important: print(diff)
        self.time=current

def remove_docstrings(section: str,round_number: int=0) -> str:
    """For removing multiline strings; sometimes used as docstrings (only raw strings)"""
    avoid='"' if round_number%2==1 else "'"
    new_section,in_string,in_comment,count="",False,False,0
    for char in section:
        prev=(count,)
        if char==avoid and in_comment==False:
            count+=1
            if count==3: in_string,count=(in_string+1)%2,0
        elif char=="#" and in_string==False: in_comment=True
        elif char=="\n" and in_comment==True: prev,count,in_comment=(0,),0,False # it should be 0
        if in_string==False: new_section+=char
        if prev[0]-count==0: count=0
    new_section=new_section.replace(avoid*3," str ")# the start of the string will still be there
    return new_section if round_number==1 else remove_docstrings(*(new_section,round_number+1))

def remove_strings(section: str) -> str:
    """For removing strings (only on raw strings)"""
    new_section,in_string,in_comment,prev_char="",False,False,None
    for char in section:
        if (char=="'" or char=='"') and in_comment==False:
            if prev_char==None: in_string,prev_char=(in_string+1)%2,(char,)
            elif char==prev_char[0]: in_string,prev_char=(in_string+1)%2,None
        elif char=="#" and in_string==False: in_comment=True
        elif char=="\n" and in_comment==True: in_comment=False
        if in_string==False: new_section+=char
    ## remove the various types of string (since the starting piece is still there) ##
    return re.sub(r"r\"|r'|b\"|b'|f\"|f'|\"|'"," str ",new_section)

def get_variables(code: str) -> set[str]:
    """
    Extract only variable names from strings
    
    Note: only works on raw strings e.g.
    the kind you may get from reading files
    else it won't remove strings correctly
    """
    sub=extract_code(code=code) # returns an object
    ## keep float types
    sub(r"[-+]?\d+\.\d+\."," float.")
    sub(r"\([-+]?\d+\.\d+\)\."," float.")
    ## keep int types (they cannot be i.e. 1.to_bytes() only (1).to_bytes() else it expects a float)
    sub(r"\([-+]?\d+\)\."," int.")
    # get letters and numbers only (retaining '.' to extract the base dictionary)
    sub(r"[^\w.]+|\d+"," ")
    # remove any spaces between attributes
    #r"\s+\.|\.\s+"
    sub(r"\s+\.",".")
    sub(r"\.\s+",".") # for some reason splitting them up rather than using | works
    ## check for errors
    matches=re.findall(r"\W[-+]?\d+\.\D",sub.get)
    if len(matches) > 0:
        raise SyntaxError(f"""The following syntaxes are not allowed as they will not execute: {matches}

Cannot have i.e. 1.method() but you can have (1).method() e.g. for int types""")
    # get unique names
    variables=set(sub.get.split(" "))
    # filter to identifier and non keywords only with builtins removed
    builtins=get_builtins()
    return {i for i in variables if (i.isidentifier() == True and iskeyword(i) == False and i not in builtins) or "." in i}

def str_ascii(obj: str | list[int]) -> list[int] | str:
    """
    for converting a string to list of ascii or list of 
    ints to string according to ascii convention
    i.e. try the following to see common non-escape ascii 
    characters:
    
    ls=[]
    for i in range(32,127):
        ls+=[i]
    str_ascii(ls)

    or 

    ls=[]
    for j in [[48,58],[65,91],[97,123]]:
        for i in range(j[0],j[1]):
            ls+=[i]
        ls+=[32]
    str_ascii(ls)
    """
    return list(obj.encode("ascii")) if isinstance(obj,str) else bytes(obj).decode()

## need to update this function eventually given the source codes can be found else where
SOURCE_CODES={}
def undecorate(FUNC: Callable,keep: Callable|list[Callable]=[],inplace: bool=False,key: str|None="") -> None|Callable:
    """
    Redefines functions removing all decorators except those specified to keep
    
    Ensure that the order of keep is the order of decorators you want i.e. first
    to last corresponds to top to bottom as it would be for when defining decorators
    
    Note that 'keep' just redefines the decorators (and order they're in), so you can 
    add new decorators e.g. decorate a function dynamically if desired
    
    Additionally the original and new source code get recordered and are accesible via 
    the SOURCE_CODES global variable
    """
    global SOURCE_CODES
    keep=as_list(keep)
    # get source code split into parts
    decorators,head,doc_string,body=source_code(FUNC,False)
    head_body=head+doc_string+body
    source=decorators+head_body
    # has decorators
    if len(decorators) > 0: # should work? can also do > 2
        # apply keep e.g. head_body is the new code
        if len(keep) > 0:
            # decorators to keep
            head_body="@"+"\n@".join([func.__name__ for func in keep])+"\n"+head_body
        # record code
        if FUNC.__name__ not in SOURCE_CODES: SOURCE_CODES[FUNC.__name__]={"original":source}
        set_source=SOURCE_CODES[FUNC.__name__]
        if key == "" or key == None: set_source["new"]=head_body
        else: set_source[key]=head_body
        # make sure the functions are defined in local scope
        for func in keep: locals()[func.__name__]=func
        # redefine function
        if inplace==True: return exec(head_body,globals())
        exec(head_body)
        return locals()[FUNC.__name__]
    return FUNC

def user_yield_wrapper(FUNC: Callable) -> Callable: # test
    """wrapper for the user_yield function"""
    @wraps(FUNC) # only needed for the function taken in not the args
    def wrapper(func: Callable,*args,**kwargs) -> None: # *desired function*
        return user_yield(FUNC(func)(*args,**kwargs))
    return wrapper

def user_yield(gen: iter) -> None:
    """For user interactive yielding"""
    clear_display=clear_output if has_IPython() else lambda : print("\033[H\033[J",end="")
    while True:
        try:gen_locals=next(gen)
        except StopIteration:break
        user_input=input(": ")
        if user_input.lower() == "break": break
        elif user_input.strip() == "cls": clear_display()
        elif user_input[:8] == "locals()":
            while True:
                if user_input[:8] == "locals()":
                    if len(user_input) < 9: print(gen_locals)
                    elif user_input[8]+user_input[-1] == "[]":
                        try:
                            exec("temp=gen_locals"+user_input[8:])
                            print(locals()["temp"])
                        except: pass
                user_input=input(": ")
                if user_input.lower() == "break": break
                elif user_input.strip() == "cls": clear_display()

def slice_occ(string: str,occurance: str,times: int=1) -> str|tuple[str,str]:
    """
    slices a string at an occurance after a specified number of times occuring
    """
    count=0
    for index,char in enumerate(string):
        if char == occurance:
            count+=1
            if count == times: return string[:index],string[index:]
    return string

# seems to be a problem when running i.e. test(undecorate,do,keep=override_do) #
@user_yield_wrapper
def test(FUNC: Callable,*args,**kwargs) -> Callable:
    """
    redefines a function for printing and yield statements 
    at every line allowing testability
    i.e.
    def do(a,b):
        while True:
            b=a
            a+=1
    test(do,1,2) # or test(do,a=1,b=2)
    additionally allows access to local scope variables as well 
    i.e. by entering 'locals()' and 'locals()["a"]' into the input prompt
    
    enter 'break' to break out of the variable access while loop or the program
    (depending on which loop you're in)

    use 'cls' to clear the entire previous output display

    Note: This function will not work if there are multiline string expressions
    but is something I'll work on soon
    """
    head,doc_string,body=source_code(FUNC,False)[1:]
    lines=[]
    body_lines=body[:-1].split("\n    ")[1:]
    length=len(body_lines)
    printing=lambda code:f"display(Code('{code}', language='python'))" if has_IPython() else lambda code:f"print('{code}')"
    for indx,line in enumerate(body_lines):
        # ensure lines are indented accordingly
        indentation=get_indents(body_lines[indx+1]) if indx < length-1 else get_indents(line)
        # in case of a return statement (add white space incase of 'return' by itself)
        line_stripped=line.strip()
        if line_stripped == "return" or line_stripped.startswith("return "):
            lines+=[indentation+printing(f"line {indx+1}: {line}"),indentation+"yield locals()",line]
            continue
        lines+=[line,indentation+printing(f"line {indx+1}: {line}"),indentation+"yield locals()"]
    body="\n    "+doc_string+"\n    ".join(lines)+"\n"
    exec(head+body)
    return locals()[FUNC.__name__] # call it as you would with inputs if any #

def unstr_df(string: str) -> pd.DataFrame:
    """Convert string to pandas dataFrame"""
    return pd.read_fwf(StringIO(string)).drop("Unnamed: 0",axis=1)

def display_obj(file: str,height: float|int=500,width: float|int=500,type: str="") -> None:
    """For displaying html objects"""
    render(f"""<object height="{height}" width="{width}" type="{type}" data="{file}">
    Filename Error: Make sure the name of the file is correct
</object>""")

def run_r_script(df: pd.DataFrame|pd.Series,script: str="") -> pd.DataFrame:
    """
    For sending and recieving structured data from python to R to python
    e.g. as pd.DataFrame => data.frame => pd.DataFrame
    
    Allowing data manipulation in R via your own custom script
    """
    # allow pd.Series objects and remove newlines
    temp=pd.DataFrame(df).to_string().replace("\n","\\n") ## need to check for larger data ##
    # Give R the dataframe
    read_pd_DataFrame=f"df=read.table(text = '{temp}',check.names=FALSE)"
    # store R data into a dataframe for conversion to string data
    send_string="""library(stringr)
to_list=function(item){
  return(paste("[",toString(item) |> str_sub(3, -2),"]"))
}
to_lists=function(df){
  cols=c(df)
  for(i in df |> colnames()){
    cols[i]=cols[i] |> to_list()
  }
  return(cols)
}
convert_to_csv=function(df){
  header=paste("[",toString(paste("\\"",df |> colnames(),"\\"")),"]")
  items=df |> to_lists() |> data.frame() |> toString()
  return(paste(header,",",items,collapse = ','))
}
df |> convert_to_csv()
"""
    string_data=run_r_command(read_pd_DataFrame+script+send_string).split("\r\n")[-1][4:-1]
    # we should be able to eval everything and return as pd.DataFrame
    tup=eval(eval(string_data))
    return pd.DataFrame(list(tup[1:]),index=tup[0]).T

def clear_line(line: int=-1) -> None:
    """
    For clearing lines in jupyter notebook output areas
    Make sure ipynb_id_setup() has been ran first since its required and it 
    will only be active on a new mutation of a notebook cell not during.
    """
    # get output_areas, remove select line
    get_executing_cell(f"""var cell_stdout=runnin.parentElement.parentElement.parentElement.childNodes[1].childNodes[1].childNodes;
// remove the selected line in the output area
Array.from(cell_stdout).at({line-1}).remove();
// remove the html created
Array.from(cell_stdout).at({line}).remove();
""")

def get_executing_cell(appending_script: str="console.log(cell_id);") -> None:
    """"
    For retrieving the currently executing cells id in jupyter notebook for enabling cell manipulation
    
    Make sure ipynb_id_setup() has been ran first since its required and it 
    will only be active on a new mutation of a notebook cell not during.
    
    // to get cell id
    var cell_id = runnin.parentElement.parentElement.parentElement
    cell_id = parseInt(cell_id.getAttribute("cell_id"))
    Jupyter.notebook.get_cell(cell_id)
    """
    display(HTML("""<script id="get_executing_cell">
// get the elements
var runnin=document.querySelectorAll("[exec_id],[exec_status]")
// filter to those that are executing
runnin=Array.from(runnin).filter(el => el.getAttribute('exec_status') === 'Executing');
// get the element with the lowest exec_id
runnin = runnin.reduce((min, current) => {
  let current_exec_id = parseInt(current.getAttribute('exec_id'));
  let min_exec_id = parseInt(min.getAttribute('exec_id'));
  return current_exec_id < min_exec_id ? current : min;
});
"""+appending_script+"""
</script>"""))

def ipynb_id_setup(reload: bool=False) -> None:
    """For setting up cell_id and exec_id/exec_status for all code cells"""
    if reload==True: return refresh()
    generate_cell_ids()
    generate_exec_ids()

def refresh() -> None:
    """Refreshes jupyter notebook; it will save the current page as well"""
    display(Javascript("Jupyter.notebook.save_checkpoint();window.onbeforeunload=null;location.reload();"))

def dynamic_js_wrapper(func: Callable[...,bool]) -> Callable:
    """wrapper function for dynamically created javascript functions to save code"""
    @wraps(func)
    def wrapper(reload: bool=False) -> None:
        if reload: return refresh()
        name=str(func).split()[1]
        script,call=func(reload)
        check=f"""{script}
if(document.querySelectorAll("[id={name}]").length == 1){{
    {call}
}}else{{
    console.log('{name} already exists')
}}"""
        display(HTML(f"<script id='{name}'>{check}</script>"))
    return wrapper

@dynamic_js_wrapper
def generate_exec_ids(reload: bool=False) -> tuple[str,str]:
    """
    For generating execution order ids and status in jupyter notebook
    Note: if it causes crashes then it's because the kernel was restarted
    but the page was not refreshed, hence the script will get appended
    
    to retrieve use: document.querySelectorAll("[exec_id],[exec_status]")
    """
    script="""var exec_id=0;
const observer = new MutationObserver((mutations) => {
    for(const mutation of mutations){
        let element=mutation.target
        if(element.className == "prompt input_prompt"){
            // set id and status on execution then change status on finish
            if((element.hasAttribute("exec_id") == false) || (element.innerHTML == '<bdi>In</bdi>&nbsp;[*]:')){
                if(element.getAttribute("exec_status") == "Executing"){continue;}
                // the mutation observer will likely pick up these changes 
                // therefore we disconnect the observer while changes occur
                element=ignore_set(element);
                console.log('exec_id ',element.getAttribute('exec_id'),' status:  Executing');
                exec_id+=1;
                if(element.innerHTML != '<bdi>In</bdi>&nbsp;[*]:'){
                    element=ignore_set(element,true);
                    console.log('exec_id ',element.getAttribute('exec_id'),' status:  Done');
                }
            }else if(element.getAttribute("exec_status") == "Executing"){
                element=ignore_set(element,true);
                console.log('exec_id ',element.getAttribute('exec_id'),' status:  Done');
            }
        }
    }
});
function ignore_set(el,done=false){
    observer.disconnect();
    // make necessary changes while disconnected
    if(done==true){
        el.setAttribute("exec_status","Done")
    }else{
        el.setAttribute("exec_status","Executing")
        el.setAttribute("exec_id",exec_id)
    }
    start_obs(el)
    return el
}
function start_obs(){
    observer.observe(document.getElementById("notebook-container"), {
        childList: true,
        subtree: true
    });
}"""
    call="start_obs()"
    return script,call

@dynamic_js_wrapper
def generate_cell_ids(reload: bool=False) -> tuple[str,str]:
    """
    For generating execution order ids and status in jupyter notebook
    
    to retrieve use: document.querySelectorAll('[code_cell_id]')
    
    For use via the jupyter notebook api use: Jupyter.notebook.get_cell(cell_id)
    """
    script="""var number_of_cells=0;
function update_cell_id(){
    let cells=$(".cell")
    let length=cells.length
    if (length != number_of_cells){
        // renumber all cell ids so that they are in index order
        number_of_cells=length
        for (let i = 0; i < number_of_cells; i++){
            cells[i].setAttribute("cell_id", i)
        }
    }
}"""
    call="setInterval(update_cell_id, 100);"
    return script,call

def check_js(file: str) -> str:
    if len(file.split(".")[0]) == 0: raise Exception(f"Invalid filename: '{file}'")
    return file[:-3] if file[-3:] == ".js" else file

def import_js(file: str,id: str="") -> None:
    """
    For importing javascript files while avoiding duplicating from appending scripts
    To remove them use refresh()
    """
    file=check_js(file)
    load=f"""if (document.getElementById('{file+id}') == null){{
    const script = document.createElement('script');
    script.id = '{file+id}';
    script.src = '{file}.js';
    document.body.appendChild(script);
    console.log('{file}.js loaded');
}}else{{console.log('{file}.js is already loaded');}}
"""
    display(Javascript(load))

def remove_html_el(id: str="") -> None:
    """
    For removing html elements by id 
    """
    display(Javascript(f"document.getElementById('{id}').remove();"))
    print(f"removed element: {id}")

def has_IPython() -> bool:
    """Checks for IPython"""
    return get_ipython() != None

def list_loop(ls: Any,FUNC: Callable=lambda x:x) -> list[Any]|Any:
    """
    loops through a list of elements applying some function to each element
    """
    ls=as_list(ls)
    return FUNC(ls[0]) if len(ls) == 1 else [FUNC(item) for item in ls]

def in_valid(prompt: str,check: Callable,clear: bool=False) -> str:
    """
    Input validation assurance
    """
    print(end="\r")
#    if has_IPython():
#        line_clear=clear_line
#    else:
#        line_clear=(lambda:print("\033[A\033[J\033[A")) # partial(print,"\033[A\033[J\033[A") is also possible
    while True:
        response=input(prompt)
        if check(response):break
        # need to see if ipynb_ids_setup will run within a function before this can work
        #line_clear()
    if clear==False: display(prompt+response)
    return response

class input_ext:
    """extension to the input function for inputting and/or validating more than one prompt"""
    def __init__(self,prompts: Any="") -> None: self.prompts=prompts
    def loop(self) -> list[Any]|Any: return list_loop(self.prompts,input)
    def check(self,check: Callable=lambda x: 1,clear: bool=False) -> list[Any]|Any:
        """For validating inputs"""
        return list_loop(self.prompts,partial(in_valid,**{"check":check,"clear":clear}))

def check_and_get(packages: list[str]=[],full_consent: bool=False) -> None:
    """Gets packages that are currently not installed"""
    # first check the builtins then the pypi libraries
    standard_library = sys.stdlib_module_names
    pypi = all_packs()["Package"]
    for package in packages:
        # check standard library
        if len([i for i in standard_library if i == package]) == 1:
            print(package+" is in the standard library")
            continue
        # check current pypi
        if len(pypi[pypi == package]) == 1:
            print(package+" is in the pip local python library")
            continue
        # if not in these then
        if full_consent == False:
            ## ask for consent ##
            response=input_ext(package+" not installed. Do you want to proceed to install?: y/n --")\
            .check(lambda response: 1 if response == "y" or response == "n" else 0)
            if response == "n": continue
        process=subprocess.run("pip install "+package,capture_output=True)
        if process.returncode != 0:
            print("Error with package: "+package+"\n\n")
            print(process.stdout.decode("utf-8"))
            print("-"*20)
        else: print(package+" successfully installed\n")

def all_packs() -> pd.DataFrame:
    """Retrieves all packages and returns as a pd.DataFrame"""
    ls=subprocess.run("pip list -v",capture_output=True).stdout.decode("utf-8")
    ls=ls.split("\r\n")[2:-1] ## -1 because of the split
    ls=[new_pack if len((new_pack:=pack.split())) > 5 else new_pack[:2]+["None"]+new_pack[2:] for pack in ls]
    return pd.DataFrame(ls, columns=['Package', 'Version', 'Editable project location', 'Location', 'Installer'])

def to_module(code: str) -> Callable[..., Any]:
    """
    converts a string to a python module object that associates with a temporary file for getting source code
    # code reference: Sottile, A., (2020) https://stackoverflow.com/questions/64925104/inspect-getsource-from-a-function-defined-in-a-string-s-def-f-return-5,CC BY-SA 4.0
    # changes made: combined into one function, used while loop to ensure uniqueness of filename, shortened code, implemented my comments
    """
    class Module:
        def __init__(self, module_name: str, source: str) -> None:
            self.module_name = module_name
            self.source = source
        # there must be a get_source method used
        # in the getsource function
        def get_source(self, module_name: str) -> str:
            if module_name != self.module_name:
                raise ImportError(module_name)
            return self.source
    # create temporary file and unique module name
    while True:
        temp = tempfile.mktemp(suffix='.py')
        module_name = temp.split(".")[0].split("\\")[-1]
        if module_name not in sys.modules:break
    # create module with specialized loader e.g. to override the get_source method
    module = module_from_spec(spec_from_loader(module_name, Module(module_name, code), origin=temp))
    exec(compile(code, temp,"exec"), module.__dict__)
    # needs to be added to sys modules else it doesn't register
    sys.modules[module_name] = module
    return module

def inherit(cls: type,*bases: tuple[type]) -> type:
    """Adds inheritence to a choosen classname. This works for nested classes as well"""
    return type(cls.__name__,bases,cls.__dict__)

def req_file(directory: str="") -> None:
    """
    writes a requirements .txt file for .py and .ipynb files with modules version 
       
    version numbers are only reasonable when the project developers use this function
    and they are the versions used globally on their machines, conda environments need
    implementation.
    """
    required = ""
    for module in req_search(directory):
        required+=module+": "
        try:
            required+=__import__(module).__version__+"\n"
        except:
            required+="\n"
    print("\nwriting modules to requirements.txt...")
    with open("requirements.txt","w") as file: file.write(required)
    print("done")

def read_ipynb(filename: str,accepted_types: str|list[str]="code",join: bool=False) -> list[str]|str:
    """readlines a jupyter notebook"""
    if filename[-6:] != ".ipynb": filename+=".ipynb"
    with open(filename, "rb") as file: lines = json.load(file)
    ls=[[line[:-1] for line in lines[:-1]]+[lines[-1]] if len((lines:=cell["source"])) else [""] 
    for cell in lines["cells"] if cell["cell_type"] in accepted_types]
    return "\n".join(ls) if join else ls

def get_requirements(filename: str,unique: bool=True) -> list[str]:
    """Reads a .py or .ipynb file and tells you what requirements are used (ideally)"""
    if filename.split(".")[1] == "ipynb":
        filtered = pd.Series(read_ipynb(filename))
    else:
        # .py file; we need to read as bytes due to an error that sometimes occurs
        with open(filename, "rb") as file:
            filtered = pd.Series(file.readlines()).apply(lambda x:x.decode("utf-8"))
    # get the libraries, remove comments if any, and remove the first word (will be an import or from statement)
    regex=r"^(?:from|import)\s+.*?"
    filtered = filtered[filtered.str.contains(regex)].str.split("#").str[0].apply(lambda x:re.sub(regex,"",x))
    # perform the filters
    ## remove " as ...,", and as ...
    filtered = filtered.apply(lambda x:re.sub(r" as .*?,",",",x)).apply(lambda x:re.sub(r" as .*?$",",",x))
    ## remove secondary " import ...", remove the methods to get just the name of the module, and remove the newline characters
    filtered = filtered.apply(lambda x:re.sub(r" import .*","",x)).str.split(".").str[0].str.split("\n").str[0].str.split("\r").str[0]
    # for imports on the same line, to remove whitespace, and get only the unique imports
    filtered = filtered.str.split(",").explode().str.strip()
    filtered = filtered[filtered.str.len() > 0]
    if unique==True:
        filtered = pd.Series(filtered.unique())
    # filter out standard libraries
    filtered = filtered[filtered.isin(list(sys.stdlib_module_names)) == False]
    # return as a list
    return list(filtered)

def req_search(directory: str="",allowed_extensions: list[str]=["py","ipynb"]) -> list[str]:
    """
    Searches a directory and all its' subdirectories
    for .py or .ipynb files that are then checked for their requirements
    """
    requirements = []
    target = glob.glob(directory+"*")
    print("searching files and subdirectories for required python modules...")
    while target != []:
        next_target=[]
        for i in target:
            print(i)
            file = i[1:].split(".")
            # folder
            if len(file) == 1:
                next_target+=glob.glob(i+"/*")
                continue
            # .py file
            elif file[1] in allowed_extensions:
                requirements+=get_requirements(i)
        target = next_target
    # return only the unique requirements
    return list(set(requirements))
        
def install(library_name: str,directory: str="",setup: bool=True,get_requirements: bool=True,defaults: bool=True,build: bool=False,build_type: str="wheel",default_config: str="PYTHON_SETUP_DEFAULTS.pkl") -> None:
    """
    Creates necessary files for a library that can be accessed globally 
    on your local machine created in the same directory as the script 
    choosen to become a library.
    Make sure you create have the directory before installing.
    
    if you've uninstalled but kept the files and want to install again
    set setup = False which will go directly to the directory and run
    !pip install *library_name*

    library_name: the name of your new module to access (in anaconda potentially if using jupyter notebook)
    directory: directory where your python script is
    default_config: a .pkl file that's a pd.Series with your default information i.e.:
    pd.Series({"version":"'0.1.0'","description":"'contains useful function'","author":"'author'","email":"'email'"}).to_pickle("PYTHON_SETUP_DEFAULTS.pkl")
    build: 
      -  if False will install an editable library - pro: can be edited and reloaded quickly. - con: can be slower to load if contains lots of code
      -  if True will build a .whl file            - pro: can load the file quickly.          - con: requires reinstallation to rebuild it
    build_type: "wheel" or "sdist"
    """
    if setup:
        print("---setup configs---")
        if defaults:
            # get the default setup directory correct
            default_config = os.path.join(module_dir,default_config)
            configs = dict(pd.read_pickle(default_config))
            configs["description"] = "'"+input("description: ")+"'"
        else:
            configs = {"version":"","description":"","author":"","email":""}
            for i in configs.keys(): configs[i] = "'"+input(i+": ")+"'"
        requirements = []
        # search for requirements
        if get_requirements: requirements = req_search(directory)
        setup_content = """from setuptools import setup, find_packages

setup(
    name="""+"'"+library_name+"'"+""",  
    version="""+configs["version"]+""",
    description="""+configs["description"]+""",
    author="""+configs["author"]+""",
    author_email="""+configs["email"]+""",
    packages=find_packages(),
    install_requires="""+str(requirements)+"""
)
"""
        # create __init__.py and setup.py
        if build==False: # might consider an alternative for builds for subdirectories as well
            with open(directory+"__init__.py","w"):None
        with open(directory+"setup.py","w") as file: file.write(setup_content)
        print("Successfully created setup files __init__.py and setup.py.")
        print("\ninstalling "+library_name+"...")
        # go to the directory then run the command
        with Path(directory):
            if build:
                ## needs testing ##
                if setup:
                    ## put all files into a directory named after the modules name ##
                    source=os.getcwd()
                    files=os.listdir(source)
                    files.remove("setup.py")
                    os.mkdir(library_name)
                    for file in files:
                        shutil.move(os.path.join(source,file), library_name)
                while build_type!="wheel" and build_type!="sdist":
                    build_type=input("build_type must be either 'wheel' or 'sdist'")
                build_type=["bdist_wheel","whl"] if build_type=="wheel" else ["sdist","tar.gz"]
                process=subprocess.run("python setup.py "+build_type[0],capture_output=True)
                if process.returncode == 0:
                    os.chdir("dist")
                    build_name=glob.glob('*.'+build_type[1])[0]
                    process=subprocess.run("pip install "+build_name,capture_output=True)
            else:
                process=subprocess.run("pip install -e .",capture_output=True)
            if process.returncode != 0:
                print("---installation failed---")
                print("note: some imported packages may use dummy names")            
                print(process.stdout.decode('utf-8'))
            else:
                # wait for it to finish...
                print("\n"+library_name+" successfully installed at: "+os.getcwd())
                print("\nYou can now access the module where applicable by using: from "+library_name+" import *desired function*")
                print("or: import "+library_name)
                print("You may need to restart the kernel to use or uninstall")
                print("To uninstall whilst retaining the pre-installation files run: !pip uninstall "+library_name)

def uninstall(library_name: str,keep_setup: bool=True) -> None:
    """
    Runs pip uninstall library_name 
    with the option of removing the
    setup files stuff as well.
    
    also note that with keep_setup = False
    the __init__.py,setup.py, files and .egg-info 
    directory will be removed
    """
    # get the libraries directory (likely needs to be done first since uninstall may remove the path) - needs testing
    directory = module_file(library_name)
    directory=os.path.dirname(directory)
    print("uninstalling "+library_name) if keep_setup else print("uninstalling "+library_name+" and removing setup files")
    # you have to add yes due to no request coming back; else nothing happens
    subprocess.run("pip uninstall "+library_name+" --yes")
    if keep_setup == False:
        # remove the __init__ and setup files
        os.remove(os.path.join(directory,"__init__.py"))
        print("removed __init__.py")
        os.remove(os.path.join(directory,"setup.py"))
        print("removed setup.py")
        shutil.rmtree(os.path.join(directory,".egg-info"))
        print("removed "+library_name+".egg-info")
    print("\ndone")

def git_clone(url: list[str]|str=[],directory: list[str]|str=[],repo: bool=True) -> None:
    """For cloning multiple github repos or files into choosen directories"""
    directory,url=as_list(directory),as_list(url)
    dir_length,url_length=len(directory),len(url)
    if dir_length == 1: directory=directory*url_length
    elif url_length == 1: url=url*dir_length
    elif url_length != dir_length: raise Exception("Length mismatch between the number of supplied urls: "+str(url_length)+", and directories: "+str(dir_length))
    if repo:
        for file,path in zip(url,directory):
            if path == "":
                process=subprocess.run("git clone "+file,capture_output=True)
            else:
                current=os.getcwd()
                os.chdir(path)
                process=subprocess.run("git clone "+file,capture_output=True)
                os.chdir(current)
            if process.returncode != 0:
                print("Failed retrieving "+file+":\n\n")
                print(process.stdout.decode('utf-8'))
        return
    for file,path in zip(url,directory):
        try:
            open(path+file.split("/")[-1], "wb").write(requests.get(file).content)
        except Exception as e:
            print("Failed retrieving "+file+":\n\n")
            print(e)

def prep(line: str,FUNC: Callable,operator: str) -> str:
    """
    Takes a line splits on binary operator and rewrites with the relevant function
    If wanting all custom code interpretations you may omit all the code here and just
    define your custom functions that will perform all the desired operations and then
    use:
    
    return FUNC.__name__+"("+line_sep(line,operator,sep=",")+")"

    The final formatting will just be many functions applied to a string.
    """
    ################################################################# needs testing
    # make sure there's space and format the i.e. (a=3,b=2)
    #line=pipe_func_dict(line_sep(line,operator,sep=" |> "))
    #################################################################
    # split the = into sections where each section gets interpreted
    assignments=line.split("=")
    for index,assignment in enumerate(assignments):
        if operator in assignment:
            # split on operator and wrap with function
            assignments[index] = FUNC.__name__+"("+line_sep(assignment,operator,sep=",")+")"
    return "=".join(assignments) if len(assignments) > 1 else assignments[0]

def indx_split(indx: list[int]=[],string: str="") -> list[str]:
    """Allows splitting of strings via indices"""
    return [string[start:end] for start,end in zip(indx, indx[1:]+[None])]

def line_sep(string: str,op: str,sep: str="",split: bool=True,index: bool=False,exact_index: bool=False,avoid :str="\"'") -> list[int|str]|str:
    """separates lines in code by op avoiding op in strings"""
    in_string,indx,count,current_str,req,ls=(0,)*3,"",len(op),list(string)
    if sep == "": ls2=[]
    for i in ls:
        if i == op[count]:
            count+=1
        else:
            count=0
        if i in avoid:
            if i == current_str or current_str == "":
                in_string=(in_string+1) % 2
                if current_str !="" and in_string == 0:
                    current_str=""
                else:
                    current_str=i
        elif count == req:
            if in_string==False:
                if sep == "" and split == True:
                    if req > 1:
                        ls2+=[indx-req+1]
                    else:
                        ls2+=[indx]
                    ls2+=[indx+1]
                else:
                    if req > 1:
                        ls[indx-req+1:indx+1]=sep
                    else:
                        ls[indx]=sep
            count=0
        indx+=1
    if exact_index==True:
        return [ls2[2*i] for i in range(len(ls2)//2)]
    if index==True:
        return ls2
    # if needing to slice it
    if sep == "" and split == True:
        if type(string) == list:
            string = "".join(string)
        return indx_split([0]+ls2,string)
    # since it's already formatted
    return "".join(ls)

def get_indents(line: str) -> str:
    """Gets the number of python valid indentations for a line and then scales indentation accordingly"""
    n_white_space=0
    for char in line:
        if char == " ": n_white_space+=1
        else: break
    if n_white_space % 4 != 0: raise Exception("indentations must be 4 spaces to be valid:"+line)
    return " "*n_white_space

def bracket_up(string: str,start :str="(",end: str=")",avoid: str="\"'") -> pd.DataFrame:
    """ For pairing bracket indexes """
    current_str,left,in_string,ls="",[],0,[]
    for index,char in enumerate(string):
        if char in start: left+=[index]
        elif char in end:
            ls+=[[left[-1],index,in_string,len(left)-1]]
            left=left[:-1]
        if char in avoid:
            if char == current_str or current_str == "":
                in_string=(in_string+1) % 2
                current_str="" if current_str !="" and in_string == 0 else char
    return pd.DataFrame(ls,columns=["start","end","in_string","encapsulation_number"])

### needs testing but seems okay ###
def func_dict(string: str) -> str:
    """Turns a i.e. (a=3,b=2) into {"a":3,"b":2} """ ## why not just eval("dict"+string) ?? seems overly complex for no reason; unless the values are python invalid

    def dict_format(temp: int,commas: pd.Series|pd.DataFrame,section: list,old_section_length: int,adjust: int) -> tuple[list,int]:
        """For formatting (a=3) into {"a":3}"""
        comma=commas[commas <= temp].iloc[0] # the min it can be is 0 so... does this matter?
        # getting (no temp + 1 here because we want to exclude the '=')
        temp_string="".join(section[comma+1-adjust:temp-adjust])
        # formatting
        temp_string='"'+re.sub(" ","",temp_string)+'":'
        if comma != 0:
            temp_string=','+temp_string
        # assigning
        section[comma-adjust:temp+1-adjust]=list(temp_string)
        # we need to use adjust because the length of the section keeps changing
        adjust=old_section_length-len(section)
        return section,adjust

    def dict_format_check(start: int,end: int,string_ls: list) -> list:
        """Checks a string and modifies it into a dict format as specified"""
        # get relevant data
        adjust=0
        section=list(string_ls[start:end+1])
        old_section_length=len(section)
        eq=line_sep(section,"=",exact_index=True)
        commas=pd.Series([0]+line_sep(section,",",exact_index=True)).sort_values(ascending=False,ignore_index=True)
        # determine which are single, and format these
        is_double_eq=False
        length=len(eq)-1
        for i in range(length):
            if is_double_eq == True:
                is_double_eq=False
                continue
            elif eq[i+1] - eq[i] > 1:
                # format the string
                section,adjust=dict_format(eq[i],commas,section,old_section_length,start,adjust)
                continue
            is_double_eq=True
        if length == -1:
            return string_ls
        diff=eq[length] - eq[length-1]
        if diff != 1:
            section,adjust=dict_format(eq[length],commas,section,old_section_length,start,adjust)
        # if changes were made then it must be a dict format
        if old_section_length != len(section):
            string_ls[start:end+1]=[" "]+["{"]+section[:-1]+["}"]
        return string_ls

    # get the bracket information
    def get_brackets(ls: list,filter_out: str="") -> pd.Series|pd.DataFrame:
        df = bracket_up("".join(ls))
        if filter_out != "":
            df = df[df["start"] != filter_out]
        try:
            return df[df["in_string"] == 0].sort_values("encapsulation_number",ignore_index=True,ascending=False)
        except: # len(df) == 0
            return df

    # format the string
    string_ls=list(string)
    old_string_length = len(string_ls)
    # because the string length keeps changing we need to keep track of changes
    df=get_brackets("".join(string_ls))
    while len(df) > 0:
        filter_out=""
        previous=string_ls
        string_ls=dict_format_check(df["start"].iloc[0],df["end"].iloc[0],string_ls)
        if previous == string_ls:
            filter_out = df["start"].iloc[0]
        df=get_brackets(string_ls,filter_out)
    return "".join(string_ls)

def pipe_func_dict(string: str) -> str:
    """prep dicts used in piping"""
    string = func_dict(string)
    ls=string.split(" ")
    ls_new=[]
    for i in range(len(ls)):
        temp=ls[i]
        try:
            asterisk=""
            if temp[-1] == "*":
                temp=temp[:-1]
                asterisk="*"
            if hasattr(eval(temp),"__call__") == True:
                temp_check = ls[i+1]
                if temp_check[0] == "{" or temp_check[0:2] == "*{" or temp_check[0:3]+" " == "**{":
                    ls_new+=[temp+","+asterisk]
                    continue
        except:
            None
        # we can't use temp because the except statement will keep the modifications up to the exception
        ls_new+=[ls[i]]
    return "".join(ls_new)

### seems to work but needs testing ###
def interpret(code: str,checks: list[Callable]=[],operators: str=[]) -> str:
    """Checks code against a custom set of rules for how code should be formated"""
    # to allow for multi-line open bracket expressions
    df=bracket_up(code,start="({[",end="]})")
    df=df[(df["encapsulation_number"] == 0) & (df["in_string"] == 0)]
    # remove newline characters in multiline open brackets
    lines=list(code)
    old_length,adjust=len(lines),0
    for start,end in zip(df["start"],df["end"]):
        lines[start-adjust:end-adjust] = line_sep("".join(lines[start-adjust:end-adjust]),"\n",split=False)
        adjust = old_length-len(lines)
    code="".join(lines)
    # make sure they're lists
    checks,operators=as_list(checks),as_list(operators)
    # \n in strings becomes \\n, so you can split this way
    # line_sep takes care of the ';' occurances and comments
    # get lines and remove comments + spaces
    lines=[temp for line in line_sep(code,";","\n").split("\n") if len((temp:=line_sep(line,"#")[0])) > 0]
    # go through each line checking through the operators
    for index,(check,operator) in enumerate(zip(checks,operators)):
        for line in lines:
            if operator in line:
                # get the number of indentations at the start of the line
                lines[index]=get_indents(line)+prep(line,check,operator)
    # join lines back together
    try: return "\n".join(lines)
    except: return lines[0]

def str_anti_join(string1: str,string2: str) -> str:
    """anti_joins strings sequentially e.g. assuming appendment"""
    diff=len(string2) - len(string1)
    if diff != 0:
        string1,string2=list(string1),list(string2)
        # '__' since strings should list into single characters
        string1,string2,temp=(string1+["__"]*diff,(string2,)*2) if diff > 0 else (string1,string2+["__"]*diff,string1)
    for index,(s1,s2) in enumerate(zip(string1,string2)):
        if s1 == s2: temp[index]=""
    return "".join(temp)

## background processing
def create_variable(name: str) -> None: globals()[name] = []

standing_by_count=0
def stand_by() -> None:
    global standing_by_count
    sleep(1)
    print(standing_by_count)
    standing_by_count+=1

def stop_process(ID: int) -> None: globals()["process "+str(ID)]=True

def show_process() -> None:
    global ids_used
    for i in range(ids_used):
        try:print("process "+str(i+1)+" running...")
        except:None

ids_used=0
def background_process(FUNC: Callable=stand_by,*args,**kwargs) -> None:
    def process() -> None:
        # create an id
        global ids_used
        ids_used+=1
        name="process "+str(ids_used)
        create_variable(name)
        globals()[name]=False
        while True:
            FUNC(*args,**kwargs)
            if globals()[name]:
                break
        del globals()[name]
    Thread(target=process).start()

def capture(interpret_code: bool=True) -> None:
    """
    Function to capture the inputs in real time (only works in jupyter notebook because of the 'In' variable)
    The only issues with it is that because it runs on a thread it therefore can only work one cell at a time
    so the display may come out unexpectedly
    """
    global current_execution
    last_execution = get_ipython().__getstate__()["_trait_values"]["execution_count"]
    # add some delay to stop interference with the main thread
    sleep(1)
    # wait until cell has been executed
    if current_execution != last_execution:
        code=get_ipython().__getstate__()["_trait_values"]["parent_header"]["content"]["code"]
        if code != "background_process(capture,True)" and len(code) != len("".join(line_sep(code,"|>"," "))):
            clear_output()
            if interpret_code == True:
                print("interpreting...")
                code=interpret(code,pipe,"|>")
            #clear_output()
            print("post-processor:")
            try:
                code_ls=str(code).split("\n")
                last_line=code_ls[-1]
                code_ls="\n".join(code_ls[:-1])
                exec(code_ls+"\ndisplay("+last_line+")")#error
                print("")
            except:
                try:exec(str(code))
                # a variable must have been assigned
                except:
                    try:display(code)
                    except:None
        # make sure the current execution is updated
        current_execution = get_ipython().__getstate__()["_trait_values"]["execution_count"]

def partition(number_of_subsets: int,interval: int) -> list:
    """
    Creates the desired number of partitions on a range.
    Accepts integers for the number_of_subsets but will 
    only produce reasonable partitions for values within
    the interval (0,interval]
    """
    partitions = [0]+[int(np.floor(i*interval/number_of_subsets)) for i in range(1,number_of_subsets+1)]
    return [[partitions[i],partitions[i+1]] for i in range(len(partitions)-1)]

def thread_runs() -> bool:
    """checks if thread is still running"""
    return os.getenv('FORCE_STOP', True) == "False"

def multi_thread(number_of_threads: int,interval_length: int,FUNC: Callable,part: bool=False) -> None|list:
    """
    My own multi-threading function

    Creates a sliding window of any number of desired threads. 
    [...]........
    .[...].......
    ..[...]......
    ...[...].....
    etc.
    The FUNC arguement should be a function encapsulating 
    one iteration of your desired for loop. The use of global
    variables and threading locks is required for it to work.
    
    If you keyboard interrupt while threads are running you'll 
    need to wait until it reaches its window size else restart
    the kernel; the threads are not joined to the main thread.
    It can be useful to run in this setting for background tasks
    for example e.g. where a keyboard interrupt has no effect on
    the thread running. For example if you set your function as 
    an infinite while loop then even when keyboard interrupted 
    the threads will still run.
    
    Alternatively you can set part=True and get threads on partitions
    which allows synchorunous multi-threading as well as asynchronous 
    via temporarily creating global variables. When part=True it also
    will return a list of the values retrieved if any. (using part=False
    requires global variables, so no results will get returned)
    [....][....][...] => partition(3,11)
    [..][..][..][..][..][.] => partition(6,11)
    etc.
    (each enclosing of [] represents a thread running a for loop on as 
    many iteratios as there are dots enclosed by these brackets).
    
    Note:
    Function returns a tuple as: results,errors
    """
    # in case a thread runs a while loop that has no other way of recieving information
    os.environ["FORCE_STOP"] = "False"
    # setup thread_count
    global thread_count,errors
    def wait() -> None|list:
        """
        used last to wait for threads to finish and retrieve variables.
        
        If you have a function as follows:
        def do(i): # (or def do(variable,i))
            while True:
                continue
        Then it can be used to run background tasks.
        If it needs to be stopped you can replace True with run_threads()
        and this will stop it when the program keyboard interrupts.
        Else you can figure out on your own when it should stop based on
        other things the thread can recieve.
        """
        while thread_count != number_of_threads: continue
        print("\nThreads complete")
        if part == True:
            # retrieve all the temporary variables
            temp = []
            for name in free:
                temp+=globals()[name]
                # removes it from memory
                del globals()[name]
            return temp

    def template(variable: str=[],i: int=[],parts: list=[]) -> None:
        """Used to prevent duplicate code and for ease of use"""
        global thread_count,threads_stopped,errors
        errors=[]
        if part == True:
            for i in range(parts[0],parts[1]):
                if threads_stopped == True:
                    break
                try: FUNC(globals()[variable],i)
                except Exception as e:
                    print("error at i =",i,":",e)
                    errors+=[i]
        else:
            try: FUNC(i)
            except Exception as e:
                print("error at i =",i,":",e)
                errors+=[i]
        with lock: thread_count+=1 # tells the multi-threader that a thread's available

    try:
        if part == True:
            global threads_stopped
            threads_stopped = False
            thread_count = 0
            # create variables
            free = []
            for i in range(number_of_threads):
                name = "temp_"+str(i)
                create_variable(name)
                free.append(name)
            # get partitions
            partitions = partition(number_of_threads,interval_length)
            # setup threads
            for i in range(number_of_threads):
                Thread(target=template,kwargs={'parts':partitions[i],"variable":free[i]}).start()
        else:
            thread_count = number_of_threads
            i=0
            while i in range(interval_length):
                # as soon as a thread's available put it to work
                if thread_count != 0:
                    thread_count-=1
                    Thread(target=template,kwargs={"i":i}).start() # make sure 'i' is an input variable in your function
                    i+=1
        return wait(),errors
    except KeyboardInterrupt:
        # having a lock for this variable is unecessary as the threads do not modify it / doesn't corrupt either?
        threads_stopped = True
        os.environ["FORCE_STOP"] = "True" # in case wanting to be used in program
        return wait(),errors # defined twice in case keyboard interrupt occurs
    # once it's out of the loops that means we're waiting on the threads to finish

def skip(iter_val: iter,n: int) -> None:
    """
    Skips the next n iterations in a for loop
    
    Make sure you've set up an iterable for it to work e.g.
    myList = iter(range(20))
    for item in range(20): # can also be: for item in myList:
        # do stuff
    """
    for _ in range(n): next(iter_val)

def argmiss(returns: dict|tuple,temp: dict|tuple,func: Callable) -> Callable[...,Any]:
    """
    Sorts out the combinations of missing arguements
    
    Note: it will not work if returns takes up the arguements
    that temp takes up as temp will default to the first 
    available arguements and therefore the first dict can only use
    args those of temps' else it doesn't work.
    """
    # gather the next arguement/s
    if type(returns) == dict:
        if type(temp) == dict: return func(**returns|temp)
        elif type(temp) == tuple: return func(*temp,**returns)
        return func(temp,**returns)
    # it must be a tuple
    else:
        if type(temp) == dict: return func(*returns,**temp)
        elif type(temp) == tuple: return func(*(*returns,*temp))
        return func(*(*returns,temp))

def pipe(*args,reverse: bool=False) -> Callable[...,Any]:
    """
    pipe operator for python without class overriding methods (that require additional overhead)
        
    How to use:
    /pipe {"A":[1,2,3]},pd.DataFrame
    def do(a,b,c):
        return a+b+c
    /var=pipe 1,do,(1,1)
    print(var) # returns 3
    
    If wanting to reverse the order of piping:
    /pipe (1,1),do,1,reverse=True
    or
    ## For temporary use
    %env REVERSE_PIPE = True
    /pipe (1,1),do,1
    # Note: can also do: os.environ["REVERSE_PIPE"]= True
    or
    ## To fix the direction (has to be manually changed back to False if wanting to undo)
    %env FIX_REVERSE_PIPE = True
    /pipe (1,1),do,1    
    """
    ### alternatives for reversing the piping direction
    # in case it changes
    ## temporary
    REVERSE_PIPE = os.getenv('REVERSE_PIPE', False)
    ## fixed
    FIX_REVERSE_PIPE = os.getenv('FIX_REVERSE_PIPE', False)
    length = len(args)
    # allows you to swap the piping direction if desired
    if reverse == True or REVERSE_PIPE == "True" or FIX_REVERSE_PIPE == "True":
        interval = range(length-2,-1,-1)
        returns = args[length-1]
        scalar = -1 # ensures the direction of gathering arguements is correct
    else:
        interval = range(1,length)
        returns = args[0]
        scalar = 1
    # piping
    iter_val=iter(interval) # in case needing to skip an iteration
    for i in iter_val:
        # in case of lambda expressions
        func = args[i]
        # try in-builts first
        try:
            returns = func(returns)
            continue
        except (TypeError, ValueError):
            # handle or gather for dicts
            if type(returns) == dict:
                try:
                    returns = func(**returns)
                    continue
                except (TypeError, ValueError):
                    returns = argmiss(returns,args[i+1*scalar],func)
                    skip(iter_val,1) # because we gathered this arg we skip it
                    continue
            # set whatevers left as tuple and handle or gather for tuples
            if type(returns) != tuple: # for consistency / less try/excepts
                returns = (returns,)
        try: returns = func(*returns)
        except (TypeError, ValueError):
            returns = argmiss(returns,args[i+1*scalar],func)
            skip(iter_val,1)
    if REVERSE_PIPE == "True": os.environ["REVERSE_PIPE"] = "False"
    return returns

def standardize(x: pd.Series|pd.DataFrame) -> pd.DataFrame:
    """standardize to pd.Dataframe directly"""
    return pd.DataFrame(StandardScaler().fit_transform(x),columns=x.columns)

def Type(df: pd.DataFrame=[],ls: list[str]=[],keep: list[str]=[],this: bool=True,show: bool=False) -> pd.DataFrame:
    """
    Takes in (usually) a dataframe and what datatypes you want/don't want (this=True/False),
    Allowing you to keep columns using keep=[], and allowing you to see all current datatypes using show=True
    """
    if show: return pd.DataFrame([[i,df[i].dtype] for i in df.columns],columns=["column","dtype"]).set_index("column")
    try: keep = df[keep]
    except: keep = pd.Series([])
    for string in ls: df = df[[i for i in df.columns if (df[i].dtype == string) == this]]
    return pd.concat([df,keep],axis=1)

def cprop_plot(occ: pd.DataFrame,part: pd.DataFrame,args={"figsize":(15.5,7.5),"alpha":0.4}) -> None:
    """for plotting conditional occurance proportions against participation proportions"""
    colours = R_colours(len(occ.columns[1:]))
    plt.figure(figsize=args["figsize"])
    plt.xlabel("participation proportion")
    plt.ylabel("occurance proportion")
    plt.title("proportion of conditional occurance vs participation occurance")
    plt.scatter(part,occ,alpha=args["alpha"])
    for i in occ.columns[1:]:
        plt.scatter(part[i],occ[i],alpha=args["alpha"])
    plt.show()

def multi_cprop(df: pd.DataFrame,cols: list) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    cprop but for multi-classes; though drops the target column as you probably shouldn't include it anyway, 
    else it's not difficult to correct e.g. just use cprop on itself.
    """
    # Total overall
    Total = df[[cols]].groupby(cols).agg({cols:"count"}).T
    # data to collect
    participation = pd.DataFrame()
    occurances = pd.DataFrame()
    cts = pd.DataFrame()
    try:
        # we shouldn't do the column we're targeting
        var = df.drop(columns={cols})
    except:
        var = df
    for i in var:
        # counts overall
        counts = df.groupby(i).agg(count=(cols,"count"))
        # occurances in a category
        occ = df.groupby([i,cols]).agg(occ=(cols,"count")).reset_index().pivot(index=i,columns=cols,values="occ")
        # for testing
        #display(occ,cols)
        #return 
        temp = pd.DataFrame(occ)
        # fix the index
        temp.index = temp.index.astype('str')
        for j in range(len(temp.index.values)):
            temp.index.values[j] = temp.index.names[0] + " " + temp.index.values[j]
        temp.index.names = [""]
        # transform
        participation_temp = occ.apply(lambda x:x/Total[x.name][0])
        occurances_temp = occ.apply(lambda x:x/counts["count"])
        # set the index
        participation_temp.index = temp.index
        occurances_temp.index = temp.index 
        counts.index = temp.index
        # join
        occurances = pd.concat([occurances,occurances_temp])
        participation = pd.concat([participation,participation_temp])
        cts = pd.concat([cts,counts])
    occurances.columns.name = "occurances"
    participation.columns.name = "participation"
    return occurances,participation,cts

def cprop(df: pd.DataFrame,occurance: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    occurance must be a binary indicator; Try to use categorical variables for better results
    How to use:
    
    conditions0,conditions1 = cprop(df,occurance,graph=True)
    
    p = conditions0.plot("part_prop","occ_prop",kind="scatter",title="occ_prop vs part_prop",figsize=(15.5,7.5),alpha=0.4)
    conditions1.plot("part_prop","occ_prop",kind="scatter",ax=p,c="orange",alpha=0.4)
    plt.show()
    """
    cols = occurance.columns[0]
    conditions1 = pd.DataFrame()
    for i in df.columns:
        # save the count for later so that we don't make claims on low observations
        temp = df.groupby(i).agg(occ_prop=(cols,"mean"),count=(i,"count"),part_prop=(cols,"sum")) # all three methods and the groupby omit NAs
        # add the variable name next to it's category
        # need to change the type to make things easier
        temp.index = temp.index.astype('str')
        for i in range(len(temp.index.values)):
            temp.index.values[i] = temp.index.names[0] + " " + temp.index.values[i]
        temp.index.names = [""]
        conditions1 = pd.concat([conditions1,temp])
    # Total occurances
    Total_occurance1 = occurance.sum()
    Total_occurance0 = occurance.count() - Total_occurance1
    # conditions
    conditions0 = pd.DataFrame(conditions1)
    conditions0["occ_prop"] = 1 - conditions1["occ_prop"]
    conditions0["part_prop"] = (conditions0["count"] - conditions0["part_prop"])/(Total_occurance0[0]) # due to still being conditions1 equivalent here
    conditions1["part_prop"] = conditions1["part_prop"]/Total_occurance1[0]
    return conditions0,conditions1

def indicator_encode(data: pd.DataFrame,to: list=[]) -> pd.DataFrame:
    """encodes data as positive integers by its set; takes a parameter 'to' for custom mapping"""
    encoding = pd.Series(pd.Series.unique(data)).reset_index().rename(columns={0:data.name})
    if len(to) > 0:
        try: encoding["index"] = to
        except:
            print("mapping length required: ",len(encoding)," (",len(encoding)-len(to),"more elements needed )")
            return encoding
    encoding = data.reset_index().merge(encoding, left_on=data.name, right_on=data.name).iloc[:,2]
    encoding.name = data.name
    return encoding

def pairs_matrix(ls1: list,ls2: list) -> pd.DataFrame:
    """Forms a matrix from two lists"""
    ls2,df=pd.Series(ls2,index=ls2),pd.DataFrame()
    for i in ls1: df[i] = i+ls2
    return df

def dupes(data: pd.DataFrame,ID: str,col: str) -> pd.DataFrame:
    """
    Checks how many categories an ID represents where duplicated IDs are returned from having more than one value it maps to
    Inputs: data,ID,col
    """
    data = data.groupby(ID).agg({col:"unique"})
    data["count"] = data.applymap(len)
    data = data[data["count"] > 1]
    display(data)
    return data

def full_sim_check(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a pd.DataFrame and returns the unique pairs along with their jaccards similarity

    To filter out similarities where necessary the following should work per column:

    df.iloc[:,2*(i-1):2*i][df.iloc[:,2*(i-1):2*i]["jaccards similarity"] > 0.7]
    
    To get the columns:
    
    list(filtered[filtered != "jaccards similarity"].index)    
    """
    new_df = pd.DataFrame()
    for i in data.columns:
        df = data.loc[:,i].dropna() ### pd.Series??
        lst_pair = pairs(list(df))
        temp_df = pd.concat([lst_pair.rename(df.name),j_sim(lst_pair).rename(str(df.name)+"_similarity")],axis=1)
        new_df = pd.concat([new_df,temp_df],axis=1)
    return new_df

def j_sim_ls(lst2: list) -> float:
    """jaccards similarity of two lists"""
    return jaccard_similarity(set(lst2[0]),set(lst2[1]))

def j_sim(lst1: pd.Series) -> pd.Series|pd.DataFrame:
    """jaccards similarity of a pairs pd.Series e.g. use pairs() first if needed"""
    return lst1.str.split(",").apply(j_sim_ls)

def pairs(lst: list) -> pd.DataFrame:
    """Takes a list and returns a pd.Series of the unique pairs; works quickly on large data"""
    temp_length=length=len(lst)
    new_lst_1,new_lst_2,new_lst_3=[],[],[]
    for i in range(int(length/2+1)):
        new_lst_1 += lst[i+1:-i-1]*2
        temp_length += -2
        start = [lst[i]]
        end = [lst[length-1-i]]
        new_lst_2 += start*temp_length+end*temp_length
        new_lst_3 += [str(start[0])+","+str(end[0])]
    df = pd.concat([pd.Series(new_lst_2)+","+pd.Series(new_lst_1),pd.Series(new_lst_3)],ignore_index=True) 
    # slice the end off because new_lst_3 signals the start of 
    # the next iteration so at the last iteration it should not start again
    return df.iloc[:-1]

def time_formatted(num: int) -> str:
    """Returns the time in hours,minutes,seconds format"""
    return f"{np.floor(num/3600)} hour/s {np.floor((num/60) %60)} minute/s {np.floor(num%60)} second/s"

def run_r_command(command: str,R: str='C:/Program Files/R/R-4.3.1/bin/R.exe') -> str:
    """
    Returns R terminal output for python interface 
    
    (If used elsewhere consider changing the file location hardcoded in this module e.g. where you launch the R terminal)
    """
    process = subprocess.run([R, '--vanilla'], input=command.encode(), capture_output=True)
    output = process.stdout.decode('utf-8')
    # removes the initial stuff you get on first load
    return output[716:-5] if process.returncode == 0 else print(output)

def nas(data: pd.DataFrame|pd.Series) -> int:
    """
    Prints the total number of missing values
    """
    print("\nmissing values ",data.isna().sum().sum())

def R_colours(n: int) -> pd.Series:
    """
    Runs an R script returning its' default pallete as a pandas series
    """
    command="scales::hue_pal()("+str(n)+")"
    df = pd.DataFrame(run_r_command(command).split("#")[1:], columns=['colours'])
    return "#"+df['colours'].str[:6]

def gather(data: pd.DataFrame|pd.Series,axis: int=0) -> pd.DataFrame:
    """
    Is the inverse function to pd.Series.explode except it only works for pd.DataFrame() objects
    
    axis=0|1 (rows or cols to gather towards a singluar row or col)
    """
    # in case wanting to work on series objects: 
    data = pd.DataFrame(data)
    # if axis = 0 # retain the cols but the rows won't make sense
    # if axis = 1 # retain the rows but the cols won't make sense
    if axis==0: return pd.DataFrame([[list(data.loc[:,i]) for i in data.columns]],columns=data.columns)
    elif axis==1: return pd.DataFrame({0:[list(data.loc[i,:]) for i in data.index]},index=data.index)
    raise ValueError("axis must be 0 or 1 for rows or cols respectively")

def validate(model: str) -> pd.DataFrame:
    """
    Global validation of the Linear model using the gvlma package from R
    
    reference:
    Pena, EA and Slate, EH (2006). “Global validation of linear model assumptions,” J.\ Amer.\ Statist.\ Assoc., 101(473):341-354.
    """ 
    command="sink(tempfile())\nlibrary(gvlma)\nlibrary(tidyverse)\nmodel="+model
    command+="\ndf=data.frame(summary(gvlma(model)))\ncloseAllConnections()\ndf"
    #### Runs the R command, splits into lines, then starts at line index 7 onwards
    R = run_r_command(command).split('\n')[7:]
    ##### Puts list back into string / unsplits the list, formats the string, gets table, resets the index
    k=pd.read_csv(StringIO("\n".join(R)), sep='\s+').reset_index() 
    # remove letters from col 1 into col 0, set index as col 0, and give the index an empty name
    ## get indexs of rows in col 1 containing non-numeric characters
    indx=k[k.iloc[:,1].str.contains('[A-Za-z]')].index
    k.iloc[list(indx),0] += " " + k.iloc[list(indx),1] ## append col 0 at the indexs
    k=k.set_index(k.columns[0])
    k.index.name ='' 
    ##### shift over to the left to remove the characters at the indexs because they're still there
    k.iloc[list(indx)] = k.iloc[list(indx)].shift(-1,axis=1)
    k.iloc[:,2] = k.iloc[:,2]+" "+k.iloc[:,3]+" "+k.iloc[:,4].fillna('') ## add the conclusions
    return k.shift(3,axis=1).drop(columns=[k.columns[0],k.columns[1],k.columns[2]]) ## shift to the right, drop the unwanted columns
     
def scrape(url: str,form: str=None,header: str="") -> str | dict:
    """
    Scrapes a webpage returning the response content or prints an error message when the response code is not 200
    
    if you expect the response content to have a particular form you can specify it using form='**expected_format**'
    
    i.e. form=None | 'html' | 'json' by default form is set as form=None returning response.content
    """
    response=requests.get(url,headers=header)
    if response.status_code != 200: return print("Error: ",response.status_code," ",response)
    if form == 'html': return BeautifulSoup(response.content, "lxml")
    elif form == 'json': return json.loads(response.content)
    return response.content

def render(html: str,head: str=':None') -> None:
    """
    Formats Beautifulsoup, list, or string to html for display
    
    head=':None' by default but can be set to any range in the form of a string i.e. '1:2'
    
    The function will use Beautiful soup to complete unclosed html tags to ensure the output won't affect the rest of the jupyter notebook
    """
    ### allow for 'head' to be directly inserted for slicing, Markdown reuquires strings, 
    ### Beautiful soup will close tags to stop it affecting the rest of the note book (else i.e. successive divs are nested as children) 
    eval('display(Markdown(str(BeautifulSoup(html['+head+'], "lxml"))))')

def handle_file(name: str,form: str='utf8',FUNC: Callable="",head: str='') -> str:
    """
    returns the file as file.read()
    handle_file allows you to view a file and execute a function on it returning your operations output if you supply a function input. 
    It will open and close the file for you; using the function given to it during the file being open and returning after file closing
    you can pass an encoding arguement, a function, and head as arguements
    i.e.
    head = 10 to see the first 11 characters, header = None to see the entire file
    form = 'utf8' (by default)
    FUNC = type (by default) but will show regardless and if so returns file.read() unless other functions are used.
    """
    with open(name, encoding=form) as file:
        print("Name of the file: ", file.name,"\n")
        print("Type: ", type(file),"\n")
        if head == None or isinstance(head,int): print(file.read()[:head],"\n")
        return FUNC(file) if FUNC else file.read()

def my_info(data: pd.DataFrame) -> pd.DataFrame:
    """My version of .info()"""
    return pd.concat([data.count(),(np.round(data.isnull().mean()*100,2)).astype(str)+"%",data.apply(lambda x: x.dtype)],axis=1).rename(columns={0:"Non-Null count",1:"Pct missing (2 d.p.)",2:"dtype"})

def data_sets(data: pd.DataFrame) -> pd.DataFrame:
    """Returns the unique categories"""
    data_cats = data.agg(pd.unique) # can use set as well
    data_cats = pd.DataFrame(data_cats).T
    return data_cats.apply(pd.Series.explode, ignore_index=True)

def try_except(trying: Callable,exception: Callable) -> Any|NoReturn:
    """
    Functional form of try-except; useful for lambda expressions
    """
    try:return trying()
    except:return exception()

def digit_slice(num: float,places: int) -> float:
    """rounds down to the nearest n decimal places"""
    scalar=10**places
    return np.floor(num*scalar)/scalar

def preprocess(data: pd.DataFrame,file: str="",variable: str="",limit: int=10*6) -> pd.DataFrame:
    """ 
    Function for cleaning / preprocessing data 
        
    Takes a pd.DataFrame and optionally a .txt notes file

    1. gives preview of the data and optionally the notes file
    2. shows consistency in individual variable types
    3. Then splits into numeric and categorical data 
        - examining the continutiy and stats of the numerics 
        - counting and displaying unique categories

        Additionally, the following should be checked 
        if applicable:
            - validating IDs (if any)
            - checking for mistakes or messy text data
    """
    # display the data, # show notes file 
    display(Markdown("**Preview:**"))
    try:print(handle_file(file))
    except:print(variable)
    display(data)
    # show the info #
    display(Markdown("**Type consistency:**"))
    display(my_info(data))
    if len(data) < limit:
        try:
            # is there enough data for analysis
            sns.heatmap(data.isnull())
            plt.title("Heatmap of missing values")
            plt.show()
        except Exception as e:
            print(e)
    # show description # check continuity # check uniqueness
    display(Markdown("**Numeric data:**"))
    def try_mod(df: pd.DataFrame) -> pd.Series:
        """
        returns the proportion of non-continuous numbers for a variable
        """
        try:return len(df[df%1==0])/len(df) # we can only return list objects to a pd.Series or pd.DataFrame (as that's what it expects)
        except:return "Error: "+str(df.dtype)
    # use try excepts in case no numerical or categorical data to display e.g. no columns
    try:
        nums=data.describe()
        continuity=data.loc[:,nums.columns.values].fillna(0).apply(try_mod)
        continuity.name="proportion discrete"
        def try_digit_slice(cell: Any,places: int=2) -> Any:
            """attempts to round numbers"""
            try:return digit_slice(cell,places)
            except:return cell
        display(pd.concat([nums.iloc[0:1],pd.DataFrame(continuity).T,nums.iloc[1:]]).T.map(try_digit_slice))
    except Exception as e:
        print(e)
    # categories
    display(Markdown("**Categorical data:**"))
    try:
        objs=Type(data,"O")
        display(objs.describe())
        data = data_sets(objs)
        display(data.head(15))
        display(objs.apply(indicator_encode).describe().T)
    except Exception as e:
        print(e)
    return data
    # missing data? # messy data

def jaccard_similarity(A: set, B: set) -> float:
    """Jaccards similarity for two sets"""
    return len(A.intersection(B)) / len(A.union(B))

def shift_check(str1: str,str2: str) -> list:
    """
    Takes two strings and compares the amount shift that one string has to take to become the other whilst
    removing characters from the string being compared to ensuring no duplicates
    """
    str_smaller,str_bigger=(str1,str2) if len(str1) <= len(str2) else (str2,str1)
    new_ls = []
    for i in range(len(str_smaller)):
        new_ls+=[0]
        j = 0
        # calculates the shift
        while j < len(str_bigger) and str_smaller[i] != str_bigger[j]: new_ls[i]+=1;j+=1
        # reduce the bigger string
        if j == len(str_bigger): new_ls[i] = np.nan
        elif str_smaller[i] == str_bigger[j]:
            try: str_bigger = str_bigger[:j]+str_bigger[j+1:]
            except: str_bigger = str_bigger[:j-1]+str_bigger[j:]
    return new_ls

def sim_check(data_: pd.Series,mean: float=10,std: float=10,j_thr: float=0.75,l_thr: int=3,limit: int=200,dis: bool=False) -> None:# make sure all a,b,c are before d=1,e=2,f=3
    """ 
    similarity check used on pd.Series objects to check text data.
    """
    count,data=0,data_.dropna()
    print('\n',data.name,':\n')
    base_data = data
    for base in base_data: # uses each cell of the column
        for temp in data: # goes through each cell of the column
            # make sure it meets the length threshold and it's not identical
            if (abs(len(base) - len(temp)) < l_thr) & (base != temp):
                # overall dissimilarity
                if (dis == True) & (jaccard_similarity(set(base), set(temp)) < j_thr):
                    # enhanced dissimilarity for parts of strings
                    check = shift_check(base,temp)
                    if (np.nanmean(check) > mean) & (np.nanstd(check) > std):
                        print(base,temp)
                        count+=1
                        if count == limit:
                            return print("limit reached: "+str(limit)+" line/s")
                # overall similarity
                elif jaccard_similarity(set(base), set(temp)) > j_thr:
                    # enhanced similarity for parts of strings
                    check = shift_check(base,temp)
                    if (np.nanmean(check) < mean) & (np.nanstd(check) < std): # there are other stats we can try but the std is OK
                        print(base,temp)
                        count+=1
                        if count == limit:
                            return print("limit reached: "+str(limit)+" line/s")
        data = data[data.isin([base]) == False] # to reduce the uneccessary combinations / only get unique ones

class section_path:
    """
    For breaking up parts of a file path

    Note: Designed so that you can create/manipulate your own file paths 
    with ease and doesn't have to necessarily exist yet (unless removing 
    and to an extent adding e.g. at least have the parent paths present)
    """
    def __init__(self,path: str) -> None: self.path=path
    def __repr__(self) -> str: return repr(self.path)
    def __add__(self,name: str) -> object: self.path=os.path.join(self.path,name);return self
    with decorate(property):
        def name(self) -> str: return os.path.basename(self.path) if self.isdir else os.path.basename(self.path).split(".")[0]
        def dir(self) -> str: return self.path if self.isdir else os.path.dirname(self.path)
        def ext(self) -> str: return ".".join(self.path.split(".")[1:])
        def file(self) -> str: return os.path.basename(self.path)
        def back(self) -> object: self.path=os.path.dirname(self.path);return self
        def make(self) -> None: os.makedirs(self.dir)
        def remove(self) -> None:
            if os.path.isdir(self.path): shutil.rmtree(self.path)
            else: os.remove(self.path)
        def isdir(self) -> bool: return True if self.ext=="" else False
        def isfile(self) -> bool: return not self.isdir

@wrap.inplace(next)
def analyze_pickle(filename: str,show: bool=False) -> Generator:
    """
    Analyzes a .pkl file
    
    reading in a pickle file will directly execute code and it's 
    possible if the code is from an untrusted source that it may
    be malicious code. This function analyzes the pickle code without
    executing it via disassembly

    How to use:

    analyze_pickle("my_tuple.pkl") # will return the pickle opcodes used on the pickle machine 
    # (pickle machine is a class that acts like a stack basically; last thing on there at stopping is the python object)
    # (pickle also has different opcode formats depending on the task)
    """
    if filename[-4:]!='.pkl': filename+='.pkl'
    with open(filename, 'rb') as file:
        if show:
            print("%5s | %-12s| %-24s| %-4s|" % ("","code |","name |","arg"))
            print("-"*54)
            # original source code for pickletools.dis
            # code reference: Python Software Foundation. (2024). Python. 3.13. https://github.com/python/cpython/blob/main/Lib/pickletools.py#L2395
            maxproto=0
            for opcode, arg, pos in pickletools.genops(file):
                if opcode.name=="MEMOIZE" and arg==None: continue
                print("%5d: %-12s %-24s %-4s" % (pos,repr(opcode.code)[1:-1],opcode.name,arg))
                maxproto = max(maxproto, opcode.proto)
            print("maximum protocol:",maxproto)
            return (yield) ## because we used yield below it won't return as it should which is also why it's wrapped inplace with next
        yield
        yield from pickletools.genops(file)

TRACEBACK=IPython.core.interactiveshell.InteractiveShell.showtraceback if has_IPython() else sys.excepthook

if has_IPython():    
    current_execution=get_ipython().__getstate__()["_trait_values"]["execution_count"]
