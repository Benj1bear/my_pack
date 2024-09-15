"""
This module holds all the libraries and functions I use
"""
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
import datetime # needed at the very least for unstr
import os
from threading import Thread,RLock
lock=RLock()
from time import time,sleep
from types import ModuleType,BuiltinFunctionType
from typing import Any,Callable
import tempfile
from importlib.util import module_from_spec,spec_from_loader
########### for installing editable local libraries and figuring out what modules python scripts use
import re
import glob
import shutil
from inspect import getfile,getsource,signature,_empty
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
from itertools import combinations
import IPython

def get_arg_count(attr: Any,value: Any=[None]*999) -> list:
    """returns the number of accepted args"""
    ## either it's an invalid type
    ## or any of the numbers less than the length
    if isinstance(attr,Callable)==False:
        raise TypeError("attr must be Callable")
    length=len(value)
    try:
        attr(*value)
        print("\n","-"*20,"passed","-"*20)
        return 1
    except TypeError as e:
        message=" ".join(str(e).split(" ")[1:])
        if " takes no arguements" in message: return 0
        if " exactly one " in message: return 1
        arg_numbers=re.findall(r"\d+",message)
        if len(arg_numbers):
            return [num for i in arg_numbers if (num:=int(i)) < length]
        raise ValueError(f"the value of 'value' should be reconsidered for the attribute '{attr}'.\n\n Original message: "+message)

def find_args(obj: ModuleType|object,use_attr: bool=True,value: Any=[None]*999) -> list:
    """For figuring out how many args functions use"""
    messages=[]
    for attr in dir(obj):
        try:
            attribute=getattr(obj,attr)
            if isinstance(attribute,Callable):
                if use_attr==False:
                    attr=""
                try:
                    attribute(*value)
                except TypeError as e:

                    messages+=[attr+" "+" ".join(str(e).split(" ")[1:])]
                else:
                    messages+=[attr+" "+"any"]
        except:
            pass
    return messages

@property
def toggle_print() -> None:
    """Toggles between enabling and disabling printing to display"""
    global print,display
    if print("",end="")!=None and display()!=None:
        print=__builtins__.print
        display=IPython.core.display_functions.display
    else:
        print=lambda *args,**kwargs: ""
        display=lambda *args,**kwargs: ""

def class_dict(obj: Any) -> dict:
    """For obtaining a class dictionary (not all objects have a '__dict__' attribute)"""
    keys,attrs=dir(obj),[]
    for key in keys:
        try: attrs+=[getattr(obj,key)]  ## some attrs cannot be retrieved i.e. ,"__abstractmethods__" when passing in the 'type' builtin method
        except: pass
    return dict(zip(keys,attrs))

def classproperty(obj: Any) -> classmethod:
    """Short hand for:
    @classmethod
    @property
    """
    return classmethod(property(obj))

#### needs testing ####
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

    Note: all data and methods (except special methods) have been made private in this class
    to allow for more commonly named attributes to be added.

    In python private data and methods strictly don't exist (in what I know currently) i.e.
    in the chain class we have __cache as a private variable but this is accessible via:
    
    chain._chain__cache
    and can be assigned new values or overwritten
    """
    __cache=[]
    def __init__(self,obj: Any=[],**kwargs) -> None:
        self.__obj=obj
        self.__clear
        if hasattr(kwargs,"override"):
            self.__override=kwargs["override"]
        self.__get_attrs(obj)
    # all dunder methods not allowed to be shared (else the chain classes attributes needed for it to work will get overwritten)
    __not_allowed=["__class__","__dir__","__dict__","__doc__","__init__","__call__","__repr__","__getattr__",
                  "__getattribute__","__new__","__setattr__","__init_subclass__","__subclasshook__","__name__",
                  "__qualname__","__module__","__abstractmethods__"]
    def __get_attrs(self,obj: Any) -> None:
        """Finds the new dunder methods to be added to the class"""
        not_allowed=self.__not_allowed.copy()
        for key,value in class_dict(obj).items():
            if re.match("__.*__",key)!=None:
                if key in not_allowed:
                    not_allowed.remove(key)
                elif isinstance(value,Callable): ## we're only wanting instance based methods for operations such as +,-,*,... etc.
                    self.__share_attrs(key,self.__wrap(value))
    @staticmethod
    def __wrap(method: Callable) -> Callable:
        """
        wrapper function to ensure methods assigned are instance based 
        and that the dunder methods return values are wrapped in a chain object
        """
        @wraps(method) ## retains the docstring
        def wrapper(self,*args) -> object: ## will return an instance based method since those are the methods we're after
            return self.__chain(method(*args))
        return wrapper
    __show_errors=False
    @classmethod
    def __share_attrs(cls,key: Any,value: object) -> None:
        """Shares the dunder methods of an object with the class"""
        try:
            setattr(cls,key,value)
            cls.__cache+=[key]
        except:
            if cls.__show_errors:
                print(cls,key,value)

    def __call__(self,*args,**kwargs) -> Any:
        """For calling or instantiating the object"""
        return self.__chain(self.__obj(*args,**kwargs))

    def __repr__(self) -> str:
        return repr(self.__obj)
    @classmethod
    def __add_attr(cls,attr: str) -> None:
        """Dynamically adds new attributes to a class"""
        if hasattr(cls,attr)==False:
            if hasattr(__builtins__,attr): # might add an override here; though you shouldn't really be monkey patching builtins with global functions
                setattr(cls,attr,getattr(__builtins__,attr))
            else:
                setattr(cls,attr,globals()[attr])
            cls.__cache+=[attr]
    @classmethod
    def __static_setter(cls,attr: str) -> None:
        """Sets an attribute as a staticmethod"""
        setattr(cls,attr,staticmethod(getattr(cls,attr)))
    
    def __getattr__(self,attr: str) -> Any:
        """Modified to instantiate return values as chain objects"""
        ## consider global vs local overrides
        if hasattr(self.__obj,attr) and self.__override:
            return self.__chain(getattr(self.__obj,attr))
        self.__add_attr(attr)
        attribute=getattr(self,attr)
        if isinstance(attribute,Callable):
            try: ## pass in the object stored to the Callable
                if len(signature(attribute).parameters) > 0:
                    return self.__chain(partial(attribute,self.__obj))
            except ValueError: 
                try:
                    return self.__chain(partial(attribute,self.__obj)) # if can't use signature since not all (particularly builtins) don't have them
                except:
                    pass
            ## if the Callable has no params then it has to be a staticmethod
            ## (because it's set to an instance of a class which means it will expect 'self' as the first arg)
            if isinstance(attribute,staticmethod):
                return self.__chain(attribute)
            self.__static_setter(attr)
            return self.__chain(getattr(self,attr))
        return self.__chain(attribute)

    def __chain(self,attr: Any) -> object:
        """For creating new chain objects"""
        return chain(attr,override=self.__override)
    
    __override=False
    @property
    def _(self) -> object:
        """Changes scope from global to local or local to global"""
        self.__override=False if self.__override else True
        return self
    @classproperty
    def __clear(cls) -> None:
        """Clears the cache"""
        for attr in cls.__cache:
            if hasattr(cls,attr):
                delattr(cls,attr)
        cls.__cache=[]

class ext:
    """
    Extensions for python data types whereby you can now dynamically create/use methods

    i.e. you can now do:
    def method(self):
        print("hi")
        return self
    @staticmethod
    def testing(self):
        print("hello")
    @classmethod
    def testing2(cls):
        print(cls)
    @property
    def testing3(self):
        print("is a property")
    ext().method().testing()
    ext().testing2()
    ext().testing3()
    # should print:
    # hi
    # hello
    # <class '__main__.ext'>
    # is a property
    Note: these instance based methods are dynamically added e.g. they were not
    part of the original class definition but were already defined elsewhere
    """
    def __init__(self,obj: Any=[]) -> None:
        self.obj=obj
        
    def __repr__(self) -> str:
        return str(self.obj)
    @classmethod
    def __add_attr(cls,attr: str) -> None:
        """Dynamically adds new attributes to a class"""
        if hasattr(cls,attr)==False:
            setattr(cls,attr,globals()[attr])

    def __getattr__(self,attr: str) -> Any:
        self.__add_attr(attr)
        return getattr(self,attr)

class tup_ext:
    """Extensions for tuples"""
    def __init__(self,tup: tuple) -> None:
        self.tup=tup

    def __repr__(self) -> str:
        return str(self.tup)
    
    def __len__(self) -> int:
        return len(self.tup)
    
    def __getitem__(self,index) -> tuple:
        return itemgetter(index)(self.tup)
    
    def __setitem__(self,index,value) -> None:
        temp=list(self.tup)
        temp[index]=value
        self.tup=tuple(temp)

    def __delitem__(self,index) -> None:
        remove=list(range(len(self.tup)))[index]
        if type(remove)!=list:
            remove=[remove]
        self.tup=tuple(value for index,value in enumerate(self.tup) if index not in remove)

class Print:
    """In-time display"""
    def __init__(self,initial: int=0) -> None:
        self.prev=initial
    
    def __call__(self,message: Any) -> None:
        self.clear
        print(message,end="\r")
        self.prev=len(message)
    @property
    def clear(self) -> None:
        print(" "*self.prev,end="\r")

def import_sklearn_models(kind: str) -> None:
    """Convenience function for importing lots of sklearn models"""
    if kind!="classifiers" and kind!="regressors":
        raise ValueError("'kind' must be in [\"classifiers\",\"regressors\"]")
    if kind=="classifiers":
        models=zip(["tree","neighbors","ensemble","linear_model","naive_bayes","dummy","neural_network","svm"],
                ["DecisionTreeClassifier","KNeighborsClassifier","RandomForestClassifier,GradientBoostingClassifier",
                    "LogisticRegression","GaussianNB","DummyClassifier","MLPClassifier","SVC"])
    if kind=="regressors":
        pass ### to add

    for directory,model in models:
        exec("from sklearn."+directory+" import "+model,globals())

def all_multi_combos_dict(arr: dict) -> list:
    """Same as all_multi_combos but for retaining key/column placing"""
    def all_multi_combos(current_combo: list=[]) -> list:
        nonlocal arr
        """Returns a list of all combinations of multiple lists"""
        all_combos,index=tuple(),len(current_combo)
        if index==len(arr):
            return [current_combo]
        for i in list(dct_ext(arr)[index].values())[0]:
            all_combos=(*all_combos,*all_multi_combos(current_combo+[i]))
        return all_combos
    return all_multi_combos()

def all_multi_combos(arr: list[list],current_combo: list=[]) -> list:
    """Returns a list of all combinations of multiple lists"""
    all_combos,index=tuple(),len(current_combo)
    if index==len(arr):
        return [current_combo]
    for i in arr[index]:
        all_combos=(*all_combos,*all_multi_combos(arr,current_combo+[i]))
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

SKLEARN_IMPORTED=False
def get_classifier(mod: str="",show: bool=False,plot: bool=False,**kwargs) -> tuple[Callable,str]:
    """Convenience function for obtaining common sklearn and other classifier models"""
    global SKLEARN_IMPORTED
    if show:
        print("Available models: tree, knn, forest, nb, dummy, nnet, svm, gb, log")
        return
    if SKLEARN_IMPORTED==False: 
        import_sklearn_models()
        SKLEARN_IMPORTED=True
    get=lambda _mod: _mod if len(kwargs)==0 else _mod(**kwargs)
    match mod:
        case "tree":
            FUNC,depth = get(DecisionTreeClassifier),"Depth"
        case "knn":
            FUNC,depth = get(KNeighborsClassifier),"neighbors"
        case "forest":
            FUNC,depth = get(RandomForestClassifier),"estimators"
        case "nb":
            FUNC,depth = get(GaussianNB),"iteration"
        case "dummy":
            FUNC,depth = get(DummyClassifier),"iteration"
        case "nnet":
            FUNC,depth = get(MLPClassifier),"iteration"
        case "svm":
            FUNC,depth = get(SVC),"iteration"
        case "gb":
            FUNC,depth = get(GradientBoostingClassifier),"iteration"
        case "log":
            FUNC,depth = get(LogisticRegression),"iteration"
        case _:
            return print(f"model '{mod}' doesn't exist")

    if plot:
        return FUNC,depth
    return FUNC

def dct_join(*dcts) -> dict:
    """For concatenating multiple dicts together where all have the same keys"""
    keys=biop(map(lambda x:x.keys(),dcts),"|")
    return {key: [dct[key] if key in dct else None for dct in dcts] for key in keys}

def biop(data: Any,op: str) -> Any:
    """applies a reduce using a binary operator"""
    def bi_op(x: Any,y: Any) -> Any:
        """dynamic binary operation"""
        exec("temp=x"+op+"y")
        return locals()["temp"]
    return reduce(bi_op,data)

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
    ## code reference: https://stackoverflow.com/questions/3854692/generate-password-in-python
    # changes made: allowed for a wider range of characters
    """
    if selection==None:
        if type(char_range)==int:
            char_range=range(char_range)
        selection="".join(char for i in char_range if "\\" not in repr((char:=chr(i))))
    return "".join(secrets.choice(selection) for _ in range(length))

def random_shuffle(arr: list|str) -> list|str:
    """Pseudo-randomly shuffles a list or string"""
    flag,index=0,[]
    if type(arr)==str:
        flag,arr=1,list(arr)
    for i in range(len(arr)):
        temp=secrets.choice(range(len(arr)))
        index+=[arr[temp]]
        del arr[temp]
    if flag:
        return "".join(index)
    return index

def format_password(upper: int=0,lower: int=0,punc: int=0,num: int=0,other: int=0,char_range: int=range(127)) -> str:
    """Creates a new password formatted to password rules"""
    selection=""
    if upper:
        selection+=create_password(upper,string.ascii_uppercase)
    if lower:
        selection+=create_password(lower,string.ascii_lowercase)
    if punc:
        selection+=create_password(punc,string.punctuation)
    if num:
        selection+=create_password(num,string.digits)
    if other:
        selection+=create_password(other,char_range=char_range)
    return random_shuffle(selection)

def save_tabs(filename: str,*args,**kwargs) -> None:
    """save urls to .txt file"""
    with open(filename,"w") as file:
        file.write("\n".join(get_tabs(*args,**kwargs)))

def browse_from_file(filename: str) -> None:
    """Browses urls from .txt file"""
    with open(filename, 'r') as file:
        urls=[url for line in file if (url:=line.strip())]
    browse(urls) if urls else print("No links found in the file.")

def browse(urls: list[str]) -> None:
    """Uses Google Chrome to browse a selection of links"""
    return [webbrowser.open(url, new=2, autoraise=True) for url in urls]

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
    if close_init:
        pyautogui.hotkey('ctrl', 'w')
    return links

def to_pickle(obj: object,filename: str) -> None:
    """Convenience function for pickling objects in python with context management"""
    with open(filename+'.pkl','wb') as file:
        pickle.dump(obj, file)

def read_pickle(filename: "str") -> object:
    """Convenience function for reading pickled objects in python with context management"""
    with open(filename+'.pkl', 'rb') as file:
        return pickle.load(file)

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
            if isinstance(arg,annotation)==False:
                raise TypeError(message)
        if type(annotation) in (tuple,list,set):
            if type(arg) not in (tuple,list,set):
                raise TypeError(f"arguement '{key}' must be of type {annotation}. Instead recieved: {arg}")
            if len(arg)!=len(annotation):
                raise Exception(f"length mismatch between arguement '{key}' and annotation {annotation}")
            for type_annotation,arguement in zip(annotation,arg):
                checker(type_annotation,arguement)
        else:
            checker(annotation,arg)

    args,kwargs,annotations=kwargs["args"],kwargs["kwargs"],FUNC.__annotations__
    try:
        annotations["return"]
    except:
        raise KeyError("Key 'return' does not exist in .__annotations__. You must annotate a return type")
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
            except:
                None
        ## then the args
        args=iter(args)
        for key,annotation in annotations.items():
            if key=="return":
                continue
            try:
                arg=next(args)
            except StopIteration:
                break
            try_check(*(arg,annotation,key,f"arguement '{key}' must be of type {annotation}"))
        ## check for kwargs, args specific types
    else:
        try_check(*(arg,annotations["return"],"return",f"return arguement must be of type {annotations['return']}"))

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
        if defaults:
            self.checks=(*self.checks,*args)
        else:
            self.checks=args

    def __repr__(self) -> str:
        return str(self.FUNC)

    def __call__(self,*args,**kwargs) -> Callable:
        """Runs through all the checks before calling using the functions arguements"""
        for __check in self.checks:
            __check(self.FUNC,args=args,kwargs=kwargs)
        return self.FUNC(*args,**kwargs)
    @classmethod  ## if we want global class manipulation before instantiation we have to use @classmethod ##
    def add(cls,*args: Callable) -> object:
        """adds additional custom checks to inputs (globally e.g. for all instances)"""
        cls.checks=(*cls.checks,*args)
        return cls
    @classmethod
    def remove(cls,*args: Callable) -> object:
        """removes checks from all instances of the class"""
        cls.checks=tuple(__check for __check in cls.checks if __check not in args)
        return cls
    @classmethod
    def use(cls,*args: Callable,defaults: bool=True) -> object:
        """adds additional custom checks to inputs (locally e.g. only on the instance used)"""
        return partial(cls,args=args,defaults=defaults) ## make an instance of the class with the new args ##

class Sub:
    """shorthand version of re.sub"""
    def __init__(self,code: str) -> None:
        self.code=code

    def __repr__(self) -> str:
        return self.code

    def __call__(self,regex: str,repl: str="",flags=re.DOTALL) -> object:
        """For string substitution with regex"""
        self.code=re.sub(regex,repl,self.code,flags=flags)
        return self

    @property
    def get(self) -> str:
        return self.code

def extract_code(code: str,repl: str=" str ") -> str:
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

# needs more testing but works in simple cases
def export(section: str | Callable,source: str | None=None,to: str | None=None,option: str="w",show: bool=False,recursion_limit: int=10) -> str | None:
    """
    Exports code to a string that can then be used to write to a file or for use elsewhere
    Example: (save the following into a file called test.py)
    ##########################################
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
    ##########################################
    Then run:

    from test import hi
    from my_pack import export
    export(hi,show=True,to="new.py",option="a")
    # It will write to a file called new.py
    # and should print
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
    if isinstance(section,Callable)==True:
        section,FUNC,source=source_code(section),[section.__name__],section.__module__
    else:
        ## check for functions ## raw strings only
        FUNC=extract_code(FUNC).get
        FUNC=[i[3:-1].strip() for i in re.findall(r"def\s*\w*\s*\(",FUNC)]
    # prep section
    variables,code_export=get_variables(section),section
    if len(variables)>0:
        # gather all functions,classes available to the .py file
        callables,source=all_callables(source,True)
        if len(FUNC)>0:
            callables=[func for func in callables if func.__name__ not in FUNC]
        # start exporting code
        if show:print("initial section:\n"+"-"*20+"\n"+section+"\n"+"-"*20)
        ######################### add an option to keep the initial section separate from the rest of the code??
        code_export,modules,module_names=get_code_requirements(*(section,callables,variables,variables,source,show,{},pd.DataFrame()),limit=recursion_limit)
        ## add the modules ##
        if len(modules)>0:
            header=""
            for key in list(modules.keys()):
                module=modules[key]
                if len(module)==0:
                    header+="import "+key
                    name=module_names[module_names[0]==key][2]
                    if len(name)>0:
                        header+=" as "+list(name)[0]
                    header+="\n"
                else:
                    header+=f"from {key} import "+", ".join(module)+"\n"
            # append all modules to the top of the section
            code_export=header+"\n"+code_export
    if to==None:
        return code_export
    #####################################################################
    print("----------final results---------- \n")
    print(code_export)
    #####################################################################
    with open(to,option) as file:
        file.write(code_export)

def split_list(reference: list[str],condition: Callable) -> (list,list):
    """For splitting one list into two based on a condition function"""
    new,remaining=[],[]
    for i in reference:
        try:
            if condition(i):
                new+=[i]
            else:
                remaining+=[i]
        except:
            None
    return new,remaining
    
def get_code_requirements(section: str,callables: list[str],temp_variables: list[str],variables_present,source: str,show: bool=False,modules: dict={},current_modules: pd.DataFrame=pd.DataFrame(),recursions: int=0,limit: int=20) -> str:
    """Gets the required code in order to export a section of code from a .py file maintainably"""
    # separate variables and attributes
    attrs,variables=split_list(temp_variables,lambda var:True if "." in var else False)
    ## search attrs for callables and modules ##
    ################################################ needs testing ##############################################
    attr_exports,definitions,callables,module_names=search_attrs(*(attrs,source,callables))
    #############################################################################################################
    ## do the same but for the variables and then combine ##
    new_exports,callables=split_list(callables,lambda func:True if (func.__name__ in variables)==True else False)
    new_exports+=attr_exports
    if len(new_exports) > 0:
        ## add the new code ##
        for func in set(new_exports):# a list of functions from the module
            try:
                exec(f'temp=__import__("{source}").{func.__name__}')
            except Exception as e:
                name=module_names[module_names[0]==func].dropna()
                if len(name)>0:
                    exec(f'temp=__import__("{source}").{list(name[2])[0]}')
                    name[0]=list(name[0])[0].__name__
                    current_modules=pd.concat([current_modules,name])
                else: ## attribute doesn't exist
                    continue
            local_temp=locals()["temp"]
            section,modules=add_code(*(section,modules,local_temp,source))
        section+=definitions
        ## print the current Recursion ##
        if show:print(f"Recursion: {recursions}"+":\n"+"-"*20+"\n"+section+"\n"+"-"*20)
        ## limit the amount of variables needing to be used ##
        new_variables_present=get_variables(section)
        temp_variables=[i for i in new_variables_present if i not in variables_present]
        if len(temp_variables)==0:
            return section,modules,current_modules
        ## make sure there's some safety in case errors occur ##
        if recursions==limit:
            print(f"recursion limit '{limit}' reached\n\nNote: the function may not have completed, if true, adjust the recursion limit or enter in the current code section to continue")
            return section,modules
        recursions+=1
        ## you have to return the recursion else it won't work properly ##
        return get_code_requirements(*(section,callables,temp_variables,new_variables_present,source,show,modules,current_modules,recursions))
    return section+definitions,modules,current_modules

def add_code(section: str,modules: dict,local_temp: Callable,source: str) -> (str,dict):
    """For retrieving and appending necessary code the section depends on"""
    try:
        ## assume it's a function or class ##
        if local_temp.__module__ == source:
            section+="\n"+source_code(local_temp)
        else:
            if local_temp.__module__ not in modules:
                modules[local_temp.__module__]=[local_temp.__name__]
            else:
                modules[local_temp.__module__]+=[local_temp.__name__]
    except:
        ## is it a module ##
        if type(local_temp)==ModuleType:
            if local_temp.__name__ not in modules:
                modules[local_temp.__name__]=[]
        else:
            raise TypeError(f"Variable '{local_temp}' from new_exports is not a Callable or module type")
    return section,modules

def search_attrs(attrs: list[str],source: str,callables: list[Callable]) -> (list[str],str,list[Callable],pd.DataFrame):
    """Traverses an attribute to uncover where each of the individual attribute came from"""
    new_exports=[]
    # only add it in if there's a callable for it
    for attr in attrs: # go through all the attrs
        ## make sure the attr itself is not a module ##
        try:
            exec(f"temp=__import__('{source}').{attr}")
            local_temp=locals()["temp"]
            if isinstance(local_temp,type): ## new class
                if source+"."+attr in str(local_temp):
                    new_exports+=[local_temp]
            elif isinstance(local_temp,ModuleType): 
                # then we need to import this module
                new_exports+=[local_temp]
                continue
        except: ## attribute doesn't exist
            None
        module_name,attribute,module="","",ModuleType("")
        for i in attr.split("."): # go up starting with the first one
            attribute+=i
            try:
                exec(f"temp=__import__('{source}').{attribute}")
                local_temp=locals()["temp"]
                if isinstance(local_temp,Callable): # it won't be a variable but might be a module
                    if "." not in attribute:
                        # get it's source code or check it for module
                        new_exports+=[[local_temp,None,None]]
                    else:
                        if len(module.__name__)>0:
                            if module.__name__==local_temp.__module__ or (module.__name__ in local_temp.__module__)==True:
                                new_exports+=[[module,None,module_name]]
                                module=ModuleType("")
                        # if the source code exists then it has been assigned
                        # else it's already defined from the class definition
                        ## the only flaw to this approach is type based methods and builtins
                        elif local_temp.__name__!=i:  ## this could be an easy point of failure ##
                            new_exports+=[[local_temp,attribute+"="+local_temp.__name__+"\n",None]]
                            if module.__name__!="":
                                new_exports+=[[module,None,module_name]]
                                module=ModuleType("")
                        else:
                            try:
                                source_code(local_temp) # if we can't get it it's because it's a builtin or it's already defined
                                new_exports+=[[local_temp,attribute+"="+local_temp.__name__+"\n",None]]
                            except:
                                if isinstance(local_temp,BuiltinFunctionType):  ########## what about partial functions?
                                    new_exports+=[[None,attribute+"="+local_temp.__name__+"\n",None]]
                elif isinstance(local_temp,ModuleType):
                    module,module_name=local_temp,attribute
            except: ## attribute doesn't exist
                None
            attribute+="."
        if module.__name__!="":
            new_exports+=[[module,None,module_name]]
    new_exports=pd.DataFrame(new_exports).drop_duplicates() # columns are callable/module,definitions,module_name
    if len(new_exports)>0: # if there are any
        allowed_exports,callables=split_list(callables,lambda func:True if (func in list(new_exports[0]))==True else False)
        definitions=new_exports[new_exports[0].isin(allowed_exports)][1].dropna().sum()
        if type(definitions)!=str: # in case pd.Series([]).sum() which returns 0
             definitions=""
        else:
            definitions="\n"+definitions
        new_exports=new_exports[[0,2]]
        return allowed_exports,definitions,callables,new_exports[new_exports.isnull()==False]
    return [],"",callables,[]

def all_callables(module: str,return_module: bool=False) -> list[str] or (list[str],str):
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
    callables=[]
    for i in dir(source):
        exec("temp=source."+i)
        if isinstance(locals()["temp"],Callable)==True:
            callables+=[locals()["temp"]]
        if return_module==True and isinstance(locals()["temp"],ModuleType)==True:
            callables+=[locals()["temp"]]
    if return_module:
        return callables,module
    return callables

def side_display(dfs:pd.DataFrame | list[pd.DataFrame,...], captions: str | list=[], spacing: int=0) -> None:
    """
    # code reference: Lysakowski, R., Aristide, (2021) https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side,CC BY-SA,
    # changes made: Added flexibility to user input and exception handling, made minor adjustments to type annotations and code style preferences, implemented my comments
    Display pd.DataFrames side by side
    
    dfs: list of pandas.DataFrame
    captions: list of table captions
    spacing: int number of spaces
    """
    # for flexibility and exception handling
    if type(dfs)!=list:
        dfs=[dfs]
    if type(captions)!=list:
        captions=[captions]
    length_captions,length_dfs=len(captions),len(dfs)
    if length_captions>length_dfs:
        raise Exception(f"The number of catpions '{length_captions}' exceeds the number of data frames '{length_dfs}'")
    elif length_captions<length_dfs:
        captions+=[""]*(length_dfs-length_captions)
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
    flag=0 # add one to stop to be inclusive for string slicing
    if type(stop) == str:
        flag=1
    for index,key in enumerate(ls):
        if start==key and type(start) != int:
            start=index
        if stop==key and type(stop) != int:
            stop=index
        if type(start) != str and type(stop) != str:
            break
    # check for errors
    if type(start) == str or type(stop) == str:
        # try to be helpful with the error message
        if type(start) == str and type(stop) == str:
            raise KeyError("key_slice failed: Both start and end slice arguements are not in the dictionaries key values")
        for which,value in [("Starting",start),("Ending",stop)]:
            if type(value) == str:
                raise KeyError(f"in function key_slice: {which} slice '{value}' is not in the dictionaries key values")
    if flag==1 and stop > 0:
        stop+=1
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
    def __init__(self,dct: dict) -> None:
        self.dct=dct
    
    def __repr__(self) -> str:
        """Makes sure to display a dict and not a memory location"""
        return str(self.dct)

    def __len__(self) -> int:
        return len(self.dct)

    def __getitem__(self,index: int|list|tuple|slice) -> dict:
        if isinstance(index,int):
            temp_dct=list(self.dct.items())[index]
            return dict.fromkeys([temp_dct[0]],temp_dct[1])
        if isinstance(index,str):
            return {index:self.dct[index]}
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
        if type(args) != list and type(args) != tuple:
            args=[args]
        # make sure keys exist
        for key in index:
            if key not in self.dct:
                self.dct[key]=None
        # get keys and set them
        dct=self.__getitem__(index)
        keys=list(dct.keys())
        # catch errors
        if len(keys)-len(args)!=0:
            raise Exception(f"in dct_ext.__setitem__: Mismatch between number of keys to set and arguements to be assigned\nkeys: {keys}\nargs: {args}")
        for key,arg in zip(keys,args):
            self.dct[key]=arg
    @property
    def keys(self) -> list:
        return list(self.dct.keys())
    @property
    def values(self) -> list:
        return list(self.dct.values())
    @property
    def items(self) -> list:
        return list(self.dct.items())

def digit_format(number: str | int | float) -> str:
    """Formats numbers using commas"""
    number=str(number)
    parts=number.split(".")
    try:number="."+parts[1]
    except:number=""
    count=-1
    for i in parts[0][::-1]:
        count+=1
        if abs(count) % 3 == 0:
            number=i+","+number
        else:
            number=i+number
    return number[:-1]

def file_size(filename: str,decimals: int=2) -> str:
    """Returns the file size formatted according to its size"""
    unit=" KMGT"
    size=os.path.getsize(filename) # in bytes
    power=len(str(size))-1
    with open(filename) as file:
        return f"""file size: {size/10**(int(power/3)*3):.{decimals}f} {unit[int(power/3)].strip()}B
lines: {digit_format(sum(1 for _ in file))}"""

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
        if diff > self.important:
            print(diff)
        self.time=current

def get_builtins() -> list:
    """
    Gets a list of the builtin functions
    seems to be differences between jupyter notebook and python
    """
    if isinstance(__builtins__,dict) == False:
        return dir(__builtins__)
    return __builtins__

def remove_docstrings(section: str,round_number: int=0) -> str:
    """For removing multiline strings; sometimes used as docstrings (only raw strings)"""
    avoid="'"
    if round_number%2==1:
        avoid='"'
    new_section,in_string,in_comment,count="",False,False,0
    for char in section:
        prev=(count,)
        if char==avoid and in_comment==False:
            count+=1
            if count==3:
                in_string,count=(in_string+1)%2,0
        elif char=="#" and in_string==False:
            in_comment=True
        elif char=="\n" and in_comment==True:
            prev,count,in_comment=(0,),0,False # it should be 0
        if in_string==False:
            new_section+=char
        if prev[0]-count==0:
            count=0
    new_section=new_section.replace(avoid*3," str ")# the start of the string will still be there
    if round_number==1:
        return new_section
    return remove_docstrings(*(new_section,round_number+1))

def remove_strings(section: str) -> str:
    """For removing strings (only on raw strings)"""
    new_section,in_string,in_comment,prev_char="",False,False,None
    for char in section:
        if (char=="'" or char=='"') and in_comment==False:
            if prev_char==None:
                in_string,prev_char=(in_string+1)%2,(char,)
            elif char==prev_char[0]:
                in_string,prev_char=(in_string+1)%2,None
        elif char=="#" and in_string==False:
            in_comment=True
        elif char=="\n" and in_comment==True:
            in_comment=False
        if in_string==False:
            new_section+=char
    ## remove the various types of string (since the starting piece is still there) ##
    return re.sub(r"r\"|r'|b\"|b'|f\"|f'|\"|'"," str ",new_section)

def get_variables(code: str) -> list[str]:
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
    if len(matches)>0:
        raise SyntaxError(f"""The following syntaxes are not allowed as they will not execute: {matches}

Cannot have i.e. 1.method() but you can have (1).method() e.g. for int types""")
    # get unique names
    variables=set(sub.get.split(" "))
    # filter to identifier and non keywords only with builtins removed
    builtins=get_builtins()
    return [i for i in variables if (i.isidentifier() == True and iskeyword(i) == False and i not in builtins) or "." in i]

def str_ascii(obj: str | list[int,...]) -> list[int,...] | str:
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
    if type(obj) == str:
        return list(obj.encode("ascii"))
    return bytes(obj).decode()

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
    if isinstance(keep,list)==False:
        keep=[keep]
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
        if FUNC.__name__ not in SOURCE_CODES:
            SOURCE_CODES[FUNC.__name__]={"original":source}
        set_source=SOURCE_CODES[FUNC.__name__]
        if key == "" or key == None:
            set_source["new"]=head_body
        else:
            set_source[key]=head_body
        # make sure the functions are defined in local scope
        for func in keep:
            locals()[func.__name__]=func
        # redefine function
        if inplace==True:
            return exec(head_body,globals())
        exec(head_body)
        return locals()[FUNC.__name__]
    return FUNC

def wrap(FUNC: Callable,*wrap_args,**wrap_kwargs) -> Callable:
    """
    Decorator for wrapping functions with other functions
    i.e.
    
    def do2(string):
        return str(string)

    @wrap(do2)
    def do(string):
        return int(string)

    do(3.0)
    # should print
    # '3'
    """
    def wrap_wrapper(func: Callable): # function to wrap
        @wraps(func) # transfers the docstring since the functions redefined as the wrapper
        def wrapper(*args,**kwargs) -> None: # its args
            return FUNC(func(*args,**kwargs),*wrap_args,**wrap_kwargs)
        return wrapper
    return wrap_wrapper

def user_yield_wrapper(FUNC: Callable) -> Callable: # test
    """wrapper for the user_yield function"""
    @wraps(FUNC) # only needed for the function taken in not the args
    def wrapper(func: Callable,*args,**kwargs) -> None: # *desired function*
        return user_yield(FUNC(func)(*args,**kwargs))
    return wrapper

def user_yield(gen: iter) -> None:
    """For user interactive yielding"""
    if has_IPython():
        clear_display=clear_output
    else:
        clear_display=lambda : print("\033[H\033[J",end="")
    while True:
        try:gen_locals=next(gen)
        except StopIteration:break
        user_input=input(": ")
        if user_input.lower() == "break":
            break
        elif user_input.strip() == "cls":
            clear_display()
        elif user_input[:8] == "locals()":
            while True:
                if user_input[:8] == "locals()":
                    if len(user_input) < 9:
                        print(gen_locals)
                    elif user_input[8]+user_input[-1] == "[]":
                        try:
                            exec("temp=gen_locals"+user_input[8:])
                            print(locals()["temp"])
                        except:
                            None
                user_input=input(": ")
                if user_input.lower() == "break":
                    break
                elif user_input.strip() == "cls":
                    clear_display()

def slice_occ(string: str,occurance: str,times: int=1) -> str:
    """
    slices a string at an occurance after a specified number of times occuring
    """
    count=1
    for i in range(len(string)):
        if string[i] == occurance:
            if count == times:
                return string[:i],string[i:]
            count+=1
    return string

def source_code(FUNC: Callable,join: bool=True,key: str="original") -> (str,str,str):
    """
    my function for breaking up source code
    (will further develop later once I've
    figured out how to resolve some issues)
    
    If def do():pass then you have to modify
    the string first else it doesn't work
    
    Also true for 
    def do(a,
           b,
           c):
           
    key="original","new", or other custom specified key available
    """
    try:
        source=getsource(FUNC)
    except:
        try:
            global SOURCE_CODES
            source=SOURCE_CODES[FUNC.__name__]
        except:
            raise Exception("source code not found")
        try:
            source=source[key]
        except:
            raise Exception(f"source code not found at key '{key}' but the original source code may exist i.e. try 'original' as key value")
    if join == True:
        return source
    head_body=source
    if source[:4]!="def ":
        head_body=re.sub(r"@(.+?)\n","",source)
    diff=len(source)-len(head_body)
    decorators,head,body=source[:diff],*slice_occ(head_body,"\n")
    doc_string=""
    # temporarily remove docstring if it exists
    if FUNC.__doc__ != None:
        doc_string=f'\n    """{FUNC.__doc__}"""'
        body=re.sub(r'"""(.+?)"""|\'\'\'(.+?)\'\'\'',"",body, count=1,flags=re.DOTALL) # re.DOTALL incase of new-lines
    return decorators,head,doc_string,body

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
    if has_IPython():
        printing=lambda code:f"display(Code('{code}', language='python'))"
    else:
        printing=lambda code:f"print('{code}')"
    for indx,line in enumerate(body_lines):
        # ensure lines are indented accordingly
        if indx < length-1:
            indentation=get_indents(body_lines[indx+1])
        else:
            indentation=get_indents(line)
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
    # we should be able to unstr everything
    tup=unstr(unstr(string_data))
    # convert to pandas dataframe
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
    if reload==True:
        return refresh()
    generate_cell_ids()
    generate_exec_ids()

def refresh() -> None:
    """Refreshes jupyter notebook; it will save the current page as well"""
    display(Javascript("Jupyter.notebook.save_checkpoint();window.onbeforeunload=null;location.reload();"))

def dynamic_js_wrapper(func: Callable[bool,...]) -> None:
    """wrapper function for dynamically created javascript functions to save code"""
    @wraps(func)
    def wrapper(reload: bool=False)->None:
        if reload:
            return refresh()
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
def generate_exec_ids(reload: bool=False) -> None:
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
def generate_cell_ids(reload: bool=False) -> None:
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
    if len(file.split(".")[0]) == 0:
        raise Exception(f"Invalid filename: '{file}'")
    if file[-3:] == ".js":
        file = file[:-3]
    return file

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
    if get_ipython() == None:
        return False
    else:
        return True

def list_loop(ls: Any,FUNC: Callable=lambda x:x) -> list[Any]|Any:
    """
    loops through a list of elements applying some function to each element
    """
    if type(ls) != list:
        ls=[ls]
    returns=[]
    for item in ls:
        returns+=[FUNC(item)]
    if len(returns) == 1:
        return returns[0]
    return returns

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
    if clear==False:
        display(prompt+response)
    return response

class input_ext:
    """extension to the input function for inputting and/or validating more than one prompt"""
    def __init__(self,prompts: Any="") -> None:
        """gets the prompts"""
        self.prompts=prompts
        
    def loop(self) -> list[Any]|Any:
        """goes through each prompt for inputs"""
        return list_loop(self.prompts,input)

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
            if response == "n":
                continue
        process=subprocess.run("pip install "+package,capture_output=True)
        if process.returncode != 0:
            print("Error with package: "+package+"\n\n")
            print(process.stdout.decode("utf-8"))
            print("-"*20)
        else:
            print(package+" successfully installed\n")

def all_packs() -> pd.DataFrame:
    """Retrieves all packages and returns as a pd.DataFrame"""
    ls=subprocess.run("pip list",capture_output=True).stdout.decode("utf-8")
    ls = ls.split("\r\n")
    ls = [ls[0]]+ls[2:]
    ls = [re.split(r"\s{2,}",i) for i in ls]
    return pd.DataFrame(ls[1:],columns=ls[0])

def get_functions(code: str) -> (list[int],list[int]):
    """finds function definitions in strings"""
    # get starting indexes for all 'def ' statements made
    indexes=line_sep(code,"def ",exact_index=True)
    # get ending indexes for the end of the function
    end=[]
    for start in indexes:
        temp=code[start:]
        for indx in range(len(temp)):
            if temp[indx] == "\n":
                temp2=temp[indx:indx+6]
                if len(temp2[:5].strip()) != 0 and len(temp2[-1:].strip()) != 0 or len(temp2) < 6:
                    end+=[start+indx] # since it's relative
                    break
    return indexes,end

def to_module(code: str) -> Callable[..., Any]:
    """
    converts a string to a python module object that associates with a temporary file for getting source code
    # code reference: Sottile, A., (2020) https://stackoverflow.com/questions/64925104/inspect-getsource-from-a-function-defined-in-a-string-s-def-f-return-5,CC BY-SA,
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

def inherit(class_name: object,*args: object) -> object:
    """Adds inheritence to a choosen classname. This works for nested classes as well"""
    # get the names
    def get_name(x):
        """gets the literal name of the variable that's passed in"""
        return x.split("__main__.")[1][:-2]
    name = get_name(str(class_name))
    cls=name+","
    for arg in args:
        cls+=get_name(str(arg))+","
    # define / redefine class with inheritance
    exec(compile(f"class temp(*("+cls[:-1]+")):pass","","exec"))
    return locals()["temp"] # works

@property
class str_df:
    """String accessor extension for pd.DataFrame"""
    def __init__(self,df: pd.DataFrame) -> None:
        self.__df = df # save df as private variable

    # use df in methods
    def __getitem__(self,index: list) -> pd.DataFrame:
        return self.__df.map(lambda x:x[index])

    def split(self,sep: str=None) -> pd.DataFrame:
        return self.__df.map(lambda x:x.split(sep))

pd.DataFrame.str=str_df # we use @property else it expects df which would be cumbersome #

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
    with open("requirements.txt","w") as file:
        file.write(required)
    print("done")

def read_ipynb(filename: str,join: bool=False) -> list[str]|str:
    """readlines a jupyter notebook"""
    if filename[-6:] != ".ipynb":
        filename+=".ipynb"
    with open(filename, "rb") as file:
        lines = json.load(file)
    ls=[]
    for cell in lines["cells"]:
        lines=cell["source"]
        if len(lines) == 0:
            ls+=[""]
        else:
            ls+=[line[:-1] for line in lines[:-1]]+[lines[-1]]
    if join == True:
        return "\n".join(ls)
    return ls

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
    try:
        current_dir=os.getcwd()
        if setup:
            print("---setup configs---")
            if defaults == True:
                # get the default setup directory correct
                default_config = "\\".join(getfile(install).split("\\")[:-1])+"\\"+default_config
                configs = dict(pd.read_pickle(default_config))
                configs["description"] = "'"+input("description: ")+"'"
            else:
                configs = {"version":"","description":"","author":"","email":""}
                for i in configs.keys():
                    configs[i] = "'"+input(i+": ")+"'"
            requirements = []
            # search for requirements
            if get_requirements == True:
                requirements = req_search(directory)
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
            with open(directory+"setup.py","w") as file:
                file.write(setup_content)
            print("Successfully created setup files __init__.py and setup.py.")
        print("\ninstalling "+library_name+"...")
        # go to the directory then run the command
        if directory != "":
            os.chdir(directory)
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
        os.chdir(current_dir)
    except Exception as e:
        os.chdir(current_dir)
        print(e)

def uninstall(library_name: str,keep_setup: bool=True) -> None:
    """
    Runs pip uninstall library_name 
    with the option of removing the
    setup files stuff as well.
    
    also note that with keep_setup = False
    the __init__.py,setup.py, files and .egg-info 
    directory will be removed
    """
    # get the libraries directory
    print("getting package directory...",end="\r")
    try:
        directory = __import__(library_name).__file__
        directory = "\\".join(directory.split("\\")[:-1])+"\\"
    except:
        # if the name of the package is different to the name of the module
        process=subprocess.run("pip show "+library_name,capture_output=True)
        if process.returncode != 0:
            print(process.stdout.decode("utf-8"))
            return
        df=pd.Series(process.stdout.decode("utf-8").split("\r\n")).str.split(":")
        df.index=df.str[0]
        df=df.str[1:]
        def joinup(x):
            if type(x)==list:
                return ":".join(x)
            return x
        # directory
        directory=df.apply(joinup)["Location"].strip()+"\\"
    if keep_setup == False:
        print("uninstalling "+library_name+" and removing setup files")
    else:
        print("uninstalling "+library_name)
    # you have to add yes due to no request coming back; else nothing happens
    try:
        subprocess.run("pip uninstall "+library_name+" --yes")
        if keep_setup == False:
            # remove the __init__ and setup files
            os.remove(directory+"__init__.py")
            print("removed __init__.py")
            os.remove(directory+"setup.py")
            print("removed setup.py")
            shutil.rmtree(directory+library_name+".egg-info")
            print("removed "+library_name+".egg-info")
    except Exception as e:
        print(e)
        print("if the module name and directory is correct you may need to restart the kernel if the import doesn't work")

def git_clone(url: list[str]|str=[],directory: list[str]|str=[],repo: bool=True) -> None:
    """For cloning multiple github repos or files into choosen directories"""
    if type(directory) == str:
        directory=[directory]
    if type(url) == str:
        url=[url]
    dir_length=len(directory)
    url_length=len(url)
    if dir_length == 1:
        directory=directory*url_length
    elif url_length == 1:
        url=url*dir_length
    elif url_length != dir_length:
        raise Exception("Length mismatch between the number of supplied urls: "+str(url_length)+", and directories: "+str(dir_length))
    if repo == True:
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
    else:
        for file,path in zip(url,directory):
            try:
                open(path+file.split("/")[-1], "wb").write(requests.get(file).content)
            except Exception as e:
                print("Failed retrieving "+file+":\n\n")
                print(e)

def unstr(x: str) -> Any:
    """
    Allows simple conversion of a string of a type to its type
    globals()[x] gets deleted to free the memory so that all
    variables within this function are strictly only temporary.
    """
    exec("temp="+x)
    return locals()["temp"]

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
    indx=0
    for assignment in assignments:
        if operator in assignment:
            # split on operator and wrap with function
            assignments[indx] = FUNC.__name__+"("+line_sep(assignment,operator,sep=",")+")"
        indx+=1
    if len(assignments) > 1:
        return "=".join(assignments)
    return assignments[0]

def indx_split(indx: list[int]=[],string: str="") -> list[str]:
    """Allows splitting of strings via indices"""
    return [string[start:end] for start,end in zip(indx, indx[1:]+[None])]

def line_sep(string: str,op: str,sep: str="",split: bool=True,index: bool=False,exact_index: bool=False,avoid :str="\"'") -> list[int]|str|list[str]:
    """separates lines in code by op avoiding op in strings"""
    in_string=0
    indx=0
    req=len(op)
    count=0
    ls=list(string)
    current_str=""
    if sep == "":
        ls2=[]
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
        if char == " ":
            n_white_space+=1
        else:
            break
    if n_white_space % 4 != 0:
        raise Exception("indentations must be 4 spaces to be valid:"+line)
    return " "*n_white_space

def bracket_up(string: str,start :str="(",end: str=")",avoid: str="\"'") -> pd.DataFrame:
    """ For pairing bracket indexes """
    indx=0
    left=[]
    ls=[]
    in_string=0
    current_str=""
    for i in string:
        if i in start:
            left+=[indx]
        elif i in end:
            ls+=[[left[-1],indx,in_string,len(left)-1]]
            left=left[:-1]
        if i in avoid:
            if i == current_str or current_str == "":
                in_string=(in_string+1) % 2
                if current_str !="" and in_string == 0:
                    current_str=""
                else:
                    current_str=i
        indx+=1
    return pd.DataFrame(ls,columns=["start","end","in_string","encapsulation_number"])

### needs testing but seems okay ###
def func_dict(string: str) -> str:
    """Turns a i.e. (a=3,b=2) into {"a":3,"b":2} """

    def dict_format(temp: int,commas: pd.Series|pd.DataFrame,section: list,old_section_length: int,adjust: int) -> (list,int):
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
            if hasattr(unstr(temp),"__call__") == True:
                temp_check = ls[i+1]
                if temp_check[0] == "{" or temp_check[0:2] == "*{" or temp_check[0:3]+" " == "**{":
                    ls_new+=[temp+","+asterisk]
                    continue
        except:
            None
        # we can't use temp because the except statement will keep the modifications up to the exception
        ls_new+=[ls[i]]
    return "".join(ls_new)

####### not sure if this works... will check later ##############
def line_enclose(string,start,end,FUNC="",sep="",separate=False):
    start=line_sep(string,start,index=True)
    end=line_sep(string,end,index=True)
    enclosing=[]
    if separate == True or sep!= "":
        ls=list(string)
        for i in range(len(start)//2):
            ls[start[2*i]:end[2*i+1]] = sep
        return "".join(ls)
    if FUNC != "":
        for i in range(len(start)//2):
            enclosing+=[[start[2*i],end[2*i+1]]]
        return FUNC(string,enclosing)
    for i in range(len(start)//2):
        enclosing+=[start[2*i],end[2*i+1]]
    return indx_split([0]+enclosing,string)
###################################################################

### seems to work but needs testing ###
def interpret(code: str,checks: list[Callable]=[],operators: str=[]) -> str:
    """Checks code against a custom set of rules for how code should be formated"""
    # to allow for multi-line open bracket expressions
    df=bracket_up(code,start="({[",end="]})")
    df=df[(df["encapsulation_number"] == 0) & (df["in_string"] == 0)]
    # remove newline characters in multiline open brackets
    lines=list(code)
    old_length = len(lines)
    adjust=0
    for start,end in zip(df["start"],df["end"]):
        lines[start-adjust:end-adjust] = line_sep("".join(lines[start-adjust:end-adjust]),"\n",split=False)
        adjust = old_length-len(lines)
    code="".join(lines)
    # make sure they're lists
    if type(checks) != list:
        checks=[checks]
    if type(operators) != list:
        operators=[operators]
    # \n in strings becomes \\n, so you can split this way
    # line_sep takes care of the ';' occurances and comments
    lines=[]
    # get lines
    for line in line_sep(code,";","\n").split("\n"):
        # remove comments
        temp=line_sep(line,"#")[0]
        # remove spaces
        if len(temp) > 0:
            lines+=[temp]
    # go through each line checking through the operators
    for check,operator in zip(checks,operators):
        indx=0
        for line in lines:
            if operator in line:
                # get the number of indentations at the start of the line
                lines[indx]=get_indents(line)+prep(line,check,operator)
            indx+=1
    # join lines back together
    try:
        return "\n".join(lines)
    except:
        return lines[0]

def str_anti_join(string1: str,string2: str) -> str:
    """anti_joins strings sequentially e.g. assuming appendment"""
    diff=len(string2) - len(string1)
    if diff != 0:
        string1=list(string1)
        string2=list(string2)
        if diff > 0:    
            string1=string1+["__"]*diff # since strings should list into single characters
            temp=string2
        else:
            string2=string2+["__"]*diff
            temp=string1
    indx=0
    for s1,s2 in zip(string1,string2):
        if s1 == s2:
            temp[indx]=""
        indx+=1
    return "".join(temp)

## background processing
def create_variable(name: str) -> None:
    globals()[name] = []

standing_by_count=0
def stand_by() -> None:
    global standing_by_count
    sleep(1)
    print(standing_by_count)
    standing_by_count+=1   

def stop_process(ID: int) -> None:
    globals()["process "+str(ID)]=True

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

try:current_execution=get_ipython().__getstate__()["_trait_values"]["execution_count"]
except:None
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
    partitions = [0]
    for i in range(1,number_of_subsets+1):
        partitions += [int(np.floor(i*interval/number_of_subsets))]
    interval_partitions = []
    for i in range(len(partitions)-1):
        interval_partitions += [[partitions[i],partitions[i+1]]]
    return interval_partitions

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
        while thread_count != number_of_threads:
            continue
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
                try:
                    FUNC(globals()[variable],i)
                except Exception as e:
                    print("error at i =",i,":",e)
                    errors+=[i]
        else:
            try:
                FUNC(i)
            except Exception as e:
                print("error at i =",i,":",e)
                errors+=[i]
        with lock:
            thread_count+=1 # tells the multi-threader that a thread's available

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
    for _ in range(n):
        next(iter_val)

def argmiss(returns: dict|tuple,temp: dict|tuple,func: Callable) -> Callable[Any,...]:
    """
    Sorts out the combinations of missing arguements
    
    Note: it will not work if returns takes up the arguements
    that temp takes up as temp will default to the first 
    available arguements and therefore the first dict can only use
    args those of temps' else it doesn't work.
    """
    # gather the next arguement/s
    if type(returns) == dict:
        if type(temp) == dict:
            return func(**returns|temp)
    
        elif type(temp) == tuple:
            return func(*temp,**returns)
        return func(temp,**returns)
    # it must be a tuple
    else:
        if type(temp) == dict:
            return func(*returns,**temp)
        elif type(temp) == tuple:
            return func(*(*returns,*temp))
        return func(*(*returns,temp))

def pipe(*args,reverse: bool=False) -> Callable[Any,...]:
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
        try:
            returns = func(*returns)
        except (TypeError, ValueError):
            returns = argmiss(returns,args[i+1*scalar],func)
            skip(iter_val,1)
    if REVERSE_PIPE == "True":
        os.environ["REVERSE_PIPE"] = "False"
    return returns

def standardize(x: pd.Series|pd.DataFrame) -> pd.DataFrame:
    """standardize to pd.Dataframe directly"""
    return pd.DataFrame(StandardScaler().fit_transform(x),columns=x.columns)

def Type(df: pd.DataFrame=[],ls: list[str]=[],keep: list[str]=[],this: bool=True,show: bool=False) -> pd.DataFrame:
    """
    Takes in (usually) a dataframe and what datatypes you want/don't want (this=True/False),
    Allowing you to keep columns using keep=[], and allowing you to see all current datatypes using show=True
    """
    if show == True:
        return pd.DataFrame([[i,df[i].dtype] for i in df.columns],columns=["column","dtype"]).set_index("column")
    try:
        keep = df[keep]
    except:
        keep = pd.Series([])
    for string in ls:
        df = df[[i for i in df.columns if (df[i].dtype == string) == this]]
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

def multi_cprop(df: pd.DataFrame,cols: list) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame):
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

def cprop(df: pd.DataFrame,occurance: pd.DataFrame) -> (pd.DataFrame,pd.DataFrame):
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
        try:
            encoding["index"] = to
        except:
            print("mapping length required: ",len(encoding)," (",len(encoding)-len(to),"more elements needed )")
            return encoding
    encoding = data.reset_index().merge(encoding, left_on=data.name, right_on=data.name).iloc[:,2]
    encoding.name = data.name
    return encoding

def pairs_matrix(ls1: list,ls2: list) -> pd.DataFrame:
    """Forms a matrix from two lists"""
    df = pd.DataFrame()
    ls2 = pd.Series(ls2,index=ls2)
    for i in ls1:
        df[i] = i+ls2
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
    length=len(lst)
    new_lst_1=[]
    new_lst_2=[]
    new_lst_3=[]
    temp_length=length
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

def str_split(x: str,string: str) -> list:
    return x.split(string)

def run_r_command(command: str,R: str='C:/Program Files/R/R-4.3.1/bin/R.exe') -> str:
    """
    Returns R terminal output for python interface 
    
    (If used elsewhere consider changing the file location hardcoded in this module e.g. where you launch the R terminal)
    """
    # trial and error got to this
    process = subprocess.run([R, '--vanilla'], input=command.encode(), capture_output=True)
    output = process.stdout.decode('utf-8')
    if process.returncode == 0:
        return output[716:-5] # removes the initial stuff you get on first load
    print(output)

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
    if axis==0:
        # if axis = 0 # retain the cols but the rows won't make sense
        return pd.DataFrame([[list(data.loc[:,i]) for i in data.columns]],columns=data.columns)
    else:
        # if axis = 1 # retain the rows but the cols won't make sense
        return pd.DataFrame({0:[list(data.loc[i,:]) for i in data.index]},index=data.index)

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
    if response.status_code != 200:
        return print("Error: ",response.status_code," ",response)
    # maybe export this out to a function that deals with form?
    if form == None:
        return response.content
    elif form == 'html':
        return BeautifulSoup(response.content, "lxml")
    elif form == 'json':
        return json.loads(response.content)
    return response.content

def render(html: str,head: str=':None') -> None:
    """
    Formats Beautifulsoup, list, or string to html for display
    
    head=':None' by default but can be set to any range in the form of a string i.e. '1:2'
    
    The function will use Beautiful soup to complete unclosed html tags to ensure the output won't affect the rest of the jupyter notebook
    """
    # convert to string for slicing
    if isinstance(html,str) == False:
        html = str(html[0])
    ### allow for 'head' to be directly inserted for slicing, Markdown reuquires strings, 
    ### Beautiful soup will close tags to stop it affecting the rest of the note book (else i.e. successive divs are nested as children) 
    exec(compile('display(Markdown(str(BeautifulSoup(html['+head+'], "lxml"))))', '', 'exec'))

def handle_file(name: str,form: str='utf8',FUNC: Callable=type,head: str='') -> str:
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
        if head == None or isinstance(head,int) == True:
            print(file.read()[:head],"\n")
        if FUNC != type:
            returns = FUNC(file)
        else:
            returns = file.read()
    return returns

def my_info(data: pd.DataFrame) -> pd.DataFrame:
    """My version of .info()"""
    return pd.concat([data.count(),(np.round(data.isnull().mean()*100,2)).astype(str)+"%",data.apply(lambda x: x.dtype)],axis=1).rename(columns={0:"Non-Null count",1:"Pct missing (2 d.p.)",2:"dtype"})

def data_sets(data: pd.DataFrame) -> pd.DataFrame:
    """Returns the unique categories"""
    data_cats = data.agg(pd.unique) # can use set as well
    data_cats = pd.DataFrame(data_cats).T
    return data_cats.apply(pd.Series.explode, ignore_index=True)

def try_except(trying: Any,exception: Any) -> Any:
    """
    Functional form of try-except useful for lambda expressions (but may not always work)
    """
    try:return trying
    except:return exception

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

# my function to go through each cell of a column comparing to each other cell
def shift_check(str1: str,str2: str) -> list:
    """
    Takes two strings and compares the amount shift that one string has to take to become the other whilst
    removing characters from the string being compared to ensuring no duplicates
    """
    # which is the smaller and bigger string
    if len(str1) <= len(str2):
        str_smaller = str1 
        str_bigger = str2
    else:
        str_smaller = str2 
        str_bigger = str1
    new_ls = []
    for i in range(len(str_smaller)):
        new_ls+=[0]
        j = 0
        # calculates the shift
        while j < len(str_bigger) and str_smaller[i] != str_bigger[j]:
            new_ls[i]+= 1
            j+=1
        # reduce the bigger string
        if j == len(str_bigger):
            new_ls[i] = np.nan
        elif str_smaller[i] == str_bigger[j]:
            try:
                str_bigger = str_bigger[:j]+str_bigger[j+1:]
            except:
                str_bigger = str_bigger[:j-1]+str_bigger[j:]
    return new_ls

def sim_check(data_: pd.Series,mean: float=10,std: float=10,j_thr: float=0.75,l_thr: int=3,limit: int=200,dis: bool=False) -> None:# make sure all a,b,c are before d=1,e=2,f=3
    """ 
    similarity check used on pd.Series objects to check text data.
    """
    count=0
    data=data_.dropna()
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
                            print("limit reached: "+str(limit)+" line/s")
                            return
                # overall similarity
                elif jaccard_similarity(set(base), set(temp)) > j_thr:
                    # enhanced similarity for parts of strings
                    check = shift_check(base,temp)
                    if (np.nanmean(check) < mean) & (np.nanstd(check) < std): # there are other stats we can try but the std is OK
                        print(base,temp)
                        count+=1
                        if count == limit:
                            print("limit reached: "+str(limit)+" line/s")
                            return
        data = data[data.isin([base]) == False] # to reduce the uneccessary combinations / only get unique ones

def str_strip(x: str) -> str:
    if isinstance(x,str) == True:
        return x.strip() 
    else: return x

## how to create methods to existing classes in python:

#FUNC.indicator_encode=indicator_encode
#pd.DataFrame.Type=Type
#pd.Series.indicator_encode=indicator_encode
#pd.DataFrame.dupes=dupes
#pd.DataFrame.full_sim_check=full_sim_check
#pd.DataFrame.gather=gather

## Piping:
#from my_pack import pipe
#pipe(a,b,c,...)

#or in jupyter notebook (IPython allows this (I think?))

#/var=pipe a,b,c,...
#var
