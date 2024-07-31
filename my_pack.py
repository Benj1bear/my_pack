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
#from types import ModuleType
from typing import Any,Callable
import tempfile
from importlib.util import module_from_spec,spec_from_loader
########### for installing editable local libraries and figuring out what modules python scripts use
import re
import glob
import shutil
from inspect import getfile,getsource
import sys
from functools import partial,wraps
from keyword import iskeyword

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
    FUNC=None
    if isinstance(section,Callable)==True:
        section,FUNC,source=source_code(section),section.__name__,section.__module__
    # prep section
    variables=get_variables(section)
    if len(variables) == 0:
        return section
    # gather all functions,classes available to the .py file
    callables=all_callables(source)
    if FUNC!=None:
        callables=[func for func in callables if func.__name__!=FUNC]
    # start exporting code
    if show:print("initial section:\n"+"-"*20+"\n"+section+"\n"+"-"*20)
    code_export=get_code_requirements(*(section,callables,variables,source,show),limit=recursion_limit) ## needs implementation
    if to==None:
        return code_export
    with open(to,option) as file:
        file.write(code_export)

def get_code_requirements(section: str,callables: list[str],variables: list[str],source: str,show: bool=False,recursions: int=0,limit: int=20) -> str:
    """Gets the required code in order to export a section of code from a .py file maintainably"""
    # callables, section, and variables can change
    changes=lambda string="before":print(string+":\n"+"-"*20+"\n"+section+"\n"+"-"*20) if show else None
    new_exports,remaining_callables=[],[]
    # get which functions 
    for func in callables:
        if (func.__name__ in variables)==True:
            new_exports+=[func]
        else:
            remaining_callables+=[func]
    if len(new_exports) > 0:
        for func in new_exports:# a list of functions from the module
            exec(f'temp=__import__("{source}").{func.__name__}')
            section+="\n"+source_code(locals()["temp"]) ########################## This needs to be replaced to deal with i.e. imports from modules
        changes(f"Recursion: {recursions}")
        # make sure there's some safety in case errors occur
        if recursions==limit:
            print(f"recursion limit '{limit}' reached\n\nNote: the function may not have completed, if true, adjust the recursion limit or enter in the current code section to continue")
            return section
        recursions+=1
        get_code_requirements(*(section,remaining_callables,get_variables(section),source,show,recursions))
    return section

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
    
    def __getitem__(self,index: slice) -> dict:
        if type(index) == slice:
            # get keys as ints if not already
            if type(index.start) == str or type(index.stop) == str:
                index=key_slice(self.dct,index)
            # numeric # convert to list
            return {i: self.dct[i] for i in list(self.dct)[index]}
        if type(index) != tuple:
            index=(index,)
        keys=list(self.dct.keys())
        condition=lambda i:i if type(i)==str else keys[i]
        return {condition(i): self.dct[condition(i)] for i in [*index]} # decided not to use set([*index]) to retain ordering

    def __setitem__(self,index,args) -> None:
        if type(args) != list and type(args) != tuple:
            args=[args]
        # get keys and set them
        dct=self.__getitem__(index)
        keys=list(dct.keys())
        # catch errors
        if len(keys)-len(args)!=0:
            raise Exception(f"in dct_ext.__setitem__: Mismatch between number of keys to set and arguements to be assigned\nkeys: {keys}\nargs: {args}")
        for key,arg in zip(keys,args):
            self.dct[key]=arg

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

def get_variables(code: str) -> list[str]:
    """
    Extract only variable names from strings
    
    Note: only works on raw strings e.g.
    the kind you may get from reading files
    else it won't remove strings correctly
    """
    def sub(regex: str,repl: str="") -> None:
        """For reducing the amount of code written"""
        nonlocal code # makes it accessible to a scope one above
        code=re.sub(regex,repl,code,flags=re.DOTALL)
    # make sure we can distinguish strings
    sub(r"\\\"|\\\'")
    ## in case methods are applied to string or int types
    ## keep string types
    sub(r"\'[^']*\'\.","str.")
    sub(r"\(\'[^']*\'\)\.","str.")
    sub(r'\"[^"]*\"\.',"str.")
    sub(r'\(\"[^"]*\"\)\.',"str.")
    ## keep float types
    sub(r"[-+]?\d+\.\d+\.","float.")
    sub(r"\([-+]?\d+\.\d+\)\.","float.")
    ## keep int types (they cannot be i.e. 1.to_bytes() only (1).to_bytes() else it expects a float)
    sub(r"\(\d+\)\.","int.")
    ## check for errors
    matches=re.findall(r"\d+\.",code)
    if len(matches)>0:
        raise SyntaxError(f"""The following syntaxes are not allowed as they will not execute: {matches}

(Cannot have i.e. 1.method() but you can have (1).method() e.g. for int types)""")
    # remove all strings
    sub(r"\'[^']*\'")
    sub(r'\"[^"]*\"')
    # remove all comments
    code+="\n"
    sub(r"#(.+?)\n"," ")
    # get letters and numbers only (retaining '.' to extract the base dictionary)
    sub(r"[^\w.]+"," ")
    # remove any spaces between attributes
    sub(r"\s*\.",".")
    # get unique names
    variables=set(code.split(" "))
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
        
def install(library_name: str,directory: str="",setup: bool=True,get_requirements: bool=True,defaults: bool=True,default_config: str="PYTHON_SETUP_DEFAULTS.pkl") -> None:
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
    """
    try:
        current_dir=os.getcwd()
        if setup == True:
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
            with open(directory+"__init__.py","w"):None
            with open(directory+"setup.py","w") as file:
                file.write(setup_content)
            print("Successfully created setup files __init__.py and setup.py.")
        print("\ninstalling "+library_name+"...")
        # go to the directory then run the command
        if directory != "":
            os.chdir(directory)
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

def preprocess(data: pd.DataFrame,file: str="",variable: str="") -> pd.DataFrame:
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
