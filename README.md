# my_pack
python functions I made for 297201 assignments and other stuff
# Note:
you may use any of the functions in here but I've used this at massey university so... it's possible you'll be plagarized if you don't at the very least cite the relevant sections from which you got what you used if you copy sections of the code here.
# pipe example
```python
from my_pack import pipe
def do(a,b):
    return a+b
/pipe 1,do,1
## should print:
# 2
import pandas as pd
/var=pipe [1,2,3],pd.DataFrame,do,pd.DataFrame([1,2,3])
var
## should print (but as a pd.DataFrame):
#   0
# 0 2
# 1 4
# 2 6
```
# multi_thread example
```python
from my_pack import multi_thread
def for_loop(variable,i):
    variable+=[i]
number_of_threads,interval_length = 3,10
multi_thread(number_of_threads,interval_length,for_loop,part=True)[0] # it returns results,errors as tuple
## should print:
#Threads complete
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
# how to retrieve:
in a jupyter notebook or python shell or otherwise run something similar to either of the following code:
```python
!git clone https://github.com/Benj1bear/my_pack
# or for the library file itself if wanted
import requests
file = requests.get("https://raw.githubusercontent.com/Benj1bear/my_pack/main/my_pack.py")
open("my_pack.py", "wb").write(file.content)
# or
#open("*alternative directory and filename desired*", "wb").write(file.content)
```
# how to get started
open the jupyter notebook and run the first cell containing with the following:
```python
from my_pack import install

# install the package so it can be used elsewhere
install("my_pack")
```
# install and uninstall
If you want to use libraries beyond one directory and in across many projects like you would with regular imports of libraries then you can install it using the install function. Note: this kind of install is editable meaning you can make saved changes to your library and it will update; though you'll need to reload the importing of it again.
```python
# either in jupyter notebook or python shell (after running install("my_pack") from within the same directory)
# in the same or different directory as the my_pack.py file run:
from my_pack import install
# if you're in a directory one above
#from my_pack.my_pack import #*desired function*

# once installed you can then install scripts from different directories,
install("another_pack","*the directory where it's at*")
```
Also, you can uninstall using the uninstall function. Setting keep_setup to False will remove the \_\_init\_\_.py and setup.py files along with the .egg-info folder but retain the original script. So uninstalling by itelf only cut soff the 'connection' but still leaves behind the setup files. Note: you should be able to apply the install and uninstall functions to any python script ideally.
```python
from my_pack import uninstall
uninstall("my_pack",keep_setup=False)
```
# changing file path
if you do this then you'll need to reinstall. 
i.e. if:

old path: c:/a/b/c/my_pack.py

new path: c:/a/c/d/my_pack.py

Then go into the new directory and run the second code block in the How to get started jupyter notebook that will reinstall without having to modify any of the files e.g. to 're-link' it. You don't need to run the uninstall function (I think).
```python
from my_pack import install
install("my_pack",setup=False)
```
