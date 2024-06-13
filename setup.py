
from setuptools import setup, find_packages

setup(
    name='my_pack',  
    version='0.1.0',
    description='',
    author='author',
    author_email='email',
    packages=find_packages(),
    install_requires=['setuptools', 'matplotlib', 'inspect', 'pandas', 'sklearn', 'IPython', 'seaborn', 'bs4', 'requests', 'html5lib', 'numpy']
)
