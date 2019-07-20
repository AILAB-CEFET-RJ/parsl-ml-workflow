import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='parsl-ml-workflow',
     version='0.1',
     author="Raphael Fialho",
     author_email="raphael.fialho@eic.cefet-rj.br",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )