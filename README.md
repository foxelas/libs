# libs
Common libs I am using

The latest version is in branch 'v1'.

## Contents

## Installation
### Local installation:
Install locally at a virtual environment with:
```
$ python -m pip install -e .
```
To clean up the egg files:
```
$ python setup.py clean     
```

### Make requirements file: 
```
pipreqs --force --encoding=utf8 .
```

### Add package requirement: 
In the requirements.txt add the following line: 

```
libs @ git+https://github.com/foxelas/libs@v1
```

