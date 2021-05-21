from setuptools import setup, find_packages


setup(
    name='sandro_project',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    python_requires='>=3.7.2',
    install_requires=[
        'numpy>=1.18.1',
        'torch>=1.8.1',
        'matplotlib',
        'tqdm',
    ],
)

# run in terminal in a root folder:
# >>> pip install -e ./
# -e key stands for "editable"

# to clone to your local machine (do one time):
# make sure you have git installed (maybe try first PyCharm terminal).
# https://git-scm.com/downloads
# git clone https://github.com/Kantafan/sandro_project.git
# at this point, you have the latest version of your project on your local machine


# edit project in PyCharm on your local machine (always)
# you always have the latest version on your local machine.

# when you want to update it on the remote machine (blackbox3):
# First, push it to github.com (git push -u origin master)
# Second, pull it from github on your blackbox3 (git pull origin master)
# Check version (pip list | grep sandro-project)
# Restart jupyter kernel (click button restart)

# later in jupyter notebook
# from sandro_project import Scene, Agent, ...

# init git repo:
# git remote set-url origin https://github.com/Kantafan/sandro_project.git
# git push -u origin master

