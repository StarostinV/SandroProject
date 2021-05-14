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
# git clone https://github.com/someguy/somerepo

# later in jupyter notebook
# from sandro_project import Scene, Agent, ...
