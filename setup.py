from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='omda',
    version='0.1.0',
    packages=find_packages(),
    description='OpenMind Decison Analysis, a tool for decision analysis under the framework of OpenMindClub.',
    author='Kara Tsing-Yu',
    author_email='maiyunfei2000@gmail.com',
    url='https://github.com/karanotsingyu/omda',
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)