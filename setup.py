from setuptools import setup

setup(
    name='MetaFE',
    version='0.1.0',
    description='The best automated feature engineering method',
    url='https://github.com/schaeferbasti/MetaFE',
    author='Bastian Schäfer',
    author_email='schaefer.bastian@mail.de',
    license='',
    packages=['src'],
    install_requires=['pandas','numpy','autogluon',],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
    ],
)
