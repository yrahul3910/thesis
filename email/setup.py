from setuptools import setup
import os

setup(name='ses',
      version='1.0.0',
      description='Amazon SES Wrapper',
      author='Rahul Yedida',
      author_email='ryedida@ncsu.edu',
      packages=[
          'ses'
      ],
      install_requires=[
          'boto3'
      ]
      )
