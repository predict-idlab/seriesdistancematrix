from setuptools import setup
from setuptools import find_packages

def readme():
    with open("README.md") as readme:
        return readme.read()

setup(name='seriesdistancematrix',
      version='0.3.1', # Also update distancematrix/__init__.py!
      description=(
        'Flexible time series analysis library'
        'implementing Matrix Profile related functionality.'
      ),
      long_description_content_type="text/markdown",
      long_description=readme(),
      keywords=[
        'time series',
        'matrix profile',
        'contextual matrix profile',
        'radius profile',
        'series distance matrix',
        'motif',
        'discord'
      ],
      url='https://github.com/predict-idlab/seriesdistancematrix/',
      project_urls={
        'Documentation': 'https://predict-idlab.github.io/seriesdistancematrix/',
        'Source': 'https://github.com/predict-idlab/seriesdistancematrix/'
      },
      author='Dieter De Paepe',
      author_email='dieter.depaepe@gmail.com',
      license='MIT',
      packages=find_packages(exclude=["distancematrix.tests*"]),
      classifiers=(
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent'
      ),
      install_requires=['numpy', 'scipy', 'pandas']
)
