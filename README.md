# Series Distance Matrix 

This library implements the [Series Distance Matrix framework](https://doi.org/10.1016/j.engappai.2020.103487),
a flexible component-based framework that bundles various [Matrix Profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
related techniques.
These techniques can be used for (time) series mining and analysis. 
Some example applications include:
- motif discovery: finding the best (imperfect) matching subsequence pair in a larger series
- discord discovery: finding the most dissimilar subsequence in a larger series
- finding repeating subsequences in one or more series (common and consensus motifs)
- visualizing series
- finding changing patterns
- ...

The **Series Distance Matrix** framework was designed to integrate the various
Matrix Profile variants that were established over the years.
It does this by splitting the generation and consumption of
the all-pair subsequence distances,
putting the focus on the distance matrix itself.
This allows for easier and more flexible experiments by
freely combining components and eliminates the need
to re-implement algorithms to combine techniques in an efficient way.


Following core techniques are implemented:
- Z-normalized Euclidean distance (including noise elimination)
- Euclidean distance
- (Left/Right) Matrix Profile
- Multidimensional Matrix Profile
- Contextual Matrix Profile
- Radius Profile
- Streaming and batch calculation


Following Matrix Profile related techniques are implemented:
- Valmod: find the top-1 motif in a series for each subsequence length in a given range
- Ostinato: find the top-1 (k of n) consensus motif in a collection of series
- Anytime Ostinato: find the radius profile for a collection of series


## Basic Usage

Calculate a standard Matrix Profile using z-normalized Euclidean distance over a single series.

```python
import numpy as np
from distancematrix.generator.znorm_euclidean import ZNormEuclidean
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLR
from distancematrix.calculator import AnytimeCalculator

data = np.random.randn(10000)
m = 100 # Subsequence length

calc = AnytimeCalculator(m, data)
gen_0 = calc.add_generator(0, ZNormEuclidean())
cons_mp = calc.add_consumer([0], MatrixProfileLR())
calc.calculate_columns()

matrix_profile = cons_mp.matrix_profile()
```

Calculate a Matrix Profile and (common-10) Radius Profile over a single series using Euclidean distance.
A combined calculation is more efficient, as it can reuse the calculated distances.

```python
import numpy as np
from distancematrix.generator.euclidean import Euclidean
from distancematrix.consumer.radius_profile import RadiusProfile
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLR
from distancematrix.calculator import AnytimeCalculator

data = np.random.randn(10000)
m = 100 # Subsequence length

calc = AnytimeCalculator(m, data)
gen_0 = calc.add_generator(0, Euclidean()) # Generator 0 works on channel 0
cons_mp = calc.add_consumer([0], MatrixProfileLR()) # Consumer consumes generator 0
cons_rp = calc.add_consumer([0], RadiusProfile(10, m//2)) # Consumer consumes generator 0
calc.calculate_columns()

matrix_profile = cons_mp.matrix_profile()
radius_profile = cons_rp.values
```

Calculate a partial multidimensional Matrix Profile over two data channels.
Partial calculations return approximated results but are significantly faster,
they are particularly interesting in interactive workflows, as they can be resumed.

```python
import numpy as np
from distancematrix.generator.znorm_euclidean import ZNormEuclidean
from distancematrix.consumer.multidimensional_matrix_profile_lr import MultidimensionalMatrixProfileLR
from distancematrix.consumer.matrix_profile_lr import MatrixProfileLR
from distancematrix.calculator import AnytimeCalculator

data = np.random.randn(2, 10000)
m = 100 # Subsequence length

calc = AnytimeCalculator(m, data)
gen_0 = calc.add_generator(0, ZNormEuclidean()) # Generator 0 works on channel 0
gen_1 = calc.add_generator(1, ZNormEuclidean()) # Generator 1 works on channel 1
cons_mmp = calc.add_consumer([0, 1], MultidimensionalMatrixProfileLR()) # Consumer consumes generator 0 & 1

# Calculate only 1/4 of all distances: faster, but returns approximated results
calc.calculate_diagonals(partial=0.25)
multidimensional_matrix_profile = cons_mmp.md_matrix_profile()

# Calculate the next quarter, so in total 1/2 of all distances are processed.
calc.calculate_diagonals(partial=0.5)
multidimensional_matrix_profile = cons_mmp.md_matrix_profile()
```

## Documentation

Documentation for the latest version is available [online](https://predict-idlab.github.io/seriesdistancematrix).

Building the documentation locally is done using Sphinx. Navigate to the `docs` folder, activate the conda environment
defined in the environment file, and run:

```bash
make html
```

## Installing

Using pip:
```bash
pip install seriesdistancematrix
```

Alternatively, clone this repositor and run:
```bash
python setup.py clean build install
```

For local development (this allows you to edit code without having to reinstall the library):
```bash
python setup.py develop
```

## Academic Usage

When using this library for academic purposes, please cite:
```
@article{series_distance_matrix,
  title = "A generalized matrix profile framework with support for contextual series analysis",
  journal = "Engineering Applications of Artificial Intelligence",
  volume = "90",
  pages = "103487",
  year = "2020",
  issn = "0952-1976",
  doi = "https://doi.org/10.1016/j.engappai.2020.103487",
  url = "http://www.sciencedirect.com/science/article/pii/S0952197620300087",
  author = "De Paepe, Dieter and Vanden Hautte, Sander and Steenwinckel, Bram and De Turck, Filip and Ongenae, Femke and Janssens, Olivier and Van Hoecke, Sofie"
}
```
