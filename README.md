# Hierarchical sparse coding (HSC)
Hierarchical sparse coding using greedy matching pursuit.

![alt tag](https://github.com/sbrodeur/hierarchical-sparse-coding/raw/master/docs/images/algorithm.png)

S. Brodeur and J. Rouat, “Optimality of inference in distributed hierarchical coding for object-based representations,” in 15th Canadian Workshop on Information Theory (CWIT), 2017.

## Dependencies

Main requirements:
- Python 2.7 with Numpy, Scipy and Matplotlib

## Installing the library

Download the source code from the git repository:
```
mkdir -p $HOME/work
cd $HOME/work
git clone https://github.com/sbrodeur/hierarchical-sparse-coding.git
```

Note that the library must be in the PYTHONPATH environment variable for Python to be able to find it:
```
export PYTHONPATH=$HOME/work/hierarchical-sparse-coding:$PYTHONPATH 
```
This can also be added at the end of the configuration file $HOME/.bashrc

## Running unit tests

To ensure all libraries where correctly installed, it is advised to run the test suite:
```
cd $HOME/work/hierarchical-sparse-coding/tests
./run_tests.sh
```
Note that this can take some time.

## Running experiments

To reproduce the experiments of the paper: 
```
cd $HOME/work/hierarchical-sparse-coding/scripts
./run_experiments.sh
```
Note that this can take some time.


## Citation

Please cite Hierarchical sparse coding (HSC) algorithm in publications when used:
> S. Brodeur and J. Rouat, “Optimality of inference in distributed hierarchical coding for object-based representations,” in 15th Canadian Workshop on Information Theory (CWIT), 2017.

BibTeX entry for LaTeX:
```
@INPROCEEDINGS{Brod1706:Optimality,
AUTHOR="Simon Brodeur and Jean Rouat",
TITLE="Optimality of inference in distributed hierarchical coding for object-based
representations",
BOOKTITLE="15th Canadian Workshop on Information Theory (CWIT)",
ADDRESS="Quebec city, Canada",
DAYS=11,
MONTH=jun,
YEAR=2017,
KEYWORDS="Unsupervised learning; Data compression; Inference mechanisms; Greedy
algorithms; Sparse matrices",
ABSTRACT="Hierarchical approaches for representation learning have the ability to
encode relevant features at multiple scales or levels of abstraction.
However, most hierarchical approaches exploit only the last level in the
hierarchy, or provide a multiscale representation that holds a significant
amount of redundancy. We argue that removing redundancy across the multiple
levels of abstraction is important for an efficient representation of
compositionality in object-based representations. With the perspective of
feature learning as a data compression operation, we propose a new greedy
inference algorithm for hierarchical sparse coding. Convolutional matching
pursuit with a L0-norm constraint was used to encode the input signal in a
distributed and non-redundant code across levels of the hierarchy. Simple
and complex synthetic datasets of temporal signals were created to evaluate
the encoding efficiency and compare with the theoretical lower bounds on
the information rate for those signals. The algorithm was able to infer
near-optimal distributed code for simple signals. However, it failed for
complex signals with strong overlapping between objects. We explain the
inefficiency of convolutional matching pursuit that occurred in such case.
This brings new insights about the NP-hard optimization problem related to
using L0-norm constraint in inferring optimally distributed and compact
object-based representations."
}
```
