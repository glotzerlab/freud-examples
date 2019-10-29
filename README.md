[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)

# freud examples

Welcome to the freud example scripts.
These Jupyter notebooks demonstrate how to utilize the functionality of [freud](http://glotzerlab.engin.umich.edu/freud/).
These notebooks may be launched [interactively on Binder](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)
or downloaded and run on your own system.

```bash
git clone https://github.com/glotzerlab/freud-examples.git
cd freud-examples
jupyter notebook
```

See [Notebook Basics](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb) and [Running Code](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Running%20Code.ipynb) for tutorials on using Jupyter itself.

To test the notebooks and ensure that they all run, use:

```bash
python -m pytest -v --nbval --nbval-lax --ignore=archive/
```
