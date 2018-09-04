### To get this started:

Make the environment settings with "requirements.txt": `pip install -r requirements.txt`  

Run : `python setup.py build_ext -if`

The Directory where this will work: `$HOME/Documents/streamminer`

Download data from the following URL http://carl.cs.indiana.edu/data/fact-checking/data.zip and decompress it inside streamminer directory.

## CSR Matrix

[Compressed Sparse Row Format (CSR)](https://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html)

## Issues:

#### Memory Usage

Streamminer's memory usage increases linearly as a function of time while Knowledge Stream's does not. Something tells me that has to be an error somewhere. We should be deleting some variables from memory or something that we aren't which is causing the memory to keep growing, making the main hard to run on most computers.

#### Paths Have Cycles

I have noted many paths with cycles in them.

#### "599 Error"

Something is going on with the mask that is hiding the pid allowing the target predicate to be found by the path extraction after a couple iterations?

## k-Shortest Simple Paths

Okay so I found a YouTube video on [Finding k Simple Shortest Paths and Cycles](https://www.youtube.com/watch?v=RXRyqyxO_jc) and she discusses her work (haven't made it all the way through), saying [Yen](http://www.ams.org/journals/qam/1970-27-04/S0033-569X-1970-0253822-7/S0033-569X-1970-0253822-7.pdf) is the best so far and references [Subcubic Equivalences Between Path, Matrix, and Triangle Problems](https://people.csail.mit.edu/rrw/tria-mmult.pdf) to say that it has been proven to be really fucking hard, but her paper on "algorithms and hardness results" (skimmed, but didn't read all of it yet) is here [Finding k Simple Shortest Paths and Cycles](https://arxiv.org/abs/1512.02157v2). I will look into this and get back to you.
