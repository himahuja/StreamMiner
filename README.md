### To get this started:

Make the environment settings with "requirements.txt": `pip install -r requirements.txt`  

Run : `python setup.py build_ext -if`

The Directory where this will work: `$HOME/Documents/streamminer`

Download data from the following URL http://carl.cs.indiana.edu/data/fact-checking/data.zip and decompress it inside streamminer directory.

## Issues:

#### Memory Usage

Streamminer's memeory usage increases linearly as a function of time while Knowledge Stream's does not. Something tells me that has to be an error somewhere. We should be deleting some variables from memory or something that we aren't which is causing the memory to keep growing, making the main hard to run on most computers.

#### Paths Are Have Cycles

I have noted many paths with cycles in them. 

#### "599 Error"

Something is going on with the mask that is hiding the pid allowing the target predicate to be found by the path extraction after a couple iterations?

