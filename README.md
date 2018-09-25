# README.md

The volume of information today is outpacing the capacity of experts to fact-check it, and in the Information Age the real-world consequences of misinformation are becoming increasingly dire. Recently, computational methods for tackling this problem have been proposed with many of them revolving around knowledge graphs. We present a novel computational fact-checking algorithm, RelPredPath, inspired by and improving on the techniques used in state-of-the-art fact-checking algorithms, PredPath and Knowledge Stream. Our solution views the problem of fact-checking as a link-prediction problem which relies on discriminitive path model, but draws on relational similarity and node generality to redefine path length. This gives our solution the advantage of training on more specific paths consisting of edges whose predicates are more conceptually similar to the target predicate. RelPredPath shows performance at-par with other state-of-the-art fact-checking algorithms, but leads to a more robust and intuitive model for computational fact-checking.

## Getting Started

Make the environment settings with "requirements.txt": `pip install -r requirements.txt`  

Run : `python setup.py build_ext -if`

The Directory where this will work: `$HOME/Documents/streamminer`

Download data from the following URL http://carl.cs.indiana.edu/data/fact-checking/data.zip and decompress it inside streamminer directory.

## Performance

|                                       | RelPredPath | PredPath | KS     | KL-REL |
|---------------------------------------|-------------|----------|--------|--------|
| cross_US_Presidents_vs_First_Lady.csv | 1.0000      | 1.0000   | 0.9805 | 0.9832 |
| Player_vs_Team_NBA.csv                | 0.96341     | 0.9231   | 0.9996 | 0.9994 |
| predpath_civil_war_battle.csv         | 0.67044     | 0.9951   | 0.7780 | 0.8634 |
| predpath_company_president.csv        | 0.79345     | 0.8867   | 0.8119 | 0.8988 |
| predpath_state_capital.csv            | 1.0000      | 0.9968   | 1.0000 | 1.0000 |
| predpath_vice_president.csv           | 0.85365     | 0.9440   | 0.7780 | 0.8729 |


## Inspiration

[Discriminative Predicate Path Mining for Fact Checking in Knowledge Graphs](https://arxiv.org/abs/1510.05911)

Baoxu Shi, Tim Weninger

> Traditional fact checking by experts and analysts cannot keep pace with the volume of newly created information. It is important and necessary, therefore, to enhance our ability to computationally determine whether some statement of fact is true or false. We view this problem as a link-prediction task in a knowledge graph, and present a discriminative path-based method for fact checking in knowledge graphs that incorporates connectivity, type information, and predicate interactions. Given a statement S of the form (subject, predicate, object), for example, (Chicago, capitalOf, Illinois), our approach mines discriminative paths that alternatively define the generalized statement (U.S. city, predicate, U.S. state) and uses the mined rules to evaluate the veracity of statement S. We evaluate our approach by examining thousands of claims related to history, geography, biology, and politics using a public, million node knowledge graph extracted from Wikipedia and PubMedDB. Not only does our approach significantly outperform related models, we also find that the discriminative predicate path model is easily interpretable and provides sensible reasons for the final determination.

[Finding Streams in Knowledge Graphs to Support Fact Checking](https://arxiv.org/abs/1708.07239)

Prashant Shiralkar, Alessandro Flammini, Filippo Menczer, Giovanni Luca Ciampaglia

> The volume and velocity of information that gets generated online limits current journalistic practices to fact-check claims at the same rate. Computational approaches for fact checking may be the key to help mitigate the risks of massive misinformation spread. Such approaches can be designed to not only be scalable and effective at assessing veracity of dubious claims, but also to boost a human fact checker's productivity by surfacing relevant facts and patterns to aid their analysis. To this end, we present a novel, unsupervised network-flow based approach to determine the truthfulness of a statement of fact expressed in the form of a (subject, predicate, object) triple. We view a knowledge graph of background information about real-world entities as a flow network, and knowledge as a fluid, abstract commodity. We show that computational fact checking of such a triple then amounts to finding a "knowledge stream" that emanates from the subject node and flows toward the object node through paths connecting them. Evaluation on a range of real-world and hand-crafted datasets of facts related to entertainment, business, sports, geography and more reveals that this network-flow model can be very effective in discerning true statements from false ones, outperforming existing algorithms on many test cases. Moreover, the model is expressive in its ability to automatically discover several useful path patterns and surface relevant facts that may help a human fact checker corroborate or refute a claim.

## Ideas for Increasing Performance

#### [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)

<center><img src="img/AstarExampleEn.gif"></center>

> In computer science, A* (pronounced as "A star") is a computer algorithm that is widely used in pathfinding and graph traversal, which is the process of finding a path between multiple points, called "nodes". It enjoys widespread use due to its performance and accuracy. However, in practical travel-routing systems, it is generally outperformed by algorithms which can pre-process the graph to attain better performance, although other work has found A* to be superior to other approaches. Peter Hart, Nils Nilsson and Bertram Raphael of Stanford Research Institute (now SRI International) first published the algorithm in 1968. It can be seen an extension of Edsger Dijkstra's 1959 algorithm. A* achieves better performance by using heuristics to guide its search.

#### [Other K-Shortest Path Algorithms](https://en.wikipedia.org/wiki/K_shortest_path_routing)

[K⁎: A heuristic search algorithm for finding the k shortest paths](https://www.sciencedirect.com/science/article/pii/S0004370211000865?via%3Dihub)

## Helpful Resources

#### CSR Matrices

Compressed Sparse Row matrix

From [Compressed Sparse Row Format (CSR)](https://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html):

Advantages of the CSR format
* efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
* efficient row slicing
* fast matrix vector products

Disadvantages of the CSR format
* slow column slicing operations (consider CSC)
* changes to the sparsity structure are expensive (consider LIL or DOK)

#### [Cython](http://cython.org/)

> Cython is an optimising static compiler for both the Python programming language and the extended Cython programming language (based on Pyrex). It makes writing C extensions for Python as easy as Python itself.

[Cython Wiki](https://github.com/cython/cython/wiki)

#### Profiling

[Line_Profiler](https://github.com/rkern/line_profiler)

> line_profiler is a module for doing line-by-line profiling of functions.

The easiest way to get started is to use the kernprof script.

```bash
$ kernprof -l -v streamminer2.py -o output -d datasets/sub_sample.csv -m sm
```

To view the results, run
```bash
$ python -m line_profiler streamminer2.py.lprof
```

[Memory Profiler](https://pypi.org/project/memory_profiler/)

> This is a python module for monitoring memory consumption of a process as well as line-by-line analysis of memory consumption for python programs. It is a pure python module which depends on the psutil module.

```bash
$ python -m memory_profiler streamminer2.py -o output -d datasets/sub_sample.csv -m sm
```

## Acknowledgements

Our work would not be possible without the relentless work of many researchers, and in particular Prashant Shiralkar who has contributed much to the field and whose [Knowledge Stream GitHub repo](https://github.com/shiralkarprashant/knowledgestream) was indispensable to our work and served as the code base for this project.

We are also extremely thankful to [UCLA's Institute for Pure and Applied Mathematics](http://www.ipam.ucla.edu/) [Research in Industrial Projects for Students (RIPS)](https://www.ipam.ucla.edu/programs/student-research-programs/research-in-industrial-projects-for-students-rips-2018/) and the NSF for funding our research and especially Susana Serna for going above and beyond to advise and support us both academically, personally, and professionally.

Special thanks are also in order to the [Indiana University Bloomington Networks & agents Network (NaN) group](http://cnets.indiana.edu/groups/nan/) for providing a [data repository](http://carl.cs.indiana.edu/data/) which we utilized and both [Google Relation Extraction Corpora (GREC)](https://ai.googleblog.com/2013/04/50000-lessons-on-how-to-read-relation.html) and [WSDM Cup 2017 Triple Scoring challenge corpus](https://www.wsdm-cup-2017.org/triple-scoring.html) who provide relational triples we utlized in evaluating our algorithm

```
