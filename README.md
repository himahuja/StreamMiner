# README.md

The volume of information today is outpacing the capacity of experts to fact-check it, and in the Information Age the real-world consequences of misinformation are becoming increasingly dire. Recently, computational methods for tackling this problem have been proposed with many of them revolving around knowledge graphs. We present a novel computational fact-checking algorithm, **We should give it a name**, *inspired from/ improving on/ inspired from and imporving on* the techniques used in state-of-the-art fact-checking algorithms, PredPath and Knowledge Stream. ~~which draws on concepts from two of the leading algorithms for computational fact-checking on knowledge graphs, Knowledge Stream and PredPath.~~ Our solution views the problem of fact-checking as a link-prediction problem which relies on discriminitive path model, but draws on the idea of relational similarity and node generality to redefine path length. This gives our solution the advantage of training on more specific paths consisting of edges whose predicates are more conceptually similar to the target predicate. ~~Combining these concepts lead to a more robust and intuitive model for computational fact-checking.~~ The proposed algorithm shows *perfromance at-par with the state-of-the-arts* but lead to a more robust and intuitive model for computational fact-checking.

## Getting Started

Make the environment settings with "requirements.txt": `pip install -r requirements.txt`  

Run : `python setup.py build_ext -if`

The Directory where this will work: `$HOME/Documents/streamminer`

Download data from the following URL http://carl.cs.indiana.edu/data/fact-checking/data.zip and decompress it inside streamminer directory.

## Performance

|                                       | StreamMiner | PredPath | KS     | KL-REL |
|---------------------------------------|-------------|----------|--------|--------|
| cross_US_Presidents_vs_First_Lady.csv | 1.0000      | 1.0000   | 0.9805 | 0.9832 |
| Player_vs_Team_NBA.csv                | 0.96341     | 0.9231   | 0.9996 | 0.9994 |
| predpath_civil_war_battle.csv         | 0.67044     | 0.9951   | 0.7780 | 0.8634 |
| predpath_state_capital.csv            | 1.0000      | 0.9968   | 1.0000 | 1.0000 |
| predpath_vice_president.csv           | 0.85365     | 0.9440   | 0.7780 | 0.8729 |


## Inspiration and Acknowledgements:

[Discriminative Predicate Path Mining for Fact Checking in Knowledge Graphs](https://arxiv.org/abs/1510.05911)

Baoxu Shi, Tim Weninger

> Traditional fact checking by experts and analysts cannot keep pace with the volume of newly created information. It is important and necessary, therefore, to enhance our ability to computationally determine whether some statement of fact is true or false. We view this problem as a link-prediction task in a knowledge graph, and present a discriminative path-based method for fact checking in knowledge graphs that incorporates connectivity, type information, and predicate interactions. Given a statement S of the form (subject, predicate, object), for example, (Chicago, capitalOf, Illinois), our approach mines discriminative paths that alternatively define the generalized statement (U.S. city, predicate, U.S. state) and uses the mined rules to evaluate the veracity of statement S. We evaluate our approach by examining thousands of claims related to history, geography, biology, and politics using a public, million node knowledge graph extracted from Wikipedia and PubMedDB. Not only does our approach significantly outperform related models, we also find that the discriminative predicate path model is easily interpretable and provides sensible reasons for the final determination.

[Finding Streams in Knowledge Graphs to Support Fact Checking](https://arxiv.org/abs/1708.07239)

Prashant Shiralkar, Alessandro Flammini, Filippo Menczer, Giovanni Luca Ciampaglia

> The volume and velocity of information that gets generated online limits current journalistic practices to fact-check claims at the same rate. Computational approaches for fact checking may be the key to help mitigate the risks of massive misinformation spread. Such approaches can be designed to not only be scalable and effective at assessing veracity of dubious claims, but also to boost a human fact checker's productivity by surfacing relevant facts and patterns to aid their analysis. To this end, we present a novel, unsupervised network-flow based approach to determine the truthfulness of a statement of fact expressed in the form of a (subject, predicate, object) triple. We view a knowledge graph of background information about real-world entities as a flow network, and knowledge as a fluid, abstract commodity. We show that computational fact checking of such a triple then amounts to finding a "knowledge stream" that emanates from the subject node and flows toward the object node through paths connecting them. Evaluation on a range of real-world and hand-crafted datasets of facts related to entertainment, business, sports, geography and more reveals that this network-flow model can be very effective in discerning true statements from false ones, outperforming existing algorithms on many test cases. Moreover, the model is expressive in its ability to automatically discover several useful path patterns and surface relevant facts that may help a human fact checker corroborate or refute a claim.

Our work would not be possible without the relentless work of many researchers, and in particular Prashant Shiralkar who has contributed much to the field and whose [Knowledge Stream GitHub repo](https://github.com/shiralkarprashant/knowledgestream) was indispensable to our work.

Special thanks are also in order to the [Indiana University Bloomington Networks & agents Network (NaN) group](http://cnets.indiana.edu/groups/nan/) for providing a [data repository](http://carl.cs.indiana.edu/data/) which we utilized and both [Google Relation Extraction Corpora (GREC)](https://ai.googleblog.com/2013/04/50000-lessons-on-how-to-read-relation.html) and [WSDM Cup 2017 Triple Scoring challenge corpus](https://www.wsdm-cup-2017.org/triple-scoring.html) who provide relational triples we utlized in evaluating our algorithm

## Helpful Resources

#### CSR Matrices

[Compressed Sparse Row Format (CSR)](https://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html)

#### Cython

[Cython Wiki](https://github.com/cython/cython/wiki)
