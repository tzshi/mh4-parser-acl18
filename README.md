# Code Companion For "Global Transition-based Non-projective Dependency Parsing"

## What are included

- Global and transition-based arc-hybrid parser
- Global and transition-based MH4 parser
- Neural third-order 1EC parser

## Requirements

- python 3.x
- Cython
- numpy

## Usage

Firstly, run `python setup.py build_ext --inplace` to build the non-projective MST decoder. 

Training scripts are in the `scripts` folder, and `acl18-dev.sh` is used for prediction. 

More details coming soon ...

## Acknowledgement

Our code includes the non-projective decoder from [TurboParser](http://www.cs.cmu.edu/~ark/TurboParser/).

## Reference

If you make use of our code or data for research purposes, we'll appreciate your citing the following:

```
@InProceedings{Gomez+Shi+Lee-2018,
  author = 	"G{\'o}mez-Rodr{\'i}guez, Carlos
		and Shi, Tianze
		and Lee, Lillian",
  title = 	"Global Transition-based Non-projective Dependency Parsing",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2663--2674",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1248"
}
```
