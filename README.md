# MVA2022-Geometric-Data-Analysis
Codebase studying [Dong & al's](https://arxiv.org/abs/1106.2233v1), for a school project in the Geometric Data Analysis Course (Jean Feydy) at Master 2 MVA in Ecole Normale Superieure Paris-Saclay. 

We studied the clustering of multilayer graphs on two different datasets. Specifically, we implemented two algorithms derived from the single-layer Normalized Spectral Clustering algorithm: one based on eigen-decomposition to build a joint spectrum between the different layers of the graph and the other based on spectral regularization. We used the Kernel K-means algorithm as a baseline and compared our results with the single-layer clustering algorithm to show that multi-layer graph clustering performs better. You can read the report for more details. 

## Authors

- Sonia MAZELET ([github](https://github.com/SoniaMaz8))

- Capucine GARÇON ([github](https://github.com/CapucineGARCON))

## Datasets

We have used the [CORA Dataset](https://graphsandnetworks.com/the-cora-dataset) and the [MIT Reality Mining Dataset](http://realitycommons.media.mit.edu/realitymining.html) to construct our dataset.


## Bibliography

### Reference article

Dong & al., *Clustering with Multi-Layer Graphs : a Spectral Perspective* ([arxiv](https://arxiv.org/abs/1106.2233v1))

```
@article{Xiaowen_Dong_2012,
	doi = {10.1109/tsp.2012.2212886},
	url = {https://doi.org/10.1109%2Ftsp.2012.2212886},
	year = 2012,
	month = {nov},
	publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
	volume = {60},
	number = {11},
	pages = {5820--5831},
	author = {Xiaowen Dong and Pascal Frossard and P. Vandergheynst and N. Nefedov},
	title = {Clustering With Multi-Layer Graphs: A Spectral Perspective},
	journal = {{IEEE} Transactions on Signal Processing}
}
```
