
# Projet Cookies - Mille Mercis

![Logo](img/logo.jpg)

This GIT provides the code written during the project "Cookies" done with 1000mercis and CentraleSupelec during the year 2016-2017 by Vernhet Paul & Zhang Jiayi.

The git is divided as follows :
* ECP_Code_1 FOLDER : 1st and 2nd datasets with ROC curves, visualizations
* ECP_Code_2 FOLDER : 3rd dataset with ROC curves, visualizations
* look-alike-cookies.py : last dataset generation from 1000mercis database
* img and README.cm : display front page

## Summary of datasets characteristics

* 2 classes : csp+ = positive labels & csp- = negative labels
* number of cookies & domains
* sparsity (= # of different domains visited / # of domains)
* balance

### First dataset

* 26 000 cookies, 13 000 domains
* balanced dataset (50% csp+ 50% csp-)
* sparsity coefficient : 0.24%

### Second dataset

* 49 000 cookies, 13 000 domains
* balanced dataset (50% csp+ 50% csp-)
* sparsity coefficient : 0.23%

| labels   | mean traffic | mean number of different domains visited | sparsity | 
|--------------|--------------------|------------|----------|
| positive     | 311          | 30.7     | 0.23 %     | 
| negative     | 297          | 30.8     | 0.23 %     | 
|    all       | 304          | 30.7     | 0.23 %     | 

### Third dataset
* 100 000 cookies, 10 000 domains
* unbalanced dataset (25% csp+ 75% csp-)
* sparsity coefficient : 0.26%

| labels   | mean traffic | mean number of different domains visited | sparsity | 
|--------------|--------------------|------------|----------|
| positive     | 260          | 26.6     | 0.26 %     | 
| negative     | 255          | 26.3     | 0.26 %     | 
|    all       | 256          | 26.4     | 0.26 %     | 
