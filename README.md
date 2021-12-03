# Introduction

This project is a python-based module, aims to extract the company list (with titles) of IPO sponsors and underwriters from indicated pdf pages from pdf documents of IPO prospectus (English version only).

The main script is the api_server.py and it was run at the background by .service script in UAT and production environment. 

The key inputs of requests include stockcodes, asa filepath and type of information to be retrived (either sponsor or underwriter)

The module is mainly built by python3.7, and the api_server.py mainly uses the open-library pymupdf (https://github.com/pymupdf/PyMuPDF) to read and extract the structure and content of pdf files and relies on two NLP models (one for line gluer  and one for Name entity recongition NER) to process the textual data extracted from PDF and finally apply the another open library, intervaltree (https://github.com/chaimleib/intervaltree) to regroup the company lists of sponsors and underwriters based on their span and location in the paragraph.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# **Module Structures**
![](pic/structure.JPG)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **Module dependencies** 
### Config:
#### yaml2pyclass : 0.1.0 (source : https://github.com/a-nau/yaml2pyclass  )

### General Utility
#### Numpy: 1.20.3 (source : https://www.numpy.org/ )
#### filetype.py : v1.0.0 (source:  https://github.com/h2non/filetype.py )

### Pdf processing:
#### Pymupdf : 1.18.13 (source : https://github.com/pymupdf/PyMuPDF )

### Machine learning
#### Pytorch : 1.8.1+cpu (source : https://pytorch.org/  )
#### Scikit-learn : 0.24.2 (source : https://scikit-learn.org/ )
#### Scipy: 1.6.3 (source : https://www.scipy.org/ )
#### intervaltree : 3.1.0 (source : https://github.com/chaimleib/intervaltree )

### NLP:
#### Clean-text : 0.4.0 (source : https://github.com/jfilter/clean-text )
#### Rapidfuzz : 1.4.1 (source : https://github.com/maxbachmann/RapidFuzz )
#### Pyflashtext : 2.7.1  (source :  https://github.com/francbartoli/pyflashtext )

### API+Server
#### Hug : 2.6.1 (source : https://hugapi.github.io/hug/ )

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
