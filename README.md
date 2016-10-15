# oversampling
Implementation of the SMOTE and ADASYN algorithms using the scikit-learn library
Pre-alpha build. Please note that this is only suitable for binary classifications at this moment in time.

##Roadmap

- To allow the SMOTE and ADASYN algorithm to be used as part of a SK-pipeline
- To enable SMOTE and ADASYN to be used in non-binary classification

The imbalanced learn project: https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn
is significantly more feature complete (and allows the use of pipelines) and users are encouraged to use that project. 
This project will attempt to implmement over-sampling algorithms in multi-classification contexts.
