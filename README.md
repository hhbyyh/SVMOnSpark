# SVMOnSpark

Yuhao Yang
May 2016

SVMOnSpark is an implementation of distributed SMO algorithm to train a binary Support Vector Machine.
The scalability is supposed to be very good as it avoids shuffle and unnecessary communication.

It also supports arbitrary kernels. Currently linear and RBF are embedded.


## Usage:
Typical Spark ml pattern. `SVM` is an `Estimator` and `SVMModel` is the corresponding model. You can also refer to the
Example folder.


## Accuracy and Scalability
The implementation has been tested against MNist dataset and get an accuracy about 99% on the test data set (one vs rest).
We'll post more results once they are ready.



