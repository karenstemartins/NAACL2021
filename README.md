# NAACL 2021

This repository contains all the data and code to reproduce the experiments from the "Why Do Document-Level Polarity Classifiers Fail?" paper published in NAACL 2021.

Files available in this package:

* `Data/Movie reviews from Metacritic/`: all movie reviews collected from Metacritic. Each json file contains experts and users reviews from a movie.

* `Data/Human Classifier/`: movie reviews classified by the Human Classifier. Each file contains the polarity (Human_classifier_polarity) and label (label) defined by the human classifier, the original polarity from Metacritic (Metacritic_polarity) and the review (review). For each type of user, expert and regular, there are two files: one used in the first and second experiment, and the other selected from Bert misclassified and correctly classified reviews used in the third experiment.

* `Data/Train/`: the data used to train the classifiers.

* `Code/`: the code of the machine classifiers.


