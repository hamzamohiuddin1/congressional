# congressional

Creates an SVM based machine learning model that can be used to predict whether a specific member of congress (query) will vote to pass or fail a piece of legislation. Has an average f1 accuracy score of 0.88

Usage:
python congressional.py congressdir/data/118/bills {query}

Example:
python congressional.py congressdir/data/118/bills scott peters

                precision    recall  f1-score   support

           0       0.86      0.92      0.89        13
           1       0.90      0.82      0.86        11

    accuracy                           0.88        24
    macro avg      0.88      0.87      0.87        24
    weighted avg   0.88      0.88      0.87        24


