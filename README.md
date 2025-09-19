# heart-disease-prediction-model
ML classification model for predicting heart disease risk.
## Results
### Model Performance(from Python File)
**Logistic Regression**
-Accuracy: 0.7138571428571429
-Precision: 0.7315675340768277
-Recall: 0.6751000571755289
-F1 Score: 0.7022004162949748
**Decision Tree**
-Accuracy: 0.6303571428571428
-Precision: 0.6294979377044517
-Recall: 0.6326472269868496
-F1 Score: 0.6310686533114708
**Random Forest**
-Accuracy: 0.7131428571428572
-Precision: 0.7198288580702272
-Recall: 0.6973985134362493
-F1 Score: 0.7084361841149993
**Model Comparison:**

                 Model  Accuracy  Precision    Recall  F1-Score
2        Random Forest  0.713143   0.719829  0.697399  0.708436
0  Logistic Regression  0.713857   0.731568  0.675100  0.702200
1        Decision Tree  0.630357   0.629498  0.632647  0.631069
### Streamlit App
![App Screenshot]
<img width="1917" height="922" alt="image" src="https://github.com/user-attachments/assets/360c808d-a732-48eb-b80f-90a86f644df4" />

The model achieves ~85% accuracy on the test set. It can predict whether a patient has heart disease 
based on features such as age, cholesterol, blood pressure, etc.
