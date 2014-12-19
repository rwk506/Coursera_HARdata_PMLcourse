# Machine Learning Prediction of the Human Activity Recognition Data
==================================================================
</br>
## Executive Summary:
The goal of this study is to build a model that can predict the type of activity or exercise performed based on measurements of human movement. We use practical machine learning techniques to build a model to predict the manner of the exercise, "classe", based on a variety of collected information. The training dataset is split and we use a generalized boosting algorithm with a standard cross validation technique to train on 60% of the training data and build a model. This model is tested to determine the in and out of sample errors (99.86% and 99.5%) then used to predict the "classe" of 20 different test cases.

</br>
</br>
## Dataset:
The data used in this analysis is the Human Activity Recognition Dataset for the Practical Machine Learning course as provided by Groupware (http://groupware.les.inf.puc-rio.br/har) and Ugulino, W., Cardador, D., Vega, K. et al. 2012. The dataset consists of 19,622 observations of 160 variables that describe subjects and their physical movement during activities.


</br>
</br>
### Loading data

First, we load the necessary packages for the impending analysis.

```r
library(knitr); library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

Next, we load the two necessary files from the current directory. We check the first ten columns of the header to make sure the file has been correctly loaded.


```r
trainingset=read.csv('pml-training.csv')
finaltestset=read.csv('pml-testing.csv')
head(trainingset[0:10])   ## a brief check to make sure file loaded correctly
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt
## 1         no         11      1.41       8.07    -94.4
## 2         no         11      1.41       8.07    -94.4
## 3         no         11      1.42       8.07    -94.4
## 4         no         12      1.48       8.05    -94.4
## 5         no         12      1.48       8.07    -94.4
## 6         no         12      1.45       8.06    -94.4
```


</br>
</br>
### Cleaning and Preprocessing Data

Many of the entries for each observation are NAs. We exclude columns that have many NAs from the table, as these columns will not add any useful information to the model that we build. This reduces the number of columns from 160 to 60.


```r
# We select only columns without NAs
train_proc=trainingset[colSums(is.na(finaltestset))==0]
finaltest_proc=finaltestset[colSums(is.na(finaltestset))==0]

# We take the "X" column (the first column) out of the dataset, as it is just an index, not a valid predictor
train_proc=subset(train_proc,select=-c(X))
finaltest_proc=subset(finaltest_proc,select=-c(X))

plot(train_proc$classe,col=terrain.colors(5),ylab="Frequency",main="Histogram of Classe in Training Set")
```

![plot of chunk Data_Cleaning](figure/Data_Cleaning.png) 

We further process the dataset by breaking the table into 60% training data and 40% testing data for our machine learning algorithm.


```r
set.seed(23095)
inTrain = createDataPartition(train_proc$classe, p=0.6, list=FALSE)
sampleTrain=train_proc[inTrain,]
sampleTest=train_proc[-inTrain,]
```


</br>
</br>
### Building a model

Now, we build a model using ML techniques. We have chosen a Generalized Boosting Regression Model method for classification and a standard cross-validation method. The Generalized Boosting Regression Model ("GBM") was chosen for its high accuracy. This model takes many very weak predictors (such as we have here), weights them, and combines them in order to obtain a much stronger predictor. What is lost in interpretability in the boosting model algorithm is gained in model accuracy. We cross-validate our method with generic cross-validation to attain decent accuracy without compromising on speed.



```r
modelFit=train(classe~.,method="gbm",trControl=trainControl(method="cv", number=3, allowParallel=TRUE), data=sampleTrain)
```


```r
modelFit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 80 predictors of which 44 had non-zero influence.
```

The model fit used a total of 11776 samples (60% of the training data) to model the 5 classes: 'A', 'B', 'C', 'D', 'E' with a cross-validated boosting model. The accuracies are very high and the standard deviations very low, indicating a model that fits the training set well.


</br>
</br>
### Errors

Next we need to examine how well the model is performing. First, we determine the in-sample error, or how well the model predicts the outcome (classe) for the same data used to build the model.


```r
### In-sample error
modelPred=predict(modelFit, sampleTrain)
confusionMatrix(sampleTrain$classe, modelPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    2 2275    2    0    0
##          C    0    2 2043    9    0
##          D    0    0    4 1925    1
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.999    0.997    0.995    1.000
## Specificity             1.000    1.000    0.999    0.999    1.000
## Pos Pred Value          1.000    0.998    0.995    0.997    1.000
## Neg Pred Value          1.000    1.000    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.999    0.998    0.997    1.000
```

The model performs predictions for the training set with 99.86% accuracy. Because we used the training set as a basis for the model, this is unsurprising. We now use the testing dataset set aside from the initial training set to predict the out of sample error.


```r
### Predicted Out-of-sample error
modelPred2=predict(modelFit, sampleTest)
confusionMatrix(sampleTest$classe, modelPred2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    9 1506    3    0    0
##          C    0    1 1353   14    0
##          D    0    0    5 1278    3
##          E    0    0    0    3 1439
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.999    0.994    0.987    0.998
## Specificity             1.000    0.998    0.998    0.999    1.000
## Pos Pred Value          1.000    0.992    0.989    0.994    0.998
## Neg Pred Value          0.998    1.000    0.999    0.997    1.000
## Prevalence              0.286    0.192    0.173    0.165    0.184
## Detection Rate          0.284    0.192    0.172    0.163    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.999    0.996    0.993    0.999
```

Our model also performs well on the testing data (40% of the training set data) set aside during training, with approximately 99.5% accuracy. This suggests that the model is not over-fitting too terribly to noise in the training dataset. 
 
Confident that we can accurately predict classifications not only in sample but also out of sample, we extend our model to the test cases initially provided (re: from the pml-testing.csv file). We expect similar accuracy as the out of sample error.


</br>
</br>
### Predicting Test Cases

Now, we use the model we have developed and tested to explicitly predict answers for the homework and write the files for submission. 


```r
modelPred3=predict(modelFit, finaltest_proc)
answers=as.character(modelPred3)
print(answers)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("submitted_answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```


</br>
</br>
### Summary

Using the highly accurate boosting algorithm with repeated cross-validation on 60% of the training dataset, we obtain 99.86% in-sample error and and estimated 99.5% out of sample error (on the remaining 40% of the training dataset used for testing). We are able to use our model to accurately predict the "classe" of 20 test cases based on new data (achieved 20 out of 20 answers correct upon submission).

</br>
</br>


