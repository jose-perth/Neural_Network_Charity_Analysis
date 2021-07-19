# Neural_Network_Charity_Analysis

## Overview of the analysis

This project is the challenge for week 19th of the Data Science Bootcamp. It allows us to put into practice and showcase the skills learned in Module 19 of the bootcamp.

### Purpose

During this project, we created a binary classifier capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Results

Program files related to this project can be found in the `Neural_Network_Charity_Analysis` repository.

- [AlphabetSoupCharity.ipynb](AlphabetSoupCharity.ipynb) : Jupyter Notebook containing deliverables 1 and 2.
- [AlphabetSoupCharity_Optimization.ipynb](AlphabetSoupCharity_Optimization.ipynb) : Jupyter Notebook containing deliverable 3.
- [AlphabetSoupCharity.h5](AlphabetSoupCharity.h5) : trained model from deliverable 2.
- [AlphabetSoupCharity_Optimized.h5](AlphabetSoupCharity_Optimized.h5) : trained model from deliverable 3.
- [/Resources/charity_data.csv](/Resources/charity_data.csv) : dataset
- [/checkpoints_deliverable2/](/checkpoints_deliverable2/) : folder containing checkpoints during training of model in deliverable 2.
- [/checkpoints_deliverable3/](/checkpoints_deliverable2/) : folder containing checkpoints during training of model in deliverable 3.

### Data Preprocessing

#### Target Variables

The target variable was identified to be the `IS_SUCCESSFUL` column. This was already a binomial category (0 or 1). So, no preprocessing was needed. Just the assignment to the target variable `y`.

```python
y = application_df.IS_SUCCESSFUL.values
```

#### Features

The following columns in the dataset were imported and preprocessed as outlined in the table below.  After the encoding was completed and the feature set X split between the train and test set, the X_train and X_test set were standardized using the `StandardScaler()`.

|Feature|Type|Preprocessing|
|:---|---:|:---|
|EIN| text| dropped, not relevant to the result|
|NAME| text| dropped, not relevant to the result|
|APPLICATION_TYPE| text| reclassified to reduce unique values and encoded into numeric features|
|AFFILIATION    |text| encoded into numeric features|
|CLASSIFICATION| text| reclassified to reduce unique values and encoded into numeric features|
|USE_CASE |  text| encoded into numeric features|
|ORGANIZATION| text| encoded into numeric features|
|STATUS | numeric| dropped as it is heavily unbalanced|
|INCOME_AMT| text|encoded into numeric features|
|SPECIAL_CONSIDERATIONS |text| dropped as it is heavily unbalanced|
|ASK_AMT| numeric| truncated possible max value to 100,000 to reduce effect of outliers|

#### - Variables removed from dataset

As outlined above, the variables from the dataset that were removed are:

EIN, NAME, STATUS, SPECIAL_CONSIDERATIONS

Most of this preprocessing can be seen in the Deliverable 1 file: [AlphabetSoupCharity.ipynb](AlphabetSoupCharity.ipynb)

### Compiling, Training and Evaluating the Model

#### Initial model

The first model, attempted in Deliverable 2, had the following parameters:

|Layer | Input_dim| units| activation|
|:---:|:---:|:---:|:----:|
|1|41|80|relu|
|2||30|relu
|output||1|sigmoid|

It was run for 30 epochs and produced these results:

`Loss: 0.5539984107017517, Accuracy: 0.7374927401542664`

This model was saved every 5 epochs by using the `callbacks` argument:

```python
# Import checkpoint dependencies
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Define path and callback
os.makedirs("checkpoints/", exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"

cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                             save_weights_only=True,
                             save_freq='epoch', period=5)

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=30, callbacks=[cp_callback])
```

The intermediate saves can be seen in the folder [checkpoints_deliverable3\]checkpoints_deliverable3\

Additionally, the model was saved to an [hdf5 file](AlphabetSoupCharity.h5) file using:

```python
# Save model to hdf5 file
nn.save("AlphabetSoupCharity.h5")
```

#### Target model performance

I was not able to reach 75% for the target performance.

#### Searching for better accuracy

During the work on deliverable 3, I analyse the available features in the dataset and concluded that `SPECIAL_CONSIDERATIONS` and `STATUS` were too unbalanced and removed them. 
I also looked at the distribution of the feature `ASK_AMT`. Since in more than 87% of the cases its value was under 10,000, I recategorized the feature to have a maximum of 100,000.

After a few manual attempts, I used the Keras Tuner to speed up the process to find a better set of parameters for the model.

Here are the results and parameters for 10 trials attempted by the Tuner in one of the runs.  Each trial was run for 60 epochs.

|Trial | Activation| 1st Layer units | Additional Hidden Layers| Units| Accuracy|
|---|---|---|---|---|---|
|1| relu| 80| 4| units_0: 120; units_1: 60; units_2: 140; units_3: 140| 0.7372|
|2| relu| 120|4|units_0: 160; units_1: 80; units_2: 160; units_3: 120| 0.7371|
|3|tanh|40|6|units_0: 40; units_1: 140; units_2: 140; units_3: 80; units_4: 140; units_5: 160|0.7341|
|4| tanh| 60|4| units_0: 100; units_1: 40; units_2: 60; units_3: 120|0.7339|
|5|sigmoid|60|6|units_0: 100; units_1: 100; units_2: 100; units_3: 80; units_4: 40; units_5: 40|0.7315|
|6|tanh| 100|4|units_0: 100; units_1: 80; units_2: 100; units_3: 80|0.7142|
|7|tanh|100|3|units_0: 120; units_1: 140; units_2: 60; units_3: 120|0.6736|
|8|relu|120|4|units_0: 60; units_1: 40; units_2: 40;  units_3: 40|0.4692|
|9|sigmoid|120|4|units_0: 60; units_1: 40; units_2: 100;  units_3: 120| 0.4692|
|10|sigmoid|140|5|units_0: 100; units_1: 120; units_2: 100; units_3: 160; units_4: 140; units_5: 60| 0.4692

The setting for the Keras Tuner can be found in the Jupyter Notebook: [AlphabetSoupCharity_Optimization.ipynb](AlphabetSoupCharity_Optimization.ipynb)

A model fit and tested based on the the best result from the Keras Tuner was saved in [AlphabetSoupCharity_Optimized.h5](AlphabetSoupCharity_Optimized.h5).

## Summary

After cleaning and standardizing of the dataset, the best model created has an accuracy of 73%.

It can still be used as a reliable indicator of the future success of the applicants.

### Recommendations

Further work with the Keras Tuner may allow us to tune other design settings that have not been included in this project, ie. learning rate.

Additionally, we can do more attempts to cleaning the dataset.  For example, there are multiple variables with categories that are almost neglible and could simplify the model.