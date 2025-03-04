# Salary Prediction Model

This project is a simple machine learning model that predicts salaries based on experience using linear regression.

## Features
- Uses **pandas** and **numpy** for data manipulation.
- **Matplotlib** for visualization.
- **Scikit-learn** for building a linear regression model.
- Handles missing values and converts textual experience values to numerical format.

## Dataset
The dataset (`salary_sheet.xlsx`) contains information on employee experience and salaries. Missing values in the experience column are filled with "zero" and converted to numerical values using `word2number`.

## Installation
To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn word2number
```

## Usage
1. Load the dataset using pandas.
2. Preprocess the data by handling missing values and converting text to numbers.
3. Train a linear regression model using **scikit-learn**.
4. Make predictions on salaries based on experience.

## Code Overview
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

df = pd.read_excel("salary_sheet.xlsx")
df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)

model = linear_model.LinearRegression()
model.fit(df[['experience']], df['salary'])
print(model.predict([[5]]))  # Predict salary for 5 years of experience
```

## Results
The model predicts salary based on given experience values. You can modify the dataset and retrain the model to improve accuracy.

## Contributing
Feel free to fork this repository and improve the project by adding:
- More features for better salary prediction.
- Data cleaning and outlier detection.
- Hyperparameter tuning for better model performance.

## License
This project is open-source and available under the MIT License.

