
# Predicting Outbound Shipping Cost

* Predictive model in python to calculate actual cost of shipping based on fetaures from 12 months of shipping data.  

# Libraries Used.

## 1.Pandas
Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
## 2.Numpy
NumPy is the fundamental package for scientific computing with Python. It contains among other things:

-A powerful N-dimensional array object
-Sophisticated (broadcasting) functions
-Tools for integrating C/C++ and Fortran code
-Useful linear algebra, Fourier transform, and random number capabilities

## 3.Matplotlib
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

## 4.Seaborn
Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

## 5.Sklearn
Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language.It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

# Data Importing & Pre-Processing

- Collected one year product sales data(dimensions, weight, zone…)
- Assigned specific data types for columns
- Removed observations having incorrect data points based on exploratory analysis 
- Used standard deviation approach for handling outliers
- Not applied imputations since less % of missing values 
- Pearson correlation was used for feature selection
- Found features having multi-collinearity


# Case Study

## Company X is using two calculators for estimating package shipping amount
- Expedited Shipping calculator
- Average of Actuals calculator

## Built Prediction model to replace Average of Actuals calculator
- Predicting the package shipping amount
- Predicting a product likely to be shipped to a particular zone by a specific partner

# Extensions & Improvements
- Model accuracy can be increased using advanced techniques like neural nets
- Use of distributed processing help in saving the model training time
- Can use data science model for expedited shipping calculator also

## Authors

* **Nithin Kamavaram** 
* **Vidnyan Siddamshetty**
* **Anusha Teerdala**

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
