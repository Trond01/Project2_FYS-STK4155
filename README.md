# Project2 FYS-STK4155

## Structure

This is a repository for the second project in the course FYS-STK4155 - Applied Data Analysis and Machine Learning with the following structure:

├── Code # The code used in the project, including experiment functions with user friendly interface \
├── Report # A report of our results, methods, etc...\
├── Runs # The code used to perform experiments generating the figures in the report\
├──── Figures # figures generated for the report\
├──── testing # additional notebooks use to verify code and benchmark \
├── README.md \
└── .gitignore

The code is demonstrated in jupyter notebooks found in the folder Runs. 

## Some notes on the code

A central goal of this project has been the implementation of descent methods that can solve a wide range of optimisation problems. The implementation ensures that given a dictionary of parametres and a function to evaluate the gradient of some cost function wrt. each key, we can perform gradient descent. 

