# Project2 FYS-STK4155

This is a repository for the second project in the course FYS-STK4155 - Applied Data Analysis and Machine Learning.

## Structure

```bash
├── Code          # Project code, including experiment functions with a user-friendly interface
├── Report        # Report detailing results, methods, and analysis
├── Runs          # Code used to perform experiments and generate report figures
│   ├── Figures   # Generated figures for the report
│   ├── Testing   # Additional notebooks for code verification and benchmarking
├── README.md
└── .gitignore    
```

The code is demonstrated in jupyter notebooks found in the folder Runs. 

## Some notes on the code

A central goal of this project has been the implementation of descent methods that can solve a wide range of optimisation problems. The implementation ensures that given a dictionary of parametres and a function to evaluate the gradient of some cost function wrt. each key, we can perform gradient descent. 
