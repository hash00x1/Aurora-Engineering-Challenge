# Aurora-Engineering-Challenge
Coded Response to Aurora Engineering Challenge: https://hackmd.io/@birchmd/Syj4Hkcmc 

The code features a Multilinear Regression Model as well as a Random Forest Regression Model, written with sci-kit learn. Both models are optimized for different sample sizes and processing times. Multilinear Regression is fast and proves accurate with smaller sample sizes, but will need a higher number of training features (>=4) to work well. Random Forest takes some time to compute and is training-intensive, but can process with high accuracy while using a smaller number of training features.

The code has been written in Visual Studio and is optimized to run on an AWS t3a.xlarge or t3a.2xlarge instance (or similar).

To test the model, adjust the "path"-parameter to your local copy of AURORA Challenge .json- training files and simply run the scripts. The code features no option to display graph-outputs in a non-IDE environment. This will need to be added in a future version.

To observe the changing accuracy output between Linear Regression and Random Forest modelling, simply adjust the sample_size parameter for readFiles().
