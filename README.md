# Aurora-Engineering-Challenge
Coded Response to Aurora Engineering Challenge: https://hackmd.io/@birchmd/Syj4Hkcmc 

The code features a Multilinear Regression Model as well as a Random Forest Regression Model, written with sci-kit learn. Both models are optimized for different sample sizes and processing times. Multilinear Regression is fast and proves accurate with smaller sample sizes, but will need a higher number of features (min 4) to work well. Random Forest takes some time to compute and is training-intensive, but can process with high accuracy, while using a smaller number of features.

The code has been written in Visual Studio and is optimized to run on an AWS t3a.xlarge or t3a.2xlarge instance (or similar).

