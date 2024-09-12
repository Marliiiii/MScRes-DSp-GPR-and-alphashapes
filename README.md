# MScRes-DSp-GPR-and-alphashapes
This repository contains 5 notebooks used in the study which correspond to the following list of mathematical problems commonly used for optimisation analysis:

<img width="496" alt="Screenshot 2024-09-12 at 16 02 44" src="https://github.com/user-attachments/assets/af263011-c591-4bb8-9e63-2a49f624fc6d">



All the notebooks follow the methodology’s framework below. Within the notebooks the steps are numbered and explained with reference to this framework.


<img width="382" alt="Screenshot 2024-09-12 at 16 11 05" src="https://github.com/user-attachments/assets/06f762d0-333f-4510-9e70-3a29a69062eb">



As well as this the kernel_opt_file contains the following list of functions developed for the study:
- Custom MAPE function which accounts for small true values.
- Percentage of small values function  which shows what percentage of true values is close to zero (useful to understanding high MAPE values).
- Sobol sampling function.
- Kernel to string function which enables kernels to be outputted as its names.
- Find best kernel function is *the kernel optimisation function* developed for this study utilising greedy tree search to find the best composite kernel based on MAE.

