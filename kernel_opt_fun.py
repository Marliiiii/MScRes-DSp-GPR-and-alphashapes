

# Custom MAPE calculation to handle small true values
def custom_mape(y_true, y_pred, epsilon=1e-3):
    import numpy as np
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
# example usage:
# custom_MAPE = custom_mape(Y_true, Y_pred)
# print(f"Custom MAPE = {custom_MAPE:.2f}%")



def percentage_small_values(y_true, threshold=1e-3):
    import numpy as np
    small_value_count = np.sum(np.abs(y_true) < threshold)
    return (small_value_count / len(y_true)) * 100
# Example usegae:
# threshold = 1e-3
# percentsmallY = percentage_small_values(Y_true, threshold)
# print(f"Percentage of small true values (|Y_true| < {threshold}): {percentsmallY:.2f}%")

def Sobol_sequence(lbd, ubd, power_no=12):
    """
    Create 2^power_no of inputs for sampling based on the lists
    of lbd (lower bound) and ubd (upper bound).
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d = len(lbd), scramble = False)
    inputs = sampler.random_base2(m = power_no)
    inputs = qmc.scale(inputs, lbd, ubd)
    return inputs


def kernel_to_str(kernel):
    import GPy
    if isinstance(kernel, GPy.kern.Add):
        # Handle addition of multiple parts
        return ' + '.join([kernel_to_str(part) for part in kernel.parts])
    elif isinstance(kernel, GPy.kern.Prod):
        # Handle multiplication of multiple parts
        return ' * '.join([kernel_to_str(part) for part in kernel.parts])
    else:
        return kernel.name


def find_best_kernel(X, Y, input_dim=1, max_iterations=5):
    """
    Function to iteratively find the best kernel combination based on MAE.
    
    Parameters:
    - X: Input set of samples
    - Y: Target values
    - input_dim: Dimension of the input space
    - max_iterations: Maximum number of iterations to perform
    
    Returns:
    - final_best_kernel: The kernel combination with the lowest MAE across all iterations
    - final_best_MAE: The MAE of the final best kernel

    # Example usage
    # best_kernel, best_MAE = find_best_kernel(X, Y)
    # print("Best kernel:", kernel_to_str(best_kernel))
    # print("Best MAE:", best_MAE)
    """
    import numpy as np
    import GPy
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Initialize base kernels
    kA = GPy.kern.RBF(input_dim=input_dim)
    kB = GPy.kern.Linear(input_dim=input_dim)
    kC = GPy.kern.RatQuad(input_dim=input_dim)
    kD = GPy.kern.StdPeriodic(input_dim=input_dim, period=1.0)

    kD.lengthscale.constrain_bounded(0.1, 10.0)
    kD.variance.constrain_bounded(0.1, 10.0)
    kD.period.constrain_bounded(0.1, 10.0)


    base_kernels = [kA, kB, kC, kD]
    
    best_kernels_per_iteration = []
    
    for iteration in range(max_iterations):
        best_MAE = float('inf')
        best_kernel = None
        new_kernels = []
        
        if iteration == 0:
            bk = base_kernels
        else:
            bk = [best_kernels_per_iteration[-1][0]]  # Start from the best kernel from the last iteration
        
        for i in range(len(bk)):
            for j in range(len(base_kernels)):
                # Addition kernel combination
                kernel_add = bk[i] + base_kernels[j]
                new_kernels.append(kernel_add)
                
                # Multiplication kernel combination
                kernel_prod = bk[i] * base_kernels[j]
                new_kernels.append(kernel_prod)
        
        for kernel in new_kernels:
            model = GPy.models.GPRegression(X_train, Y_train, kernel)
                
            model.optimize()
            
            Y_pred, _ = model.predict(X_test)
            MAE = mean_absolute_error(Y_test, Y_pred)
 
            print(f"Kernel: {kernel_to_str(kernel)}, MAE: {MAE}")
            
            if MAE < best_MAE:
                best_MAE = MAE
                best_kernel = kernel
                
        best_kernels_per_iteration.append((best_kernel, best_MAE))
        #print(f"\nIteration: {iteration + 1}, Best Kernel: {kernel_to_str(best_kernel)}, Best MAE: {best_MAE}")
     
    # Print out the results of each iteration
    print("\nResults of each iteration:")
    for i, (kernel, MAE) in enumerate(best_kernels_per_iteration):
        print(f"Iteration {i + 1}: Best Kernel: {kernel_to_str(kernel)}, MAE: {MAE:.5f}")
    
    # Find the best kernel across all iterations
    final_best_kernel, final_best_MAE = min(best_kernels_per_iteration, key=lambda x: x[1])
    return final_best_kernel, final_best_MAE


    

def evaluate_metrics(Y_true, Y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
    """
    Evaluate classification metrics given the true and predicted labels.
    
    Parameters:
    - Y_true: Array-like of true labels
    - Y_pred: Array-like of predicted labels
    
    Returns:
    - metrics: Dictionary with 'Confusion Matrix', 'Accuracy', 'Recall', 'Precision', 'F1 Score'
    """
    # Compute confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)
    print(cm)
    # Compute accuracy
    accuracy = accuracy_score(Y_true, Y_pred)
    
    # Compute recall (for each class)
    recall = recall_score(Y_true, Y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    
    # Compute precision (for each class)
    precision = precision_score(Y_true, Y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    
    # Compute F1 score (for each class)
    f1 = f1_score(Y_true, Y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    
    metrics = {
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    }
    
    return metrics


