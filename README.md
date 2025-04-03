### For Mac Users
The workshop consists of three practical parts. For one part, we will be using the Tensorflow python library which can be troublesome to set up on Mac systems.  
So it would be advisable to bring a windows laptop if possible.  
  
### Requirements
1. Python 3.13  
2. Scikit-Learn 1.6.1: https://scikit-learn.org/stable/index.html  
3. TensorFlow : https://www.tensorflow.org/install/pip?_gl=1*16t5nl*_up*MQ..*_ga*MTYyMDQ1MTk2LjE3NDMxOTYzNzM.*_ga_W0YLR4190T*MTc0MzE5NjM3My4xLjAuMTc0MzE5NjM3My4wLjAuMA
4. Matplotlib 3.10.1: https://matplotlib.org/stable/install/index.html  
5. Pandas 2.2.3: https://pandas.pydata.org/docs/getting_started/install.html  
6. Visual Studio Code: https://scikit-learn.org/stable/install.html  

### Linear regression: House price prediction
We will be predicting the prices of houses using linear regression. The dataset is included in the repository.
You can run the model using "python model.py"

1. We get the data
```python
def get_data():
    df = pd.read_csv("Housing.csv")
    df = df.head(200)
    df = df[["area", "price"]]
    return df["area"].to_numpy(), df["price"].to_numpy
```

2. We make predictions and calculate the cost  
```python
# Function to calculate the cost
def compute_cost(x, y, w, b):
   
    num_examples = x.shape[0] 
    cost = 0
    
    for i in range(num_examples):
        prediction = w * x[i] + b
        actual = y[i]
        loss = prediction - actual
        cost = cost + (loss)**2
    total_cost = 1 / (2 * num_examples) * cost

    return total_cost
```

3. Calculate the gradient (derivative) to minimize the cost
```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

4. Make adjustments to the weights and biases
```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(dj_dw)
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b
```

### Logistic Regression: Digit recognition

Download dataset: https://drive.google.com/file/d/1b-u6olkqvFqQwdE268_XH4Bi-MMliP_p/view?usp=sharing  
Extract the zip file and move the MNIST_Dataset folder into the "Digit Recognition" folder  
Run the training script using "python mnist.py"
