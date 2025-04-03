import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import math
import numpy as np

# DON'T WORRY ABOUT THIS CODE
def get_data():
    df = pd.read_csv("Housing.csv")
    df = df.head(200)
    df = df[["area", "price"]]
    return df["area"].to_numpy(), df["price"].to_numpy()

def main():
    x, y = get_data()
    
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    
    predicted_prices = []
    fig, ax = plt.subplots()
    for area in x:
        predicted_prices.append(model.predict(np))
    ax.plot(predicted_prices, x)
    plt.scatter(y, x)
    plt.show()    

if __name__ == "__main__":
    main()