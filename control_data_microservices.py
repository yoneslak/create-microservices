import pandas as pd
import numpy as np

# Step 1: Create a users dataset
users_data = {
    'User _ID': [1, 2, 3, 4],
    'User _Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com']
}

# Step 2: Create an orders dataset
orders_data = {
    'OrderID': [101, 102, 103, 104],
    'User _ID': [1, 2, 1, 3],  # Linking to UserID
    'Product': ['Laptop', 'Smartphone', 'Tablet', 'Monitor'],
    'Quantity': [1, 2, 1, 1]
}

# Step 3: Create a transactions dataset
transactions_data = {
    'TransactionID': [1001, 1002, 1003, 1004],
    'OrderID': [101, 102, 103, 104],  # Linking to OrderID
    'Amount': [1200.00, 800.00, 300.00, 400.00],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
}

# Step 4: Convert dictionaries to DataFrames
users_df = pd.DataFrame(users_data)
orders_df = pd.DataFrame(orders_data)
transactions_df = pd.DataFrame(transactions_data)

# Step 5: Merge DataFrames
merged_df = pd.merge(orders_df, users_df, on='User _ID', how='inner')
final_df = pd.merge(transactions_df, merged_df, on='OrderID', how='inner')

# Step 6: Prepare data for training
# Encoding Product as numeric values
final_df['Product'] = final_df['Product'].astype('category').cat.codes
X = final_df[['User _ID', 'Product', 'Amount']]
y = final_df['Quantity']  # This is our target variable

# Step 7: Simple Linear Regression Implementation
# Calculate coefficients using the normal equation
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # Normal Equation

# Step 8: Function to make predictions based on user input
def predict_quantity(user_id, product, amount):
    product_code = {
        'Laptop': 0,
        'Smartphone': 1,
        'Tablet': 2,
        'Monitor': 3
    }
    
    # Prepare input data
    input_data = np.array([[1, user_id, product_code[product], amount]])  # Add bias term
    prediction = input_data.dot(theta_best)
    return prediction[0]

# Step 9: Interactive user input
if __name__ == "__main__":
    print("Welcome to the AI Prediction System!")
    
    # User input for User ID
    user_id = int(input("Enter User ID (1-4): "))
    
    # User input for Product
    product = input("Enter Product (Laptop, Smartphone, Tablet, Monitor): ")
    
    # User input for Amount with validation
    while True:
        try:
            amount = float(input("Enter Amount: "))
            break  # Exit the loop if the conversion is successful
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Make a prediction based on user input
    predicted_quantity = predict_quantity(user_id, product, amount)
    print(f"The predicted quantity for User ID {user_id} buying a {product} with amount {amount} is: {predicted_quantity:.2f}")
    