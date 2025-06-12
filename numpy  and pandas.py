import numpy as np
import pandas as pd

# Task #1: Define single and multi-dimensional NumPy arrays
print("Task #1: NumPy Arrays")
# Single-dimensional array
arr1d = np.array([1, 2, 3, 4, 5])
print("Single-dimensional array:\n", arr1d)

# Multi-dimensional array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Multi-dimensional array:\n", arr2d)

# Task #2: Leverage NumPy built-in methods and functions
print("\nTask #2: NumPy Built-in Methods")
# Create an array of zeros
zeros_arr = np.zeros((3, 3))
print("Array of zeros:\n", zeros_arr)

# Create an array of ones
ones_arr = np.ones((2, 4))
print("Array of ones:\n", ones_arr)

# Create an array with a range of values
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print("Array with a range of values:\n", range_arr)

# Reshape an array
reshaped_arr = arr1d.reshape((5, 1))
print("Reshaped array:\n", reshaped_arr)

# Find the maximum value
max_val = np.max(arr2d)
print("Maximum value in arr2d:", max_val)

# Find the index of the maximum value
max_index = np.argmax(arr2d)
print("Index of maximum value in arr2d:", max_index)


# Task #3: Perform mathematical operations in NumPy
print("\nTask #3: Mathematical Operations")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Addition
addition = arr1 + arr2
print("Addition:", addition)

# Subtraction
subtraction = arr2 - arr1
print("Subtraction:", subtraction)

# Multiplication
multiplication = arr1 * arr2
print("Multiplication:", multiplication)

# Division
division = arr2 / arr1
print("Division:", division)

# Square root
sqrt_arr1 = np.sqrt(arr1)
print("Square root of arr1:", sqrt_arr1)

# Task #4: Perform array slicing and indexing
print("\nTask #4: Array Slicing and Indexing")
arr = np.array([10, 20, 30, 40, 50])

# Accessing elements by index
print("Element at index 0:", arr[0])

# Slicing
print("Slice from index 1 to 3:", arr[1:4])

# Slicing with a step
print("Slice from index 0 to 4 with a step of 2:", arr[0:5:2])

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Element at row 0, column 1:", arr2d[0, 1])
print("Row 1:", arr2d[1, :])
print("Column 2:", arr2d[:, 2])


# Task #5: Perform elements selection (conditional)
print("\nTask #5: Conditional Selection")
arr = np.array([1, 2, 3, 4, 5, 6])

# Select elements greater than 3
greater_than_3 = arr[arr > 3]
print("Elements greater than 3:", greater_than_3)

# Select elements that are even
even_numbers = arr[arr % 2 == 0]
print("Even numbers:", even_numbers)


# Task #6: Understand pandas fundamentals
print("\nTask #6: Pandas Fundamentals")
# Creating a Series
data = [10, 20, 30, 40, 50]
series = pd.Series(data)
print("Pandas Series:\n", series)

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 28],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)
print("Pandas DataFrame:\n", df)

# Accessing columns
print("Age column:\n", df['Age'])

# Accessing rows
print("First row:\n", df.iloc[0])


# Task #7: Pandas with CSV and HTML data
print("\nTask #7: Pandas with CSV and HTML Data")
# Assuming you have a CSV file named 'data.csv' in the same directory
# Create a dummy csv file
dummy_data = {'col1': [1, 2], 'col2': [3, 4]}
dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv('data.csv', index=False)

try:
    csv_df = pd.read_csv('data.csv')
    print("DataFrame from CSV:\n", csv_df)
except FileNotFoundError:
    print("data.csv not found.  Please create a dummy data.csv file.")

# Reading HTML data (requires internet connection and proper HTML table structure)
# Example: Reading a table from a webpage
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' #Example URL
    html_tables = pd.read_html(url)
    sp500_table = html_tables[0] # Assuming the first table is the S&P 500 list
    print("DataFrame from HTML:\n", sp500_table.head()) # Print the first few rows
except Exception as e:
    print(f"Error reading HTML data: {e}.  Check internet connection and URL.")


# Task #8: Pandas Operations
print("\nTask #8: Pandas Operations")
# Adding a new column
df['Salary'] = [60000, 70000, 65000]
print("DataFrame with Salary column:\n", df)

# Dropping a column
df = df.drop('City', axis=1)
print("DataFrame after dropping City column:\n", df)

# Filtering rows
filtered_df = df[df['Age'] > 27]
print("Filtered DataFrame (Age > 27):\n", filtered_df)

# Grouping data
grouped_df = df.groupby('Age')['Salary'].mean()
print("Grouped DataFrame (Average salary by age):\n", grouped_df)


# Task #9: Pandas with Functions
print("\nTask #9: Pandas with Functions")
# Applying a function to a column
def bonus(salary):
    return salary * 0.1

df['Bonus'] = df['Salary'].apply(bonus)
print("DataFrame with Bonus column:\n", df)

# Applying a lambda function
df['Tax'] = df['Salary'].apply(lambda x: x * 0.2)
print("DataFrame with Tax column:\n", df)


# Task #10: Perform sorting and ordering in pandas
print("\nTask #10: Sorting and Ordering")
# Sorting by a single column
sorted_df = df.sort_values(by='Age')
print("DataFrame sorted by Age:\n", sorted_df)

# Sorting by multiple columns
sorted_df = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])
print("DataFrame sorted by Age (ascending) and Salary (descending):\n", sorted_df)


# Task #11: Perform concatenating and merging with pandas
print("\nTask #11: Concatenating and Merging")
# Creating two DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [4, 5, 6], 'Name': ['D', 'E', 'F']})

# Concatenating DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)
print("Concatenated DataFrame:\n", concatenated_df)

# Merging DataFrames
df3 = pd.DataFrame({'ID': [1, 2, 3], 'Salary': [50000, 60000, 70000]})
merged_df = pd.merge(df1, df3, on='ID')
print("Merged DataFrame:\n", merged_df)


# Task #12: Project and concluding remarks
print("\nTask #12: Project and Concluding Remarks")
print("This code demonstrates fundamental NumPy and Pandas operations.")
print("You can expand upon this by working with real-world datasets, performing more complex analysis, and visualizing the results.")
print("Remember to explore the extensive documentation for both libraries to unlock their full potential.")
