from collections import namedtuple

import pandas as pd
import numpy as np

Point = namedtuple('Point', ['x', 'y', 'z'])

data = {
    "fixed-col1": [1, 2, 3],
    "fixed-col2": [10, 20, 40],
    "w-col-1": [
        [100, 200, 300],
        [400, 500, 600],
        [700, 800, 900]
    ],
    "w-col-2": [
        (1000, 2000, 300),
        (4000, 5000, 6000),
        (7000, 8000, 9000)
    ],
    "w-col-3": [
        Point(-1, -2, -3),
        Point(-4, -5, -6),
        Point(-7, -8, -9)
    ]
}

# Example DataFrame
data2 = {
    "col1": [1, 2, 3],
    "col2": [4, 5, 6],
    "col3": [(7, 8), (9, 10), (11, 12)]
}

if __name__ == '__main__':
    df = pd.DataFrame(data)
    print(df.head(n=5))

    # Identify columns with tuples
    tuple_columns = [col for col in df.columns if df[col].dtype == 'object']

    print(df["fixed-col1"].dtype)
    print(df["w-col-1"].dtype)
    print(df["w-col-2"].dtype)

    # Expand each tuple column into separate columns
    for col in tuple_columns:
        expanded_df = df[col].apply(pd.Series)
        expanded_df.columns = [f'{col}_{i}' for i in range(expanded_df.shape[1])]
        df = pd.concat([df.drop(columns=[col]), expanded_df], axis=1)

    # Convert the final DataFrame to a NumPy array
    arr = df.to_numpy()

    print(arr)
