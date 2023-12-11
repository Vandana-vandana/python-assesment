import pandas as pd

############  Question 1 TASK-1 #################
def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
        where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame
    result_df = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0 and set diagonal values to 0
    result_df = result_df.fillna(0).astype(int)

    return result_df

# Example usage
df = pd.read_csv('dataset-1.csv')
result_matrix = generate_car_matrix(df)
print(result_matrix)

########### Question 2 TASK -2 #################
def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'],
                            bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'],
                            right=False)

    # Calculate count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Example usage
df = pd.read_csv('dataset-1.csv')
type_counts = get_type_count(df)
print(type_counts)

########### Question 3 task 1##############

def get_bus_indexes(dataset):
    # Calculate the mean value of the bus column
    mean_bus_value = dataset['bus'].mean()

    # Identify indices where bus values are greater than twice the mean
    bus_indexes = dataset.index[dataset['bus'] > 2 * mean_bus_value].tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage
dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
result = get_bus_indexes(df)
print(result)
    
############ Question 4 task 1 ################
def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
     # Calculate the average 'truck' values for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes with average 'truck' values greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list in ascending order
    selected_routes.sort()

    return selected_routes

# Example usage
df = pd.read_csv('dataset-1.csv')
filtered_routes = filter_routes(df)
print(filtered_routes)

############# question 5 task 1#############

def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Apply custom conditions to multiply values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage (assuming 'result_matrix' is the DataFrame from Question 1)
result_matrix = generate_car_matrix(df)  # Replace 'df' with your actual DataFrame
modified_result_matrix = multiply_matrix(result_matrix)
print(modified_result_matrix)

############### Question 6 ##############
def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
   # Combine 'startDay' and 'startTime' columns to create a start timestamp
    start_timestamp = pd.to_datetime(df['startDay'] + ' ' + df['startTime'] + ' Monday', format='%d/%m/%Y %I:%M:%S %p %A', errors='coerce')

    # Combine 'endDay' and 'endTime' columns to create an end timestamp
    end_timestamp = pd.to_datetime(df['endDay'] + ' ' + df['endTime'] + ' Monday', format='%d/%m/%Y %I:%M:%S %p %A', errors='coerce')

    # Calculate the duration of each timestamp range
    duration = end_timestamp - start_timestamp

    # Check if the duration covers a full 24-hour period and spans all 7 days
    valid_duration = (duration == pd.Timedelta(days=1)) & (start_timestamp.dt.dayofweek == end_timestamp.dt.dayofweek)

    # Create a multi-index series with (id, id_2)
    result_series = valid_duration.groupby([df['id'], df['id_2']]).all()

    return result_series

# Example usage
df_dataset2 = pd.read_csv('dataset-2.csv')
result = time_check(df_dataset2)
print(result)
