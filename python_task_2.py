import pandas as pd

################# Question 1 ######################
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Create a pivot table to represent distances between tollbooths
    distance_matrix = df.pivot_table(index='id_start', columns='id_end', values='distance', aggfunc='sum', fill_value=0)

    # Convert the pivot table to a symmetric matrix
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix

# Example usage
dataset_path = 'dataset-3.csv'
result_distance_matrix = calculate_distance_matrix(dataset_path)
print(result_distance_matrix)


################# Question 2 ######################
def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Reset index to get 'id_start' as a column
    distance_matrix_reset = distance_matrix.reset_index()

    # Melt the DataFrame to create 'id_start', 'id_end', and 'distance' columns
    unrolled_df = pd.melt(distance_matrix_reset, id_vars='id_start', var_name='id_end', value_name='distance')

    # Exclude rows where 'id_start' is equal to 'id_end'
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    return unrolled_df

# Example usage
# Assuming result_distance_matrix is the DataFrame from the previous question
result_unrolled_df = unroll_distance_matrix(result_distance_matrix)
print(result_unrolled_df)
#############  Question 3 ############################

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Create a pivot table to represent distances between sources and destinations
    distance_matrix = df.pivot_table(index='id_start', columns='id_end', values='distance', aggfunc='sum', fill_value=0)
    
    # Convert the pivot table to a symmetric matrix
    distance_matrix = distance_matrix + distance_matrix.T
    
    return distance_matrix

# Example usage
df_dataset3 = pd.read_csv('dataset-3.csv')
result_distance_matrix = calculate_distance_matrix(df_dataset3)
print(result_distance_matrix)
################# Question 4 #########################

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    
    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = np.random.randint(0, 100, size=len(df)) * rate_coefficient
    
    return df

# Example usage
# Assuming unrolled_result is the DataFrame from the previous question
# Replace it with the actual DataFrame variable you have.
unrolled_result_data = {
    'id_start': [1001400, 1001400, 1001400, 1001400, 1001400, 1001400, 1001400, 1001400, 1001400, 1001400],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420],
}

unrolled_result = pd.DataFrame(unrolled_result_data)

# Call the calculate_toll_rate function
result_with_toll_rate = calculate_toll_rate(unrolled_result)

# Print the result
print(result_with_toll_rate)
################ Question 5 #########################
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define time ranges and discount factors
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    
    # Convert start_time and end_time columns to datetime.time
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time
    
    # Apply discount factors based on time ranges
    for start_time, end_time, discount_factor in time_ranges:
        # Filter rows based on time range
        mask = (df['start_time'] >= start_time) & (df['end_time'] <= end_time)
        df.loc[mask, 'moto':'truck'] *= discount_factor
    
    # Apply constant discount factor for weekends
    weekend_mask = (df['start_day'].isin(['Saturday', 'Sunday']))
    df.loc[weekend_mask, 'moto':'truck'] *= 0.7
    
    return df

# Example usage
# Assuming result_with_toll_rate is the DataFrame from the previous question
# Replace it with the actual DataFrame variable you have.
result_with_toll_rate_data = {
    'id_start': [1001400, 1001400, 1001400, 1001400, 1001408, 1001408, 1001408, 1001408],
    'id_end': [1001402, 1001402, 1001402, 1001402, 1001410, 1001410, 1001410, 1001410],
    'distance': [9.7, 9.7, 9.7, 9.7, 11.1, 11.1, 11.1, 11.1],
    'moto': [8, 16, 8, 24, 16, 24, 7.10, 6.22],
    'car': [12, 24, 12, 36, 24, 36, 10.66, 9.32],
    'rv': [15, 30, 15, 45, 30, 45, 13.32, 11.66],
    'bus': [22, 44, 22, 66, 44, 66, 19.54, 17.09 ],
    'truck': [36, 72, 36, 108, 72, 108, 31.97, 27.97],
    'start_day': ['Monday', 'Tuesday', 'Wednesday', 'Saturday', 'Monday', 'Tuesday', 'Wednesday', 'Saturday'],
    'end_day': ['Friday', 'Saturday','Sunday','Sunday', 'Friday', 'Saturday','Sunday','Sunday'],
    'start_time': ['00:00:00', '10:00:00', '18:00:00', '00:00:00', '00:00:00', '10:00:00','18:00:00','00:00:00'],
    'end_time': ['10:00:00', '18:00:00', '23:59:59', '23:59:59', '10:00:00', '18:00:00','23:59:59','23:59:59']
}

result_with_toll_rate = pd.DataFrame(result_with_toll_rate_data)

# Call the calculate_time_based_toll_rates function
result_with_time_based_rates = calculate_time_based_toll_rates(result_with_toll_rate)

# Print the result
print(result_with_time_based_rates)
