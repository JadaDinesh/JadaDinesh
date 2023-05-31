# Import necessary libraries
import pandas as pnd
from utilities.conn import SQLLiteUtils
from bokeh.models import Legend
from bokeh.plotting import figure, show
from bokeh.palettes import Spectral4
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource


# Get connect to the SQL DB
sql_connectionString = "sqlite:///data.db"


def ide_function(ideal_data, train_data):
    """
    The function selects the four ideal functions out of the provided 50 functions by calculating the deviation
    between each training data column and each ideal data column, and selecting the ideal function and deviation
    each column and the function returns the four selected ideal functions and their maximum deviation values.

    Parameters:
    ideal_data (DataFrame): Dataframe containing the 50 ideal functions
    train_data (DataFrame): Dataframe containing the training data

    Returns:
    Dictionary containing two DataFrames:
        - 'ideal_data1': Dataframe containing the four selected ideal functions and their corresponding x values
        - 'ideal_max': Dataframe containing the maximum deviation values for each of the four selected ideal functions
    """

    # Rename columns to avoid conflicts during merge
    # ideal_data = ideal_data.rename(columns={'y': 'y_ideal'})
    # train_data = train_data.rename(columns={'y': 'y_train'})
    # Rename columns to distinguish between y values in the training and ideal data sets
    #ideal_data.columns = ideal_data.columns.map(lambda x: x.replace('y', 'y_ideal'))
    #train_data.columns = train_data.columns.map(lambda x: x.replace('y', 'y_train'))

    # Replace 'y' with 'train_y' in train_data columns
    train_data = train_data.rename(columns=lambda x: x.replace('y', 'train_y'))
    
    # Replace 'y' with 'ideal_y' in ideal_data columns
    ideal_data = ideal_data.rename(columns=lambda x: x.replace('y', 'ideal_y'))

    # Merge train_data and ideal_data on 'x' using inner join
    merged_df = pnd.merge(train_data, ideal_data, how='inner', on='x')
    print(merged_df)
    # Initialize empty DataFrames to hold the computed results
    idealFuncs = pnd.DataFrame()
    max_dev = pnd.DataFrame()

    # Iterate over all the columns in merged_df that have a prefix of 'train_' in their names.
    for idx, train_col in enumerate([col for col in merged_df.columns if col.startswith('train_')]):

        # Generate a DataFrame temporarily
        temp_dataframe = pnd.DataFrame()

        # Iterate over all the columns in merged_df that have a prefix of 'ideal_' in their names.
        for ideal_col in [col for col in merged_df.columns if col.startswith('ideal_')]:

            # Calculate the least squares difference between train_col and ideal_col
            temp_dataframe[f"{ideal_col}_ls"] = (merged_df[train_col] - merged_df[ideal_col]) ** 2

        # Determine the minimum value column name and add it to the idealFuncs DataFrame
        min_col = str(temp_dataframe.sum().idxmin()).split("_")[1]
        idealFuncs[[min_col]] = merged_df[["ideal_" + min_col]]

        # Calculate the maximum deviation for the minimum value column and store it in a separate DataFrame
        max_dev[min_col] = [temp_dataframe[f"ideal_{min_col}_ls"].max() ** 0.5]

    # Insert 'x' column to the beginning of idealFuncs DataFrame
    idealFuncs.insert(loc=0, column='x', value=merged_df['x'])
    
    #print(temp_dataframe)
    #print(max_dev)
    # Reorder columns in the idealFuncs DataFrame and return results
    #idealFuncs = idealFuncs[['x', f'ideal_{min_col}']]
    return {'ideal_data1': idealFuncs, 'ideal_max': max_dev}



def test_function(test_data, ideal_data, max_dev):
    """
    This function calculates the deviation between each test column and each ideal column, and selects the ideal
    :param test: Test dataset
    :param ideal: Ideal dataset
    :param maximum_deviation: DataFrame to store maximum deviation for each selected ideal function
    :return: DataFrame with mapping and deviation
    """
    # Calculate deviation between each test column and each ideal column
    merge_df = test_data.merge(ideal_data, on=['x'], how='left')
    result_df = pnd.DataFrame(columns=['x', 'ideal_func', 'delta_y', 'ideal_y'])

    def get_ideal_func(row, max_dev):
        optimal_function = None
        deviation_threshold = float('inf')
        for col in max_dev.columns:
            delta_y = abs(row['y'] - row[col])
            # The function should only be assigned if its delta is below a certain threshold.
            # A function can only be assigned if its deviation is less than or equal to the maximum deviation multiplied by the square root of 2 
            if max_dev[col][0] * (1.4142135623730951) >= delta_y and deviation_threshold > delta_y:
                deviation_threshold = delta_y
                optimal_function = col
        return optimal_function, deviation_threshold if deviation_threshold < float('inf') else None, row[optimal_function] if optimal_function else None

    # loop over rows in the merged table and get ideal function and other data
    result_df[['ideal_func', 'delta_y', 'ideal_y']] = merge_df.apply(lambda row: pnd.Series(get_ideal_func(row, max_dev)), axis=1)
    result_df['x'] = merge_df['x']
    
    return result_df
    


def plot_ideal_function(ideal_data, train_data, deviation_data, test_data):
    """
    This function plots four different dataframes on a single Bokeh figure.
    The four dataframes are the ideal function, training data with predicted y values,
    deviation from the ideal function for training data, and test data with predicted y values.

    Parameters:
    ideal_data: pandas.DataFrame
    A dataframe with x and y values for the ideal function.
    train_data: pandas.DataFrame
    A dataframe with x values and predicted y values for the training data.
    deviation_data: pandas.DataFrame
    A dataframe with x values and deviation from the ideal function for the training data.
    test_data: pandas.DataFrame
    A dataframe with x values and predicted y values for the test data.

    Returns:
    None
    """

    # reate the dataframes with the specified parameters and data.
    test_data = pnd.DataFrame({'x': [0, 1, 2], 'y1': [1, 2, 3], 'y2': [2, 3, 4], 'y3': [3, 4, 5], 'y4': [4, 5, 6]})
    train_data = pnd.DataFrame({'x': [0, 1, 2], 'train_y1': [1, 2, 3], 'train_y2': [2, 3, 4], 'train_y3': [3, 4, 5], 'train_y4': [4, 5, 6]})
    ideal_data = pnd.DataFrame({'x': [0, 1, 2], 'y': [1, 2, 3], 'ideal_func': ['y1', 'y2', 'y3']})
    deviation_data = pnd.DataFrame({'max_deviation': [1, 2, 3, 4]})
    
    # Create a list containing different colors.
    chart_colors = Spectral4
    
    # Define the plots that will be used to visualize the data
    visualize = []
    for i, col in enumerate(test_data.columns[1:]):
        fig = figure(title=f"Training function {i+1}")
        fig.circle(test_data['x'], test_data[col], color=chart_colors[i], legend_label=f'Ideal Function {i+1}')
        fig.line(train_data['x'], train_data[f'train_{col}'], color=chart_colors[i], legend_label=f'Training {i+1}')
        visualize.append(fig)
    
    # Graph illustrating the performance of training and test data against ideal functions
    perf_fig5  = figure(title="Test_ Vs Training Ideal Data")
    perf_fig5 .circle(ideal_data['x'], ideal_data['y'], color='blue', legend_label='Test_')
    for j, cols in enumerate(deviation_data.columns):
        ideal_copy_test = ideal_data.copy()
        ideal_copy_test.loc[ideal_data['ideal_func'] != cols, 'ideal_y'] = None
        perf_fig5 .circle(ideal_data['x'], ideal_copy_test['ideal_y'], color=chart_colors[j], legend_label=f'Ideal function {cols}')
    
    # A graph displaying the maximum deviations between the actual training data and the predicted ideal functions.
    perf_fig6 = figure(title="Max. Deviations")
    y = deviation_data['max_deviation']
    deviation_src = ColumnDataSource(data=dict(left = [1, 2, 3, 4], color=chart_colors, counts=y))
    perf_fig6.vbar(x='left', top='counts', width=1/2, legend_field="counts", color='color', source=deviation_src)
    
    # cCreate a collection of figures arranged in a grid.
    gridplot_1 = gridplot([[visualize[0], visualize[1]], [visualize[2], visualize[3]], [perf_fig5, perf_fig6]], width=1200, height=600)
    show(gridplot_1)
    

# Set up the database connection
if __name__ == "__main__":
    # Set up the database connection
    conn = SQLLiteUtils(string_connection=sql_connectionString)

    # Read CSV Files
    train_data_df = pnd.read_csv('data_sets/train_data.csv')
    test_data_df = pnd.read_csv('data_sets/test_data.csv')
    ideal_data_df = pnd.read_csv('data_sets/ideal_data.csv')

    # Save the data to corresponding database tables
    tables = {'test': test_data_df, 'ideal': ideal_data_df, 'train': train_data_df}
    print(tables['train'])
    for table_name, df in tables.items():
         conn.put_df(df, table_name, string_connection=sql_connectionString)

    # Generate ideal functions and maximum deviation
    deviation_function = ide_function(ideal_data_df, train_data_df)

    # print(deviation_function['ideal_max'])
    # print(deviation_function['ideal_data1'])
    
    # Map test data to the ideal function and calculate y-deviation
    mapping_df = test_function(test_data_df, deviation_function['ideal_data1'], deviation_function['ideal_max'])
    print(mapping_df)
    # Plot the ideal function and test data
    plot_ideal_function(deviation_function['ideal_data1'], train_data_df, deviation_function['ideal_max'], mapping_df)

    # Filter the DataFrame and keep only specific columns
    mapping_df = mapping_df[['x', 'ideal_y', 'delta_y', 'ideal_func']]

    # Save the filtered DataFrame to a database table
    conn.put_df(df=mapping_df, table='test_map', string_connection=sql_connectionString)
