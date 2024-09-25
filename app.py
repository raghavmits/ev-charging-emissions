import streamlit as st
import pandas as pd
from google.cloud import bigquery
import plotly.express as px
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from datetime import time
import plotly.graph_objects as go
import json

# Set the app width to 100%
st.set_page_config(page_title="Charging Analysis App", layout="wide")


# Replace with your actual credentials path
credentials = service_account.Credentials.from_service_account_file("charging-analysis-ea51c9cef2b4.json")

# Set up BigQuery client
client = bigquery.Client(credentials=credentials, project='charging-analysis')

# Title of the Streamlit app
st.title("Potential Emissions Reduction via AER 🚗🔌⚡🔋🎯")
st.info("Cut down on EV charging-related emissions without any behavioral change")
with st.expander("How does it work?"):  
    st.write("""
    This app explores the potential for reducing EV-charging related CO2 emissions through load shifting. At any moment, the electricity we use 
    comes from various sources—coal, natural gas, oil, hydro, nuclear, solar, wind, and geothermal. Not all of these sources 
    are equally clean, with some contributing significantly more to pollution than others. 

    By forecasting the mix of electricity sources, we can estimate the cleanliness of the energy at a given time. 
    'Load shifting' leverages this knowledge, allowing us to shift energy consumption to periods when cleaner energy is available.
""")


query = """
SELECT 
DATETIME(TIMESTAMP(point_time), "America/Chicago") AS point_time,
moer_lb_per_mwh
FROM co2_moer.raw_values
ORDER BY point_time;
"""

@st.cache_data
def load_data(query):
    query_job = client.query(query)
    results = query_job.result()  # Get result
    print(type(results))
    df = results.to_dataframe(dtypes={'point_time': 'datetime64[ns, America/Chicago]'})

    return df

try:
    df = load_data(query)
except Exception as e:
    st.error(f"Error loading data: {e}")    

# Input features with a dictionary to store values
user_inputs = {}  # Create an empty dictionary to store user input slider values
with st.sidebar:
  st.header('EV Owner Profile')
  user_inputs['annual_mileage'] = st.slider('Annual mileage (miles)', 10000, 16000, 14000)
  user_inputs['ev_range'] = st.slider('EV range (miles)', 220, 400, 250)
  user_inputs['miles_per_kwh'] = st.slider('EV charging efficiency (miles per kWh)', 2.5, 4.5, 3.0)
  user_inputs['charger_power_rating'] = st.slider('Charger power rating (kWatt)', 7, 18, 11)

  st.markdown("---")

  st.header('EV Owner Charging Behaviour')
  user_inputs['num_charging_days_per_week'] = st.slider('No. of charging days per week', 4, 7, 6)

  default_time = time(23, 0)
  user_inputs['input_time'] = st.time_input('Select a baseline charging time between 9pm and 4am:', default_time)
#   st.info("This is an info message. You can provide helpful information to your users here.")

  st.write(f"You selected: {user_inputs['input_time']}")



# Let the transformations begin:                                                                                                                                        
# Create a custom grouping interval column
def custom_grouping_interval(dt_index):
    # Shift time by 12 hours, then floor to the nearest day, then shift back by 12 hours
   
#    # Convert to DatetimeIndex if not already
#     if not isinstance(dt_index, pd.DatetimeIndex):
#         dt_index = pd.to_datetime(dt_index)
    
    # Shift time by 12 hours, then floor to the nearest day, then shift back by 12 hours
    return (dt_index - pd.Timedelta(hours=12)).floor('D') + pd.Timedelta(hours=12)

    

def extract_night_interval(daily_dfs, start = '21:00:00', end = '06:00:00'):
    """
    Extracts the time interval from 9pm to 6am from all daily dataframes in the list.

    Args:
    - daily_dfs (list): A list of daily dataframes for the entire year.

    Returns:
    - night_dfs (list): A list of dataframes containing the time interval from 9pm to 6am for each day.
    """
    start_time = pd.Timestamp(start).time()
    end_time = pd.Timestamp(end).time()
    night_dfs = []

    for daily_df in daily_dfs:
        night_df = daily_df.between_time(start_time, end_time)
        night_dfs.append(night_df)
        # st.write(f"Checking to ensure that we are getting a dataframe: {type(night_df)}")


    return night_dfs

# Adding additional column to the dataframe which marks the lowest n values
def add_lowest_n_column_to_dfs(dfs, charging_time):
    n = int((charging_time*60)//5)

    for i, df in enumerate(dfs):
        # Ensure we are working with a copy of the DataFrame to avoid issues
        df_copy = df.copy()

        # Find the lowest n values of 'moer_lb_per_mwh'
        lowest_n = df_copy['moer_lb_per_mwh'].nsmallest(n)

        # Create a boolean mask to mark the lowest n rows
        df_copy.loc[:,'lowest_n'] = df_copy['moer_lb_per_mwh'].isin(lowest_n)

        # Assign the modified DataFrame back to the list
        dfs[i] = df_copy

    return dfs

# Adding another column to signify current charging time
def add_baseline_charging_column_to_dfs(dfs, charging_time, start_time):

    n = int((charging_time*60)//5)
    # Convert the start_time string to a datetime.time object
    # start_time = pd.to_datetime(start_time_str, format='%H:%M:%S').time()

    for i, df in enumerate(dfs):
        # Create a copy of each DataFrame in the list
        df_copy = df.copy()

        # Initialize 'baseline_charging' column with False
        df_copy['baseline_charging'] = False

        # Extract the time component from the index
        time_index = df_copy.index.time

        # Check if start_time exists in time_index
        matching_times = df_copy.index[time_index == start_time]
        # st.write(f"Matching times for {i}th df are: {matching_times}. Start_time: {start_time}")
        
        if matching_times.empty:
            st.write(f"Warning: No matching start_time found in DataFrame {i}.")
            continue  # Skip this DataFrame if the start_time is not found

        # Get the first occurrence of start_time (handles non-unique index)
        start_index = df_copy.index.get_loc(matching_times[0])

        # Ensure start_index is an integer
        if isinstance(start_index, (slice, list)):
            start_index = start_index.start  # get the starting index of the slice

        # Ensure 'n' does not exceed the length of the DataFrame
        end_idx = min(start_index + n, len(df_copy))

        # Mark the next 'n' intervals as True in 'baseline_charging'
        df_copy.iloc[start_index:end_idx, df_copy.columns.get_loc('baseline_charging')] = True

        # Assign the modified DataFrame back to the list
        dfs[i] = df_copy
    
    # st.markdown('---')
    # st.write(dfs[0].columns)
    # st.write(dfs[0].index)
    # st.dataframe(dfs[0])
    # st.markdown('---')

    return dfs

def transform_df(df, charging_time_hours):
    
    df_copy = df.copy()
    # st.write(df_copy.head())
    # Convert 'date' column to the index
    df_copy.set_index('point_time', inplace=True)
    df_copy.index.name = 'point_time'
    # st.write(df_copy.head())
    # st.write(df_copy.index)
    # st.markdown('---')

    #Grouping the time intervals based on the noon-to-noon time period to which they belong
    df_copy['grouping_interval'] = custom_grouping_interval(df_copy.index)
    # st.write(df_copy.shape)
    # st.write(df_copy.columns)
    # st.write(df_copy.head())
    # st.write(df_copy.tail())
    
    # Group by the custom grouping interval
    grouped = df_copy.groupby('grouping_interval')
    # st.write("Grouped Data:")
    # st.dataframe(grouped)

    # Extract each group into a separate DataFrame and store in a list
    daily_shifted_dataframes = [group.drop(columns=['grouping_interval']) for _, group in grouped]
    # st.write(f"Length of daily_shifted_dataframes: {len(daily_shifted_dataframes)}")
    # st.write("Observing the first df in daily_shifted_dataframes")
    # st.table(daily_shifted_dataframes[0].shape)
    # st.table(daily_shifted_dataframes[0].head())

    # Removing the first half day of the year and last half day of the year 
    daily_shifted_dfs = daily_shifted_dataframes[1:-1]
    # st.write(f"Length of daily_shifted_dfs ie. after removing first and last half days: {len(daily_shifted_dfs)}")
    # st.write("Observing the first df in daily_shifted_dfs")
    # st.table(daily_shifted_dfs[0].shape)
    # st.table(daily_shifted_dfs[0].head())
    
    # Extract night intervals from the daily shiftee dfs
    night_dfs = extract_night_interval(daily_shifted_dfs)
    # st.write("Observing the first df in night_dfs")
    # st.table(night_dfs[0].shape)
    # st.table(night_dfs[0].head())

    # Adding additional column to night_dfs which marks the lowest n values based on the charging time
    night_dfs_with_lowest_n_vals = add_lowest_n_column_to_dfs(night_dfs, charging_time_hours)
    # st.write("Observing the first df in night_dfs_with_lowest_n_vals")
    # st.table(night_dfs_with_lowest_n_vals[0].shape)
    # st.dataframe(night_dfs_with_lowest_n_vals[0])

    # Adding additional column to the dataframe which marks the baseline Charging Time interval
    # st.write(f"Input_time variable and its type are :{user_inputs['input_time']}, {type(user_inputs['input_time'])}")
    night_dfs_with_current_n_recommended_charging = add_baseline_charging_column_to_dfs(night_dfs_with_lowest_n_vals, charging_time_hours, user_inputs['input_time'])
    # st.write("Observing the dfs from night_dfs_with_current_n_recommended_charging")
    # st.table(night_dfs_with_current_n_recommended_charging[-2].shape)
    # st.dataframe(night_dfs_with_current_n_recommended_charging[-2])
    


    return night_dfs_with_current_n_recommended_charging



def get_daily_recommended_and_baseline_emissions(dfs, user_inputs, charging_time_hours):
    # Create a blank DataFrame with the specified columns
    final_df = pd.DataFrame(columns=['total_emissions_recommended_charging', 'total_emissions_baseline_charging', 'emissions_avoided'])
    final_df.index = pd.to_datetime([])
    final_df.index.name = 'date'
    power_rating = user_inputs['charger_power_rating']
    n = int((charging_time_hours*60)//5)

    for df in dfs:

        # Determining the date of the df
        day_date = df.index[0].date()

        # Extract values where 'lowest_n' is True
        lowest_n_values = df[df['lowest_n']]['moer_lb_per_mwh']
        sorted_series = lowest_n_values.sort_values()
        mean_lowest_n_values = sorted_series.head(n).mean()
        emissions_via_recommended_chargging = mean_lowest_n_values * power_rating * charging_time_hours/1000

        # Extract values where 'baseline_charging' is True
        baseline_charging_values = df[df['baseline_charging']]['moer_lb_per_mwh']
        mean_baseline_charging_values = baseline_charging_values.mean()
        emissions_via_baseline_charging = mean_baseline_charging_values * power_rating * charging_time_hours/1000

        final_df.loc[day_date] = [emissions_via_recommended_chargging, emissions_via_baseline_charging, emissions_via_baseline_charging - emissions_via_recommended_chargging]

    return final_df


# Function to plot daily time series graphs
def plot_time_series(df, day_long=False, mark_lowest_n=False, mark_baseline_charging=False):
    """
    Plots a time series graph for CO2 Marginal Emissions Rate (lb of CO2/MWh).

    Parameters:
    df (pd.DataFrame): DataFrame with 'point_time' as the index and 'moer_lb_per_mwh' as a column.
    day_long (bool): If True, add a title indicating the date of the plot.
    mark_lowest_n (bool): If True, highlight the lowest 'moer_lb_per_mwh' values.
    mark_baseline_charging (bool): If True, highlight 'moer_lb_per_mwh' values during baseline charging.
    
    Raises:
    ValueError: If 'point_time' is not in the index or if 'moer_lb_per_mwh' is not a column.
    """

    # Ensure 'point_time' is the DataFrame index
    if df.index.name != 'point_time':
        raise ValueError("The DataFrame's index must be 'point_time'.")

    # Ensure 'moer_lb_per_mwh' column exists
    if 'moer_lb_per_mwh' not in df.columns:
        raise ValueError("The DataFrame must contain a 'moer_lb_per_mwh' column.")

    # Plotting the time series
    plt.figure(figsize=(20, 8))
    plt.plot(df.index, df['moer_lb_per_mwh'], marker='o', linestyle='-', label='MOER Values')

    if day_long:
        plt.title(f'CO2 Marginal Emissions Rate (lb of CO2/MWh) for Date: {df.index[0].date()}', fontsize=20)

    # Highlight lowest 'moer_lb_per_mwh' values
    if mark_lowest_n and 'lowest_n' in df.columns:
        plt.scatter(df.index[df['lowest_n']], 
                    df['moer_lb_per_mwh'][df['lowest_n']],
                    color='green', marker='D', s=150,
                    zorder=10, alpha=0.5, label='Lowest lb/MWh Values')

        # Highlight time bands corresponding to 'lowest_n'
        for i in range(len(df) - 1):
            if df['lowest_n'].iloc[i]:
                plt.axvspan(df.index[i], df.index[i + 1], color='green', alpha=0.3)

    # Highlight baseline charging periods
    if mark_baseline_charging and 'baseline_charging' in df.columns:
        plt.scatter(df.index[df['baseline_charging']],
                    df['moer_lb_per_mwh'][df['baseline_charging']],
                    color='red', marker='D', s=150,
                    zorder=10, alpha=0.5, label='MOER Values During Baseline Charging')

        # Highlight time bands corresponding to 'baseline_charging'
        for i in range(len(df) - 1):
            if df['baseline_charging'].iloc[i]:
                plt.axvspan(df.index[i], df.index[i + 1], color='red', alpha=0.3)

    # Axis labels and formatting
    plt.xlabel('Time (from 9pm to 6am - Widow of Opportunity for charging)', fontsize=18)
    plt.ylabel('Emissions Intensity (lb of CO2/MWh)', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45, fontsize=14)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(plt)  # Use Streamlit's pyplot to display the figure
    plt.clf()  # Clear the figure after displaying to prevent overlapping in subsequent plots


# Function to plot emissions data using Plotly
def plot_emissions(df):
    fig = go.Figure()
    
    # Add traces for each emissions category
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['total_emissions_recommended_charging'], 
                             mode='lines+markers', 
                             name='Recommended Charging Emissions', 
                             line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['total_emissions_baseline_charging'], 
                             mode='lines+markers', 
                             name='Baseline Charging Emissions', 
                             line=dict(color='orange')))
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['emissions_avoided'], 
                             mode='lines+markers', 
                             name='Emissions Avoided', 
                             line=dict(color='green')))
    
    # Update layout for better readability
    fig.update_layout(
        title='EV Charging Related Emissions Data for the Year 2023',
        title_font=dict(size=24),  # Increase title font size
        xaxis_title='Date',
        yaxis_title='CO2 Emissions from charging (lb of CO2)',
        xaxis_title_font=dict(size=18),  # Increase x-axis title font size
        yaxis_title_font=dict(size=18), 
        xaxis_tickformat='%m-%d',  # Format for x-axis
        xaxis_rangeslider_visible=True,  # Enable the range slider
        height=600,  # Set the height of the graph
        font=dict(size=22),  # Set the font size for axes and legend
        legend=dict(font=dict(size=18)),  # Set the font size for legend
        template='plotly_white',
        xaxis=dict(tickfont=dict(size=16)),  # Increase x-axis tick font size
        yaxis=dict(tickfont=dict(size=16))   # Increase y-axis tick font size
    )
    
    # Show the plot in Streamlit
    st.plotly_chart(fig)

# Initialize session state for calculations and selected date
if 'charging_time_hours' not in st.session_state:
    st.session_state.charging_time_hours = None
if 'final_df_with_results' not in st.session_state:
    st.session_state.final_df_with_results = None
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None
if 'selected_transformed_df' not in st.session_state:
    st.session_state.selected_transformed_df = None
if 'transformed_dfs' not in st.session_state:
    st.session_state.transformed_dfs = None
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False

# Function to display calculated values
def display_calculated_values():
    if st.session_state.calculation_done:
        st.subheader("Charging goals based on the input EV owner profile and charging behaviour:")
        st.write(f"* Average charging miles needed per session: **{st.session_state.miles_needed_per_day:.2f} miles**")
        st.write(f"* Charging time duration: **{st.session_state.charging_time_hours:.2f} hours**")
        st.markdown('---')

# Access user input values and perform calculations
if st.button('Calculate charging time per session'):  # Add a button to trigger calculations
    st.session_state.miles_needed_per_day = user_inputs['annual_mileage'] / (user_inputs['num_charging_days_per_week'] * 52)
    st.session_state.charging_time_hours = st.session_state.miles_needed_per_day / (user_inputs['miles_per_kwh'] * user_inputs['charger_power_rating'])
    
    # # For visualizing the line chart and displaying the table
    # st.subheader("Visualizing the original table and the transformed table")
    
    # Example: Display a table
    st.session_state.transformed_dfs = transform_df(df, st.session_state.charging_time_hours)
    # st.write("An example dataframe from the transformed_dfs list of all the daily dfs")
    # st.dataframe(st.session_state.transformed_dfs[10])

    st.session_state.final_df_with_results = get_daily_recommended_and_baseline_emissions(st.session_state.transformed_dfs, user_inputs, st.session_state.charging_time_hours)
    # st.write("Dataframe detailing the baseline charging and recommended charging emissions for every day of the year")
    # st.dataframe(st.session_state.final_df_with_results)
    # st.markdown('---')

    # Set calculation_done to True
    st.session_state.calculation_done = True

# Display calculated values
display_calculated_values()

# Estimating the final results
if st.session_state.calculation_done:
    st.subheader("Estimated savings: Baseline Charging Approach v/s Recommended Charging Approach")
    multiplication_factor = user_inputs['num_charging_days_per_week'] * 52
    annual_emissions_recommended_charging = st.session_state.final_df_with_results['total_emissions_recommended_charging'].mean() * multiplication_factor
    annual_emissions_baseline_charging = st.session_state.final_df_with_results['total_emissions_baseline_charging'].mean() * multiplication_factor
    annual_emissions_avoided = annual_emissions_baseline_charging - annual_emissions_recommended_charging
    percentage_emissions_avoided = 100 * (annual_emissions_avoided) / annual_emissions_baseline_charging
    st.write(f"* Total annual emissions associated with Baseline Charging Approach: **{annual_emissions_baseline_charging:.2f} lb of CO2**")
    st.write(f"* Total annual emissions associated with Recommended Charging Approach: **{annual_emissions_recommended_charging:.2f} lb of CO2**")
    st.write(f"* Total annual emissions avoided: **{annual_emissions_avoided:.2f} lb of CO2**")
    st.write(f"* % Emissions avoided: **{percentage_emissions_avoided:.2f}%**")
    annual_emissions_avoided_tree_equivalent = annual_emissions_avoided / 48
    st.info(f"For perspective, the total CO2 avoided is approximately equal to the CO2 absorbed by {annual_emissions_avoided_tree_equivalent:.1f} mature trees over the course of a year")

with st.expander('Visualize the daily Marginal Operating Emisisons Rate at night'):
    if st.session_state.calculation_done:
        # Assuming transformed_dfs is a list of DataFrames and each DataFrame's index is a datetime object
        # and there's a corresponding list of dates for the DataFrames
        dates = [df.index[0].date() for df in st.session_state.transformed_dfs]  # Get the list of dates from the DataFrames

        # Create a date input for user to select a date
        st.session_state.selected_date = st.date_input("Select a date:", value=pd.to_datetime(st.session_state.transformed_dfs[0].index[0]).date())

        # Automatically update the selected transformed DataFrame when the date is changed
        if st.session_state.selected_date in dates:
            # Find the index of the selected date
            index_of_selected_date = dates.index(st.session_state.selected_date)
            # Get the corresponding DataFrame
            st.session_state.selected_transformed_df = st.session_state.transformed_dfs[index_of_selected_date]
            # Display the DataFrame or pass it to your plotting function
            st.dataframe(st.session_state.selected_transformed_df)  
            plot_time_series(st.session_state.selected_transformed_df, day_long=True, mark_lowest_n=True, mark_baseline_charging=True)


        else:
            st.error("No data available for the selected date.")
    else:
        st.warning("Please calculate the charging time first.")



with st.expander('Visualize the daily emissions associated with baseline charging approach v/s recommended charging approach'):
    if st.session_state.calculation_done:
        st.dataframe(st.session_state.final_df_with_results)
        plot_emissions(st.session_state.final_df_with_results)
    else:
        st.warning("Please calculate the charging time first.")
