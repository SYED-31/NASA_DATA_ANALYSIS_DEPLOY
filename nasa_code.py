# !pip install -r requirements.txt
import pandas as pd
import os
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import plotly.express as px

# Load data
metadata_df = pd.read_csv("metadata.csv")
file_df = metadata_df[metadata_df['type'] == 'impedance'][['start_time', 'filename', 'battery_id']]

# Filter for impedance rows and select necessary columns
impedance_df = metadata_df[metadata_df['type'] == 'impedance'][['start_time', 'Re', 'Rct', 'battery_id']]

# Get the unique battery_ids from the 'battery_id' column
battery_ids = impedance_df['battery_id'].unique()

# If you prefer to get the list in a more readable format, you can convert it to a list
battery_ids_list = battery_ids.tolist()

# Function to parse start_time
def parse_start_time(value):
    try:
        if isinstance(value, str):
            value = value.strip("[]").replace(",", "")  # Remove brackets and commas
            components = [float(x) for x in value.split()]  # Split and convert to float
            if len(components) == 6:
                year, month, day, hour, minute = map(int, components[:5])
                second = int(components[5])  # Handle fractional seconds
                return datetime(year, month, day, hour, minute, second)
        elif isinstance(value, (list, np.ndarray)) and len(value) == 6:
            year, month, day, hour, minute = map(int, value[:5])
            second = int(float(value[5]))  # Handle fractional seconds
            return datetime(year, month, day, hour, minute, second)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Failed to parse: {value}, Error: {e}")
    return pd.NaT

def parse_Re_or_Rct(value):
    return abs(complex(value))

# Apply parsing function
impedance_df['start_time'] = impedance_df['start_time'].apply(parse_start_time)
file_df['start_time'] = file_df['start_time'].apply(parse_start_time)
impedance_df['Re'] = impedance_df['Re'].apply(parse_Re_or_Rct)
impedance_df['Rct'] = impedance_df['Rct'].apply(parse_Re_or_Rct)


# Drop rows with invalid start_time
impedance_df = impedance_df.dropna(subset=['start_time'])

# Sort data by start_time
impedance_df = impedance_df.sort_values(by='start_time')

# Drop rows with invalid start_time
file_df = file_df.dropna(subset=['start_time'])

# Sort data by start_time
file_df = file_df.sort_values(by='start_time')

# Initialize an empty dictionary to store the impedance data for each battery
battery_data = {}

# Iterate through each row of file_df to process each file
for index, row in file_df.iterrows():
    battery_id = row['battery_id']
    filename = row['filename']
    start_time = row['start_time']

    # Read the CSV file using the filename
    file_path = f"data/{filename}"  # Adjust the path as necessary
    try:
        data = pd.read_csv(file_path)
        
        # Check if 'battery_impedance' column exists in the file
        if 'Battery_impedance' in data.columns:
            # Extract real part of the complex impedance and calculate the average
            # Here, we assume the Battery_impedance column contains string representations of complex numbers
            impedance_values = data['Battery_impedance'].apply(lambda x: abs(complex(x)))
            avg_impedance = impedance_values.mean()

            # Store the average impedance in the battery_data dictionary
            if battery_id not in battery_data:
                battery_data[battery_id] = {'start_time': [], 'impedance': []}
            
            battery_data[battery_id]['start_time'].append(start_time)
            battery_data[battery_id]['impedance'].append(avg_impedance)
        else:
            print(f"'Battery_impedance' column not found in {filename}")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")


# Set plot type

# Line Plot On Impedance
def line_plot_impedance(battery_id_input):
    # Filter the data for the given battery_id
    battery_info = battery_data.get(battery_id_input, None)
    
    # Check if the battery data exists for the provided battery_id
    if battery_info is None:
        print(f"No data available for battery_id {battery_id_input}")
        return
    
    # Create the figure for impedance
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=battery_info['start_time'], y=battery_info['impedance'], mode='lines+markers', name=f'Battery {battery_id_input}'))

    # Update layout for better readability
    fig.update_layout(
        title=f'Impedance Change Over Time for Battery {battery_id_input}',
        xaxis_title='Start Time',
        yaxis_title='Magnitude of Impedance (Ohms)',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()

# Scatter Plot On Impedance
def scatter_plot_impedance(battery_id_input):
    # Filter the data for the given battery_id
    battery_data_filtered = battery_data.get(battery_id_input, None)
    
    # Check if the battery data exists for the provided battery_id
    if battery_data_filtered is None:
        print(f"No data available for battery_id {battery_id_input}")
        return
    
    # Create the scatter plot for impedance
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'], 
        mode='markers',  # Use 'markers' for a scatter plot
        name=f'Battery {battery_id_input}',
        marker=dict(size=8, color='blue', opacity=0.7)  # Customizing marker appearance
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f'Impedance Change Over Time for Battery {battery_id_input}',
        xaxis_title='Start Time',
        yaxis_title='Magnitude of Impedance (Ohms)',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()

# Bar Plot On Impedance
def bar_plot_impedance(battery_id_input):
    # Check if the provided battery_id exists in the data
    battery_data_filtered = battery_data.get(battery_id_input, None)
    
    # If no data is found for the battery_id, show a message and return
    if battery_data_filtered is None:
        print(f"No data available for battery_id {battery_id_input}")
        return
    
    # Create the bar plot for impedance
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'], 
        name=f'Battery {battery_id_input}',
        marker=dict(color='blue', opacity=0.7)  # Customizing bar appearance
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f'Impedance Change Over Time for Battery {battery_id_input}',
        xaxis_title='Start Time',
        yaxis_title='Magnitude of Impedance (Ohms)',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Line Plot On Re
def line_plot_Re(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the line plot for the real part of the impedance (Re)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Use 'impedance' here, as 'Re' data is stored as impedance
        mode='lines+markers', 
        name='Re',
        line=dict(color='blue')
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Real Part of Impedance (Re) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()

# Scatter Plot On Re
def scatter_plot_Re(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the scatter plot for the real part of the impedance (Re)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Use 'impedance' here instead of 'Re'
        mode='markers',  # Scatter plot mode
        name='Re',
        marker=dict(color='blue', size=8)  # Customizing marker appearance
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Real Part of Impedance (Re) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Bar Plot On Re
def bar_plot_Re(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the bar plot for the real part of the impedance (Re)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Use 'impedance' here instead of 'Re'
        name='Re',  # Bar plot for Real part of impedance
        marker=dict(color='blue')  # Customizing bar color
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Real Part of Impedance (Re) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Line Plot On Rct
def line_plot_Rct(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the line plot for the charge transfer resistance (Rct)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Using 'impedance' instead of 'Rct'
        mode='lines+markers', 
        name='Rct',  # This name is for the legend, but you might want to change it if Rct is derived
        line=dict(color='red')  # Customizing line color
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Charge Transfer Resistance (Rct) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Scatter Plot On Rct
def scatter_plot_Rct(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the scatter plot for the charge transfer resistance (Rct)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Using 'impedance' instead of 'Rct'
        mode='markers',  # Scatter plot mode
        name='Rct',  # This name is for the legend, you may change it if Rct is derived differently
        marker=dict(color='red', size=8)  # Customizing marker appearance
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Charge Transfer Resistance (Rct) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Bar Plot On Rct
def bar_plot_Rct(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]
    
    # Check if battery_data_filtered has any data
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the bar plot for the charge transfer resistance (Rct)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Using 'impedance' instead of 'Rct'
        name='Rct',  # This name is for the legend, you may change it if Rct is derived differently
        marker=dict(color='red')  # Customizing bar color
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Charge Transfer Resistance (Rct) Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()



# Line Plot On Combined Resistance
def line_plot_combined_resistance(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]

    # Check if there is data available for the selected battery_id
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the figure
    fig = go.Figure()

    # Plot the real part of the impedance (Re)
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Assuming 'impedance' is a real part value here
        mode='lines+markers', 
        name='Re',  # You can adjust the label as needed
        line=dict(color='blue')
    ))

    # Plot the charge transfer resistance (Rct) if you have it or derive it
    # If you have a derived Rct value, replace 'impedance' with Rct
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Assuming 'impedance' for now, replace with Rct if available
        mode='lines+markers', 
        name='Rct',  # Same here, adjust if needed
        line=dict(color='red')
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Resistance Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


# Scatter Plot On Combined Resistance
def scatter_plot_combined_resistance(battery_id_input):
    # Check if the provided battery_id exists in the data
    if battery_id_input not in battery_data:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Filter the data for the selected battery_id
    battery_data_filtered = battery_data[battery_id_input]

    # Check if there is data available for the selected battery_id
    if not battery_data_filtered['start_time'] or not battery_data_filtered['impedance']:
        print(f"No impedance data available for battery_id {battery_id_input}")
        return

    # Create the figure
    fig = go.Figure()

    # Scatter plot for the real part of the impedance (Re)
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Assuming 'impedance' is real part for now
        mode='markers',  # Only markers for scatter plot
        name='Re',
        marker=dict(color='blue')  # Customize the color
    ))

    # Scatter plot for the charge transfer resistance (Rct)
    # If 'Rct' data is available, replace 'impedance' with the Rct column.
    # For now, assuming impedance is also used for 'Rct' or needs to be derived.
    fig.add_trace(go.Scatter(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['impedance'],  # Assuming 'impedance' or 'Rct' if available
        mode='markers',  # Only markers for scatter plot
        name='Rct',
        marker=dict(color='red')  # Customize the color
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Resistance Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
    )
    
    # Display the plot
    fig.show()


def bar_plot_combined_resistance(battery_id_input):
    # Filter the data for the provided battery_id
    battery_data_filtered = impedance_df[impedance_df['battery_id'] == battery_id_input]
    
    # If no data is found for the provided battery_id, show a message and return
    if battery_data_filtered.empty:
        print(f"No data available for battery_id {battery_id_input}")
        return

    # Create the figure
    fig = go.Figure()

    # Bar plot for the real part of the impedance (Re)
    fig.add_trace(go.Bar(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['Re'], 
        name='Re',
        marker=dict(color='blue')  # Customize the color for Re
    ))

    # Bar plot for the charge transfer resistance (Rct)
    fig.add_trace(go.Bar(
        x=battery_data_filtered['start_time'], 
        y=battery_data_filtered['Rct'], 
        name='Rct',
        marker=dict(color='red')  # Customize the color for Rct
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Resistance Change Over Time for Battery ID: {battery_id_input}",
        xaxis_title="Start Time",
        yaxis_title="Magnitude of Resistance (Ohms)",
        xaxis=dict(tickangle=45),  # Rotate x-axis labels for better visibility
        legend_title="Resistance Type",
        legend_font_size=16,
        template="plotly",
        height=1000,
        width=1500,
        barmode='group'  # Ensures bars for Re and Rct are grouped together for each time point
    )

    # Display the plot
    fig.show()



#Resistance over All batteries
impedance_metadata = metadata_df[metadata_df['type'] == 'impedance'][['start_time', 'Re', 'Rct', 'battery_id']]

def plot_impedance_trends(data):
    # Initialize the figure
    aggregated_fig = go.Figure()

    # Loop through each unique battery ID
    for battery_id in data['battery_id'].unique():
        battery_data = data[data['battery_id'] == battery_id]
        
        # Plot Real part of impedance (Re)
        aggregated_fig.add_trace(go.Scatter(
            x=battery_data['start_time'],
            y=battery_data['Re'],
            mode='lines+markers',
            name=f'Re (Battery {battery_id})',
            line=dict(width=2, color='blue'),
            marker=dict(size=6, color='blue'),
            hovertemplate='<b>Battery ID:</b> %{text}<br>' +
                          '<b>Re:</b> %{y:.2f} Ohms<br>' +
                          '<b>Time:</b> %{x}<extra></extra>',
            text=[f'Battery {battery_id}'] * len(battery_data)  # Hover text
        ))

        # Plot Charge Transfer Resistance (Rct)
        aggregated_fig.add_trace(go.Scatter(
            x=battery_data['start_time'],
            y=battery_data['Rct'],
            mode='lines+markers',
            name=f'Rct (Battery {battery_id})',
            line=dict(dash='dot', width=2, color='red'),
            marker=dict(size=6, color='red'),
            hovertemplate='<b>Battery ID:</b> %{text}<br>' +
                          '<b>Rct:</b> %{y:.2f} Ohms<br>' +
                          '<b>Time:</b> %{x}<extra></extra>',
            text=[f'Battery {battery_id}'] * len(battery_data)  # Hover text
        ))

        # Ensure max values are numeric
        max_re = float(battery_data['Re'].max())
        max_rct = float(battery_data['Rct'].max())

        # Optionally, annotate max/min points for both Re and Rct
        max_re_time = battery_data['start_time'][battery_data['Re'].idxmax()]
        aggregated_fig.add_annotation(
            x=max_re_time, y=max_re,
            text=f'Max Re: {max_re:.2f} Ohms',
            showarrow=True, arrowhead=2,
            font=dict(size=12, color='blue'),
            bgcolor='white'
        )
        
        max_rct_time = battery_data['start_time'][battery_data['Rct'].idxmax()]
        aggregated_fig.add_annotation(
            x=max_rct_time, y=max_rct,
            text=f'Max Rct: {max_rct:.2f} Ohms',
            showarrow=True, arrowhead=2,
            font=dict(size=12, color='red'),
            bgcolor='white'
        )

    # Update layout for enhanced visuals
    aggregated_fig.update_layout(
        title="Impedance Trends Across Batteries",
        xaxis_title="Time (Charge/Discharge Cycles)",
        yaxis_title="Resistance (Ohms)",
        xaxis=dict(
            showgrid=True,
            tickangle=45,
            ticks='inside',
            showline=True,
            linewidth=2,
            linecolor='gray'
        ),
        yaxis=dict(
            showgrid=True,
            type='log',  # Apply logarithmic scale
            showline=True,
            linewidth=2,
            linecolor='gray'
        ),
        legend_title="Parameters",
        template="plotly_dark",
        height=600,  # Adjust height for better readability
        width=1000,  # Adjust width for better fit
        margin=dict(l=40, r=40, t=80, b=80)  # Adjust margins for better layout
    )

    aggregated_fig.show()


#Heatmap Plot
def heatmap():
    # Calculate correlation matrix
    correlation_matrix = impedance_metadata[['Re', 'Rct']].corr()
    
    # Create heatmap from correlation matrix
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Correlation Heatmap: Re vs. Rct",
        color_continuous_scale="Viridis",  # Choose a color scale
        height=800
    )
    fig.show()


#3D- Plot
def Three_Dimensional_Plot():
    fig = go.Figure()

    for battery_id in impedance_metadata['battery_id'].unique():
        battery_data = impedance_metadata[impedance_metadata['battery_id'] == battery_id]
        fig.add_trace(go.Scatter3d(
            x=battery_data['start_time'],
            y=battery_data['Re'],
            z=battery_data['battery_id'],
            mode='lines',
            name=f'Battery {battery_id}'
        ))

    fig.update_layout(
        title="3D Visualization of Re Across Batteries Over Time",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Re (Electrolyte Resistance)",
            zaxis_title="Battery ID"
            ),
        template="plotly_dark",
        height=800
        )
    fig.show()