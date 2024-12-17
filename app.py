from flask import Flask, render_template, request
import logging
from nasa_code import *

# Initialize Flask app
app = Flask(__name__)

# Configure logging to a file for debugging purposes
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

@app.route('/')
def home():
    """Render the homepage with dropdowns for battery IDs and plot types."""
    try:
        return render_template('index.html', battery_ids=battery_ids)
    except Exception as e:
        logging.error(f"Error rendering home page: {e}")
        return "An error occurred while loading the home page. Please try again later.", 500

@app.route('/plot', methods=['POST'])
def plot():
    """Handle form submission to generate and display plots."""
    battery_id = request.form.get('battery_id')
    plot_type = request.form.get('plot_type')

    try:
        # Ensure valid inputs
        if not battery_id or battery_id not in battery_ids:
            raise ValueError(f"Invalid battery ID: {battery_id}")

        if not plot_type:
            raise ValueError("No plot type specified.")

        # Match plot type to function and generate the plot
        if plot_type == "line_impedance":
            plot_path = line_plot_impedance(battery_id)
        elif plot_type == "scatter_impedance":
            plot_path = scatter_plot_impedance(battery_id)
        elif plot_type == "bar_impedance":
            plot_path = bar_plot_impedance(battery_id)
        elif plot_type == "line_Re":
            plot_path = line_plot_Re(battery_id)
        elif plot_type == "scatter_Re":
            plot_path = scatter_plot_Re(battery_id)
        elif plot_type == "bar_Re":
            plot_path = bar_plot_Re(battery_id)
        elif plot_type == "line_Rct":
            plot_path = line_plot_Rct(battery_id)
        elif plot_type == "scatter_Rct":
            plot_path = scatter_plot_Rct(battery_id)
        elif plot_type == "bar_Rct":
            plot_path = bar_plot_Rct(battery_id)
        elif plot_type == "impedance_trends":
            plot_path = plot_impedance_trends(impedance_metadata)
        elif plot_type == "heatmap":
            plot_path = heatmap()
        elif plot_type == "3d_plot":
            plot_path = Three_Dimensional_Plot()
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")

        # Render the plot in the same page (using iframe or img)
        return render_template('index.html', battery_ids=battery_ids, plot_path=plot_path)

    except ValueError as ve:
        logging.warning(f"Value error: {ve}")
        return f"Input error: {ve}", 400
    except Exception as e:
        logging.error(f"Unexpected error during plot generation: {e}")
        return "An unexpected error occurred while generating the plot. Please try again later.", 500


if __name__ == '__main__':
    app.run(debug=True)
