import yfinance as yf
import json
import time
from threading import Thread, Lock
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from oil import get_ultimate_wti_prediction # Assuming oil.py is in the same directory

# --- Configuration ---
DATA_FILE = 'data.json'
MAX_DATA_POINTS = 30
WTI_TICKER = 'CL=F'
FETCH_INTERVAL_SECONDS = 2      # Interval for fetching real-time actual data
PREDICTION_INTERVAL_SECONDS = 3600 # Interval for fetching a new prediction (1 hour)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Data and Lock Initialization ---
data = {"actual": [], "predicted": [], "timeRemaining": 3600}
data_lock = Lock()

# --- Utility Functions ---
def read_data():
    """Reads data from the JSON file on startup and ensures lists are synchronized."""
    global data
    try:
        with open(DATA_FILE, 'r') as f:
            loaded_data = json.load(f)
            # Ensure both lists are of the same length to handle inconsistencies
            len_actual = len(loaded_data.get('actual', []))
            len_predicted = len(loaded_data.get('predicted', []))
            min_len = min(len_actual, len_predicted)
            
            data['actual'] = loaded_data['actual'][:min_len]
            data['predicted'] = loaded_data['predicted'][:min_len]
            data['timeRemaining'] = loaded_data.get('timeRemaining', 3600)

    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is invalid, start with empty lists
        print(f"{DATA_FILE} not found or is invalid. Starting fresh.")
        data = {"actual": [], "predicted": [], "timeRemaining": 3600}

def write_data():
    """Writes the current data state to the JSON file."""
    with data_lock:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)

def calculate_time_remaining():
    """Calculate time remaining until next hour."""
    current_time = time.time()
    # Get seconds since the start of current hour
    seconds_in_hour = current_time % 3600
    # Calculate remaining seconds until next hour
    return int(3600 - seconds_in_hour)

# --- Background Threads ---
def timer_thread():
    """Manages the countdown timer."""
    global data
    while True:
        with data_lock:
            data['timeRemaining'] = calculate_time_remaining()
        write_data()
        time.sleep(1)

def fetch_actual_data():
    """Fetches real-time WTI data and UPDATES the last value in the 'actual' array."""
    global data
    while True:
        try:
            ticker = yf.Ticker(WTI_TICKER)
            wti_price = ticker.info.get('regularMarketPrice')
            if wti_price:
                with data_lock:
                    # Only update if the 'actual' list is not empty
                    if data['actual']:
                        data['actual'][-1] = wti_price
                # This write operation makes the latest price available to the frontend immediately
                write_data()
                print(f"Updated actual WTI price: {wti_price}")
        except Exception as e:
            print(f"Error fetching actual data: {e}")
        time.sleep(FETCH_INTERVAL_SECONDS)

def fetch_predicted_data():
    """Fetches a new prediction, appends it, and adds a corresponding new slot to the 'actual' array."""
    global data
    while True:
        try:
            print("Fetching new hourly prediction...")
            predicted_price = get_ultimate_wti_prediction()
            
            # Fetch the current price once to initialize the new slot in the 'actual' array.
            initial_actual_price = None
            try:
                ticker = yf.Ticker(WTI_TICKER)
                initial_actual_price = ticker.info.get('regularMarketPrice')
            except Exception as e:
                 print(f"Could not fetch initial actual price for new slot: {e}")

            # If fetching the initial actual price fails, use the prediction as a placeholder.
            if initial_actual_price is None:
                initial_actual_price = predicted_price

            with data_lock:
                # Append new values to both lists to keep their indices and lengths synchronized.
                data['predicted'].append(predicted_price)
                data['actual'].append(initial_actual_price)

                # Ensure the lists do not exceed the max size by removing the oldest entry.
                if len(data['predicted']) > MAX_DATA_POINTS:
                    data['predicted'].pop(0)
                    data['actual'].pop(0)
                
                # Reset timer when new prediction is added
                data['timeRemaining'] = 3600
            
            # Write data after the hourly update to save the new prediction.
            write_data()
            print(f"Fetched new prediction: {predicted_price}. Added new slot for actuals. Data written.")

        except Exception as e:
            print(f"Error fetching predicted data: {e}")
        
        # Wait for the next prediction cycle.
        time.sleep(PREDICTION_INTERVAL_SECONDS)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """API endpoint to get the current data."""
    with data_lock:
        return jsonify(data)

# --- Main Execution ---
if __name__ == '__main__':
    # Read initial data from the file at startup
    read_data()

    # Start the background threads for fetching data
    timer_thread_obj = Thread(target=timer_thread, daemon=True)
    actual_thread = Thread(target=fetch_actual_data, daemon=True)
    predicted_thread = Thread(target=fetch_predicted_data, daemon=True)
    
    timer_thread_obj.start()
    actual_thread.start()
    predicted_thread.start()

    # Run the Flask web server
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5500)