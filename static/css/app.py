from flask import Flask, request, render_template, jsonify
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST']) 
def upload():
    try:
        file = request.files['file']
        data, stats = parse_csv(file)
        return jsonify({
            'data': data,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def parse_csv(file):
    try:
        data = []
        reader = csv.reader(file.read().decode().splitlines())
        headers = next(reader)
        if any(cell.isdigit() for cell in headers):  # Check if headers are numeric
            # If headers are numeric, it's likely there are no headers in the CSV
            headers = [f'Column {i+1}' for i in range(len(headers))]
            data.append(dict(zip(headers, headers)))  # Add headers as first row of data
        for row in reader:
            row_data = {}
            for i, cell in enumerate(row):
                row_data[headers[i]] = cell 
            data.append(row_data)
        stats = calculate_stats(data)
        return data, stats
    except Exception as e:
        raise Exception('Error parsing CSV: ' + str(e))

def calculate_stats(data):
    try:
        stats = {}
        for col_name in data[0].keys():
            column = [row[col_name] for row in data if row[col_name].isdigit()]
            if column:
                min_val = min(column)
                max_val = max(column)
                mean_val = sum(map(int, column)) / len(column)
                sorted_column = sorted(column)
                mid = len(sorted_column) // 2
                median_val = (int(sorted_column[mid - 1]) + int(sorted_column[mid])) / 2
                stats[col_name] = {
                    "min": min_val,
                    "max": max_val, 
                    "mean": mean_val,
                    "median": median_val
                }
        return stats
    except Exception as e:
        raise Exception('Error calculating stats: ' + str(e))

if __name__ == '__main__':
   app.run()