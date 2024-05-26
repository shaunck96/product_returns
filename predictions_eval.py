import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from fpdf import FPDF
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load credentials and set paths
file_path = r'credentials/cred.json'
with open(file_path, 'r') as file:
    creds = json.load(file)

base_path = creds['predictions_eval_base_path']
if not os.path.exists(base_path):
    os.makedirs(base_path)

predictions_path = creds['predictions_csv_path']
predictions_df = pd.read_csv(predictions_path)

# Split data based on 'Returned' status
returned_customers = predictions_df[predictions_df['Returned'] == 1]
non_returned_customers = predictions_df[predictions_df['Returned'] == 0]

# Function to plot and save visualizations
def plot_and_save(df, column, kind, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    if kind == 'hist':
        sns.histplot(df[column].dropna(), kde=True, bins=30)
    elif kind == 'bar':
        sns.countplot(x=column, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"{title} {kind} saved as {save_path}.")

# Function to generate plots based on data type
def generate_plots(df, base_path, prefix):
    plot_paths = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            path = os.path.join(base_path, f"{column}_hist_{prefix}.png")
            plot_and_save(df, column, 'hist', f"{prefix}: {column} Distribution", column, 'Frequency', path)
            plot_paths.append(path)
        elif pd.api.types.is_categorical_dtype(df[column]):
            path = os.path.join(base_path, f"{column}_bar_{prefix}.png")
            plot_and_save(df, column, 'bar', f"{prefix}: {column}", column, 'Count', path)
            plot_paths.append(path)
    return plot_paths

# Generate plots
histogram_paths_returned = generate_plots(returned_customers, base_path, 'returned')
histogram_paths_non_returned = generate_plots(non_returned_customers, base_path, 'non_returned')

# Function to create PDF report
def create_pdf_report(base_path, summary_data, plot_paths, title):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align='C')

    # Summary Statistics
    pdf.set_font("Arial", size=10)
    for i, line in enumerate(summary_data.split('\n')):
        pdf.cell(200, 10, txt=line, ln=True)
        if i % 25 == 24:  # Adjust number based on your layout preference
            pdf.add_page()

    # Visualizations
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Visualizations:", ln=True)
    for path in plot_paths:
        pdf.image(path, x=10, w=190)
        pdf.cell(200, 10, txt="", ln=True)  # Empty line for spacing

    pdf.output(os.path.join(base_path, f"{title.lower().replace(' ', '_')}_report.pdf"))
    logging.info(f"{title} report PDF generated and saved.")

# Generate reports for both returned and non-returned customers
create_pdf_report(base_path, returned_customers.describe().to_string(), histogram_paths_returned, "Summary Report of Returned Customers")
create_pdf_report(base_path, non_returned_customers.describe().to_string(), histogram_paths_non_returned, "Summary Report of Non-Returned Customers")
