import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_name):
    try:
        logging.info(f"Starting {script_name}")
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        logging.info(f"Finished {script_name}")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running {script_name}")
        logging.error(e.stderr)

if __name__ == '__main__':
    scripts = ['training.py', 'inference.py', 'predictions_eval.py']
    
    for script in scripts:
        run_script(script)
