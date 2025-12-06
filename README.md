# YOLO-based Object Detection Project

## Project Structure


## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your dataset in YOLO format (see `data.yaml`).
3. Run a quick YOLO test:
   ```bash
   python quick_test.py
   ```
4. Train your model:
   ```bash
   python train_yolo_by_GPU.py
   ```
5. Try the real-time webcam demo:
   ```bash
   python real_time_demo.py
   ```

## Dataset Conversion

## Customization


## Using `notebook.ipynb`

The `notebook.ipynb` file provides an interactive environment for:

- **Data Exploration:** Load and visualize your dataset, inspect class distributions, and check sample images.
- **Model Training:** Run YOLO training cells, experiment with hyperparameters, and monitor training progress.
- **Evaluation & Visualization:** Plot metrics, display confusion matrices, and visualize detection results on images.
- **Rapid Prototyping:** Test new ideas, preprocessing steps, or model tweaks before integrating into scripts.

### How to Use

1. Open `notebook.ipynb` in JupyterLab, VS Code, or your preferred notebook editor.
2. Run cells sequentially to follow the workflow, or jump to specific sections as needed.
3. Modify code cells to experiment with different parameters, models, or visualizations.
4. Use output cells to interpret results and guide further development.


