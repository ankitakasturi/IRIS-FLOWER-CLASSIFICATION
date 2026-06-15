# Iris Flower Classification GUI

This small project trains a Random Forest classifier on the Iris dataset and provides a Tkinter GUI to train the model, make predictions from user-entered measurements, save/load models, and visualize petal measurements.

Quick start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run the GUI:

```powershell
python "app.py"
```

Usage

- Enter sepal and petal measurements and click `Predict` to classify.
- Click `Train` to train a fresh model on the Iris dataset.
- `Save Model` and `Load Model` allow persisting models as `.joblib` files.
- `Show Plot` displays petal length vs petal width colored by species.

Notes

- The GUI uses Tkinter (bundled with Python). If you see errors, ensure the packages in `requirements.txt` are installed.
- Tested with Python 3.8+ on Windows.
