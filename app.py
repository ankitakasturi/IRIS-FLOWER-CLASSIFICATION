import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


MODEL_FILE = "iris_model.joblib"


class IrisApp:
    def __init__(self, master):
        self.master = master
        master.title("Iris Flower Classifier")

        self.model = None
        self._load_data()

        frame = ttk.Frame(master, padding=12)
        frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        inputs = [
            ("Sepal length", "sepal_length"),
            ("Sepal width", "sepal_width"),
            ("Petal length", "petal_length"),
            ("Petal width", "petal_width"),
        ]

        self.entries = {}
        for i, (label, key) in enumerate(inputs):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=4)
            ent = ttk.Entry(frame, width=15)
            ent.grid(row=i, column=1, pady=4)
            self.entries[key] = ent

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=2, rowspan=4, padx=12)

        ttk.Button(btn_frame, text="Train", command=self.train).grid(row=0, column=0, pady=2, sticky=tk.EW)
        ttk.Button(btn_frame, text="Load Model", command=self.load_model_dialog).grid(row=1, column=0, pady=2, sticky=tk.EW)
        ttk.Button(btn_frame, text="Save Model", command=self.save_model_dialog).grid(row=2, column=0, pady=2, sticky=tk.EW)
        ttk.Button(btn_frame, text="Predict", command=self.predict).grid(row=3, column=0, pady=2, sticky=tk.EW)
        ttk.Button(btn_frame, text="Show Plot", command=self.show_plot).grid(row=4, column=0, pady=8, sticky=tk.EW)

        self.status_var = tk.StringVar(value="Model: not trained")
        ttk.Label(frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=3, pady=8, sticky=tk.W)

        self.result_var = tk.StringVar(value="Prediction: -")
        ttk.Label(frame, textvariable=self.result_var, font=(None, 12, "bold")).grid(row=6, column=0, columnspan=3, pady=4, sticky=tk.W)

        self.metrics_text = tk.Text(frame, width=60, height=10)
        self.metrics_text.grid(row=7, column=0, columnspan=3, pady=8)
        self.metrics_text.configure(state=tk.DISABLED)

        # Try to load an existing model file silently
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.status_var.set(f"Model: loaded from {MODEL_FILE}")
            except Exception:
                self.status_var.set("Model: failed to load saved model")

    def _load_data(self):
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # normalize column names to simple keys
        self.df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.df['target'] = iris.target
        self.target_names = iris.target_names

    def train(self):
        try:
            X = self.df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            y = self.df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, target_names=self.target_names)
            self.model = clf
            self.status_var.set(f"Model: trained (accuracy {acc:.3f})")
            self._set_metrics_text(f"Accuracy: {acc:.3f}\n\n{report}")
        except Exception as e:
            messagebox.showerror("Train Error", f"Training failed: {e}")

    def predict(self):
        vals = []
        for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            txt = self.entries[key].get().strip()
            try:
                val = float(txt)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {key.replace('_', ' ')}: '{txt}'")
                return
            vals.append(val)

        X = np.array(vals).reshape(1, -1)
        if self.model is None:
            answer = messagebox.askyesno("No model", "Model is not trained. Train now?")
            if answer:
                self.train()
            else:
                return

        try:
            pred = self.model.predict(X)[0]
            probs = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
            label = self.target_names[pred]
            if probs is not None:
                prob_text = ", ".join([f"{n}:{p:.2f}" for n, p in zip(self.target_names, probs)])
                self.result_var.set(f"Prediction: {label} ({prob_text})")
            else:
                self.result_var.set(f"Prediction: {label}")
        except Exception as e:
            messagebox.showerror("Predict Error", f"Prediction failed: {e}")

    def save_model_dialog(self):
        if self.model is None:
            messagebox.showinfo("No Model", "Train a model before saving")
            return
        path = filedialog.asksaveasfilename(defaultextension='.joblib', filetypes=[('Joblib', '*.joblib'), ('All files', '*.*')], initialfile=MODEL_FILE)
        if not path:
            return
        try:
            joblib.dump(self.model, path)
            self.status_var.set(f"Model: saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model: {e}")

    def load_model_dialog(self):
        path = filedialog.askopenfilename(filetypes=[('Joblib', '*.joblib'), ('All files', '*.*')])
        if not path:
            return
        try:
            self.model = joblib.load(path)
            self.status_var.set(f"Model: loaded from {os.path.basename(path)}")
            messagebox.showinfo("Loaded", "Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model: {e}")

    def show_plot(self):
        try:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            for t in sorted(self.df['target'].unique()):
                sel = self.df[self.df['target'] == t]
                ax.scatter(sel['petal_length'], sel['petal_width'], label=self.target_names[t], c=colors[t])
            ax.set_xlabel('Petal length')
            ax.set_ylabel('Petal width')
            ax.set_title('Iris: petal length vs petal width')
            ax.legend()

            win = tk.Toplevel(self.master)
            win.title('Iris plot')
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plot: {e}")

    def _set_metrics_text(self, text):
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = IrisApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
