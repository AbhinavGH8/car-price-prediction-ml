# 🚗 Regression Model Comparison on Car Dataset

**Author**: Abhinav S  
**Education**: B.Tech AI/ML (Semester 3)  
**Project Type**: Beginner-Level ML Practice  

---

## 📌 Project Objective

This project is designed to compare the performance of three common regression models on a used car dataset. The primary focus is not on building a highly accurate car price predictor, but on:

- Learning how to structure ML code using `scikit-learn` pipelines  
- Preprocessing numeric and categorical data with `ColumnTransformer`  
- Evaluating models using standard regression metrics  
- Visualizing results using `matplotlib` and `seaborn`

---

## 🛠️ Tech Stack

- Python  
- pandas, NumPy  
- scikit-learn (modeling and preprocessing)  
- seaborn + matplotlib (visualization)

---

## 🧪 Models Used

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

Each model is wrapped in a `Pipeline` that:
- Scales numerical features using `StandardScaler`
- Encodes categorical features using `OneHotEncoder`
- Trains and evaluates using `train_test_split`

---

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**: Lower is better  
- **R² Score**: Closer to 1 is better  

Model performance is visualized side-by-side for both metrics.

---

## 📁 Dataset

- `car_data_full.csv` 
- Features include: `brand`, `fuel_type`, `transmission`, `year`, `mileage`, `engine_power`
- Target: `price` (used for regression comparison only, not production prediction)

---

## ⚠️ Disclaimer

This is a **learning project** and not intended for real-world car price prediction. Model predictions are not production-accurate — the purpose is to compare modeling workflows, not build a commercial-grade solution.

---

## ✅ How to Run

1. Clone the repo  
2. Place your `car_data_full.csv` in the root directory  
3. Run the Python script:
```bash
python car_regression_comparison.py

 
