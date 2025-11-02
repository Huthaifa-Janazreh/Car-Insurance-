![Cover Image](car%20insurance.webp)

# 🚗 Insurance Claim Prediction – Feature Engineering & Selection

This project focuses on applying **feature engineering and feature selection techniques** to improve a machine learning model for predicting **insurance claim outcomes**.  
The experiments were conducted using **PCA (Principal Component Analysis)**, **KMeans clustering**, and **Embedded feature selection** via **Logistic Regression coefficients**.  
The final models were evaluated using **Random Forest Classifiers**.

---

## 📊 1. Principal Component Analysis (PCA)

### **Objective**
To reduce feature dimensionality while retaining the most informative variance components.

### **Process**
- Applied **PCA** on preprocessed numerical features.  
- Selected the top **3 principal components** based on explained variance ratio.  
- Combined the PCs with the original features to create the training and test sets.

### **Explained Variance**
| Principal Component | Explained Variance Ratio |
|----------------------|--------------------------|
| PC1 | 0.3499 |
| PC2 | 0.1195 |
| PC3 | 0.0971 |

Total variance captured ≈ **56.6%**

### **Visualization**
- **2D Scatter:** Shows clear spread along PC1 and PC2, indicating moderate separability between classes.  
- **3D Plot:** Highlights data clustering patterns across PC1–PC3, revealing underlying structure in the dataset.

### **Model Results (Random Forest with PCA Features)**
| Metric | Class 0 | Class 1 | Overall |
|---------|----------|----------|----------|
| Precision | 0.86 | 0.77 | - |
| Recall | 0.91 | 0.67 | - |
| F1-Score | 0.88 | 0.71 | - |
| Accuracy | - | - | **0.83** |

📈 PCA improved model generalization slightly while simplifying data representation.

---

## 🔵 2. KMeans Clustering (Feature Engineering)

### **Objective**
To capture hidden patterns in the feature space by creating **cluster-based features**.

### **Process**
- Evaluated **K = 2 → 8** using the **Silhouette Score**.
- Optimal **K = 4** with silhouette ≈ **0.39**.
- Added cluster labels and centroid distances as new engineered features.

Final shape with engineered features:  
**Training:** (7500, 23) **Testing:** (2500, 23)

### **Model Results (Random Forest + KMeans Features)**
| Metric | Class 0 | Class 1 | Overall |
|---------|----------|----------|----------|
| Precision | 0.854 | 0.744 | - |
| Recall | 0.898 | 0.659 | - |
| F1-Score | 0.875 | 0.699 | - |
| Accuracy | - | - | **0.824** |

### **Observations**
- Model performance improved slightly compared to baseline.  
- The clusters captured meaningful relationships, improving precision for non-claims.  
- Recall dropped slightly, indicating room for balancing class detection.

---

## 🧩 3. Embedded Feature Selection (Logistic Regression Coefficients)

### **Objective**
To automatically identify and retain the most predictive features using **Logistic Regression** weights.

### **Process**
1. Trained a `LogisticRegression` model on scaled engineered features.  
2. Used `SelectFromModel` with default mean-threshold to filter out weak predictors.  
3. Retrained a **Random Forest Classifier** using the selected subset.

### **Model Results (After Embedded Selection)**
| Metric | Class 0 | Class 1 | Overall |
|---------|----------|----------|----------|
| Precision | 0.856 | 0.760 | - |
| Recall | 0.906 | 0.662 | - |
| F1-Score | 0.880 | 0.708 | - |
| Accuracy | - | - | **0.830** |

### **Confusion Matrix Insights**
- 91% of non-claims predicted correctly.  
- 66% of actual claims identified.  
- Balanced precision-recall tradeoff across both classes.

---

## 🔝 Top Features via Permutation Importance
| Feature | Importance |
|----------|-------------|
| DRIVING_EXPERIENCE | 0.187 |
| VEHICLE_OWNERSHIP | 0.060 |
| GENDER_female | 0.017 |
| PAST_ACCIDENTS | 0.017 |
| VEHICLE_YEAR_before 2015 | 0.016 |
| VEHICLE_YEAR_after 2015 | 0.006 |
| RACE_minority | 0.001 |

### **Interpretation**
- **Driving experience** dominates model influence.  
- **Vehicle ownership** and **past accidents** follow as key predictors.  
---

## 🧩 Tools & Libraries
- Python (NumPy, pandas, scikit-learn, matplotlib, seaborn)  
- Models: `RandomForestClassifier`, `LogisticRegression`, `KMeans`, `SelectFromModel`, `PCA`  
- Evaluation: Confusion matrices, Precision, Recall, F1-score, Permutation Importance

---

## 📘  Notes
This project demonstrates a complete machine learning workflow — from **data preprocessing and feature engineering** to **feature selection and model evaluation** — showcasing how each step affects interpretability, efficiency, and performance.

