# Traffic Accident Severity Prediction

A machine learning project that predicts the severity of traffic accidents using multiple classification algorithms including Logistic Regression and Support Vector Machines (SVM).

## 📋 Overview

This project implements a complete machine learning pipeline to predict traffic accident severity based on environmental factors, road conditions, and vehicle characteristics. The models classify accidents into three severity levels (0, 1, 2) using features such as speed, weather conditions, road surface type, and vehicle movement patterns.

## 🎯 Project Pipeline

**Data Loading → Feature Engineering → Preprocessing → Feature Selection → Modeling → Evaluation**

## 📊 Dataset

- **Source**: `cleaned.csv` (12,316 records, 16 features)
- **Target Variable**: `Accident_severity` (0, 1, 2)
- **Key Features**:
  - Speed (synthetic feature)
  - Weather conditions
  - Road surface type
  - Vehicle movement patterns
  - And 12 additional contextual features

## 🔧 Feature Engineering

- **Synthetic Speed Column**: Generated based on vehicle movement patterns and road surface conditions
  - Base speeds vary by movement type (e.g., going straight: 40-120 km/h, reversing: 5-20 km/h)
  - Road surface conditions apply speed multipliers (hazardous surfaces reduce speed by ~20%)
  - Gaussian noise added for realism

## 🧹 Data Preprocessing

1. **Missing Value Handling**: Categorical missing values filled with mode
2. **Data Cleaning**: Removed "Unknown" and "Other" entries to reduce noise
3. **Feature Selection**: Focus on 4 most impactful features:
   - `speed` - Vehicle speed during accident
   - `weather` - Weather conditions
   - `road_condition` - Road surface type
   - `vehicle_type` - Vehicle movement pattern
4. **Encoding**: One-hot encoding for categorical variables
5. **Scaling**: StandardScaler applied for SVM models

## 🤖 Models Implemented

| Model | Kernel/Type | Description |
|-------|-------------|-------------|
| **Logistic Regression** | - | Linear probabilistic classifier |
| **SVM Linear** | Linear | Finds linear hyperplane for class separation |
| **SVM Polynomial** | Polynomial (degree=3) | Captures curved decision boundaries |
| **SVM RBF** | Radial Basis Function | Models complex non-linear patterns |

## 📈 Evaluation Metrics

- **Primary Metric**: Weighted Recall Score
- **Visualization**: Confusion matrices for all models
- **Model Comparison**: Side-by-side performance analysis

## 🚀 Getting Started

### Prerequisites

```bash
# Required libraries
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

### Usage

1. Ensure `cleaned.csv` is in the project directory
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially to execute the complete pipeline
4. View model performance comparisons in the output

## 📁 Project Structure

```
Traffic Accident Severity Prediction/
│
├── main.ipynb          # Main Jupyter notebook with complete pipeline
├── cleaned.csv         # Dataset file
├── README.md          # Project documentation
└── venv/              # Virtual environment (if created)
```

## 🎯 Key Results

The project compares four different models using weighted recall scores:
- Models are evaluated on their ability to correctly classify accident severity
- Confusion matrices provide detailed view of classification performance
- Best performing model is automatically identified and reported

## 🔍 Features Deep Dive

### Speed Feature Engineering
- **Logic**: Speed directly correlates with accident severity
- **Implementation**: Rules-based generation considering vehicle movement and road conditions
- **Validation**: Realistic speed distributions based on real-world scenarios

### Feature Selection Rationale
- **Speed**: Primary kinetic factor in accident severity
- **Weather**: Environmental hazard affecting visibility and road conditions
- **Road Condition**: Surface quality impacts vehicle control and stopping distance
- **Vehicle Movement**: Behavioral patterns indicating risk levels

## 📝 Technical Notes

- **Reproducibility**: Random seeds set for consistent results
- **Data Quality**: Noise reduction through removal of ambiguous categories
- **Model Selection**: Multiple algorithms to identify best approach
- **Scaling**: Essential for SVM performance due to distance-based calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning for all models
- [ ] Cross-validation for more robust evaluation
- [ ] Feature importance analysis
- [ ] Additional ensemble methods
- [ ] Real-time prediction API
- [ ] Interactive dashboard for results visualization

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project uses synthetic speed data generated based on realistic assumptions. In production scenarios, actual speed data from accident reports would be preferred.