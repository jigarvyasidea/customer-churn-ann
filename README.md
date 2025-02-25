# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
Customer churn prediction is a crucial problem in the industry where businesses aim to identify customers who are likely to leave. This project uses an **Artificial Neural Network (ANN)** to predict customer churn based on historical data.

## 🏗 Tech Stack
- **Python**
- **TensorFlow/Keras** (for building the ANN model)
- **Pandas, NumPy** (for data preprocessing)
- **Matplotlib, Seaborn** (for data visualization)
- **Scikit-learn** (for data splitting and evaluation)

## 📂 Dataset
The dataset contains customer details such as:
- Customer ID
- Demographics (Age, Gender, Geography, etc.)
- Account information (Tenure, Balance, Number of Products, etc.)
- Behavioral data (Active status, Transactions, etc.)
- Target Variable: **Exited** (1 = Churned, 0 = Retained)

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jigarvyasidea/customer-churn-ann.git
   cd customer-churn-ann
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

## 🛠 Model Architecture
The ANN model consists of:
- **Input Layer**: Accepts preprocessed customer data.
- **Hidden Layers**: Multiple dense layers with **ReLU activation**.
- **Output Layer**: A single neuron with **Sigmoid activation** for binary classification.

## 📊 Performance Evaluation
The model is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Score

## 📝 Usage
- Train the model with historical data.
- Evaluate model performance.
- Use the trained model to predict customer churn probabilities.

## 📌 Future Enhancements
- Hyperparameter tuning for better performance.
- Feature engineering for improved predictions.
- Deployment as an API for real-time churn prediction.

## 🧑‍💻 Author
**Jigar Vyas**  
GitHub: [jigarvyasidea](https://github.com/jigarvyasidea)

## 📜 License
This project is licensed under the MIT License.

## 📚 Notebook
Here is the notebook: [Kaggle Notebook](https://www.kaggle.com/code/campusx/notebook8ad570467f)

