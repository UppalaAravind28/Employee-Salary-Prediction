
---

# Employee Salary Prediction App

## Overview

The **Employee Salary Prediction App** is a machine learning-based Streamlit application that predicts the **Payment Tier** of employees based on their demographic and professional details. The app uses a trained Random Forest model to make predictions and provides insights into the dataset and model performance.

Key Features:
- **Interactive User Interface**: Navigate through different pages (Home, Dataset Overview, Model Insights, Make a Prediction) using the navigation bar or sidebar.
- **Data Visualization**: Explore the dataset, target variable distribution, feature importance, and confusion matrix.
- **Prediction Capability**: Enter employee details to predict their Payment Tier in real-time.
- **Professional Styling**: Clean and modern UI with custom CSS for a polished look.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Model](#model)
6. [Deployment](#deployment)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Steps to Set Up Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/UppalaAravind28/Employee-Salary-Prediction]
   cd employee-salary-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present in the root directory:
   - `Employee.csv`: Dataset used for training and testing.
   - `salary_prediction_model.pkl`: Pre-trained machine learning model.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Navigation
The app has four main sections:
1. **Home**: Provides an introduction to the app and its purpose.
2. **Dataset Overview**: Displays the dataset, key metrics, and visualizations (e.g., target variable distribution).
3. **Model Insights**: Shows feature importance and the confusion matrix for the trained model.
4. **Make a Prediction**: Allows users to input employee details and get a predicted Payment Tier.

### Making a Prediction
1. Navigate to the "Make a Prediction" page.
2. Enter the employee's details:
   - Gender (Male/Female)
   - Age
   - Joining Year
   - Education Level (Bachelors/Masters/PHD)
   - City (Bangalore/Pune/New Delhi)
   - Ever Benched (Yes/No)
   - Experience in Current Domain (in years)
3. Click the "Predict" button to see the predicted Payment Tier.

---

## Project Structure

```
employee-salary-prediction/
├── app.py                     # Main Streamlit application script
├── Employee.csv               # Dataset used for training and testing
├── salary_prediction_model.pkl # Pre-trained machine learning model
├── requirements.txt           # List of Python dependencies
└── README.md                  # Project documentation
```

---

## Dataset

The dataset (`Employee.csv`) contains information about employees, including:
- **Gender**: Male or Female
- **Age**: Employee's age
- **JoiningYear**: Year the employee joined the company
- **Education**: Bachelors, Masters, or PHD
- **City**: Bangalore, Pune, or New Delhi
- **EverBenched**: Whether the employee has ever been benched (Yes/No)
- **ExperienceInCurrentDomain**: Years of experience in the current domain
- **PaymentTier**: Target variable indicating the payment tier (1, 2, or 3)

---

## Model

The app uses a **Random Forest Classifier** trained on the provided dataset. The model predicts the `PaymentTier` based on the input features. Key preprocessing steps include:
- Handling missing values
- Label encoding for binary variables (`Gender`, `EverBenched`)
- One-hot encoding for categorical variables (`Education`, `City`)
- Scaling numerical features (`Age`, `JoiningYear`, `ExperienceInCurrentDomain`)

The trained model is saved as `salary_prediction_model.pkl`.

---

## Deployment

### Deploying to Streamlit Cloud
1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Create a new app and connect it to your repository.
4. Ensure the following files are included in the repository:
   - `app.py`
   - `Employee.csv`
   - `salary_prediction_model.pkl`
   - `requirements.txt`
5. Deploy the app and share the link.
6. Deployment link for these app is : https://uppalaaravind28-employee-salary-prediction-app-mzzq0a.streamlit.app/

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature or fix"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request to the `main` branch of this repository.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, feel free to reach out:
- **Developer**: Uppala Aravind
- **Email**: uppalaaravind28@example.com
- **GitHub**: [UppalaAravind28](https://github.com/UppalaAravind28)

---
