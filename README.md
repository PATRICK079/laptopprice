
# 💻 Full-Cycle Machine Learning System: Laptop Price Prediction
This project represents a complete end-to-end machine learning lifecycle for a regression task — from data ingestion and validation to model training, CI/CD automation, and cloud deployment. It goes beyond building a predictive model and demonstrates how to structure, scale, and operationalize machine learning in a real-world scenario.


# 🚀 Project Goal
To build a regression model that accurately estimates the price of a laptop based on structured features. This tool can be used by e-commerce platforms, retailers, or consumers for price analysis and recommendation.

# 🔄 Lifecycle Breakdown

**✅ Phase 1: Data Pipeline & ETL Setup**
• Designed a clean project structure with modular scripts and Git-based version control.

• Connected to MongoDB Atlas for storing raw and processed datasets.

• Built an ETL pipeline to extract, transform, and load laptop data into a cloud database.

**✅ Phase 2:** Data Ingestion & Preprocessing
• Created an ingestion pipeline to handle batch data input.

• Applied standard data cleaning techniques.

• Persisted training and testing splits for reuse across model versions.

**✅ Phase 3 & 4: Data Validation & Transformation** 

• Enforced schema consistency and performed data drift detection using the Kolmogorov-Smirnov (KS) test.

• Built a custom preprocessing pipeline (feature scaling, encoding) using scikit-learn.

• Stored transformed datasets as NumPy arrays for efficient downstream usage.

**✅ Phase 5: Model Training & Evaluation**
• Compared multiple regression algorithms: Linear Regression, Random Forest, XGBoost, CatBoost.

• Performed hyperparameter tuning, evaluated using RMSE and R² Score.

• Used MLflow and DagsHub for full experiment tracking.

• Saved the final model and transformer as .pkl files for future use.

**✅ Phase 6: Batch Prediction, CI/CD & API Deployment**
• Designed a structured, automated model training pipeline.

• Set up CI/CD pipelines with GitHub Actions for continuous integration.

• Stored model artifacts in AWS S3 and containerized APIs using Docker.

• Deployed a FastAPI app on AWS EC2, supporting CSV file uploads for batch predictions.

• Implemented robust logging and efficient response handling.

# 🗂️ Project Structure
                  
<img width="767" alt="Screenshot 2025-05-13 at 21 09 07" src="https://github.com/user-attachments/assets/3a4cbf1d-f757-4b2a-9066-e23440f7647b" />


# 🔍 How to Use
 1. Clone the repo:
   ```bash
    git clone https://github.com/PATRICK079/laptopprice.git

     cd laptopprice
 ```
2. Install dependencies:
```bash
    pip install -r requirements.txt
```
3. Run Streamlit App:
 ```bash
   streamlit run app.py
```
# 🎥 Demo Walkthroughs (All Project Phases)

Each phase of this project has been documented on LinkedIn. You can view each one of them.

| Phase           | Description                              | Link                                                                                                                                                                                                                           |
| --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase 1**     | ETL Pipeline & MongoDB Setup             | [Watch Demo](https://www.linkedin.com/posts/patrickedosoma_machinelearning-datascience-etlpipeline-activity-7287481095804129280-bnML?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEHatHsBYxJWXg3JP4WHJoKGr-0IWWzfM8A)   |
| **Phase 2**     | Data Ingestion & Storage                 | [Watch Demo](https://www.linkedin.com/posts/patrickedosoma_dataingestion-machinelearning-etlpipeline-activity-7289747694443495424-GiEf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEHatHsBYxJWXg3JP4WHJoKGr-0IWWzfM8A) |
| **Phase 3 & 4** | Data Validation & Transformation         | [Watch Demo](https://www.linkedin.com/posts/patrickedosoma_machinelearning-datascience-mlops-activity-7292263489035800576-aJoo?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEHatHsBYxJWXg3JP4WHJoKGr-0IWWzfM8A)         |
| **Phase 5**     | Model Training & Experiment Tracking     | [Watch Demo](https://www.linkedin.com/posts/patrickedosoma_machinelearning-datascience-modeltraining-activity-7294979530962669569-ISvv?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEHatHsBYxJWXg3JP4WHJoKGr-0IWWzfM8A) |
| **Phase 6**     | Batch Prediction, CI/CD & API Deployment | [Watch Demo](https://www.linkedin.com/posts/patrickedosoma_machinelearning-datascience-modeltraining-activity-7297271052668182528-rDZm?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEHatHsBYxJWXg3JP4WHJoKGr-0IWWzfM8A) |



# 🌐 Demo Links

**📦 Batch Predictions:** [Try It Here](https://lnkd.in/dp2VdMh5)


**⚡ Single Prediction Web App:** [Try It Here](https://lnkd.in/dusVUCtm)

   (Note: AWS instance may go offline temporarily to manage costs. Message me to reactivate.)









