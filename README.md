# Machine-Learning-Engineer-Test

This project demonstrates a complete pipeline for housing price prediction, including data analysis, preprocessing, modeling, and online serving using FastAPI. Below is a detailed explanation of the project structure:

## Project Structure
Machine-Learning-Engineer-test/ ├── Analysis_modeling.ipynb ├── config.yaml ├── housing_price_modelling.py ├── housing.csv ├── GeoDataAnalysis.py ├── app/ │ ├── Dockerfile │ ├── main.py │ ├── preprocessing.py │ ├── requirements.txt │ ├── test.json │ ├── random_forest_model.pkl

### Files and Their Descriptions

#### **Analysis_modeling.ipynb**
This Jupyter Notebook contains:
- Data analysis and preprocessing steps.
- Model selection and evaluation.
- Detailed explanations and visualizations for each step.
For more information, refer to the inline documentation in the notebook.

#### **config.yaml**
This YAML file serves as the configuration hub, containing:
- Dataset path.
- Model parameters and hyperparameters.
- The model type used in training.
This file ensures the process is modular and easy to modify for retraining or adjustments.

#### **housing_price_modelling.py**
A Python script encapsulating the complete modeling pipeline:
- Reads configurations from `config.yaml`.
- Trains the model and supports retraining.
- Designed to integrate seamlessly with Vertex AI pipelines for scaling and automation (pending more time for setup).

#### **housing.csv**
The primary dataset used for training and evaluation.

#### **GeoDataAnalysis.py**
This script focuses on geographic data enrichment:
- Utilizes Google Maps API to fetch additional details (e.g., address, city, county) based on coordinates.
- Calculates distances from city centers for enhanced feature engineering.
**Note**: This script was not executed to avoid delays but is included for future model improvement.

#### **app/preprocessing.py**
A Python class for dataset preprocessing:
- **Outlier Detection**: Handles outliers using the IQR method.
- **Missing Value Imputation**: Fills missing values with mean or median.
- **Feature Creation**: Adds features such as `population_per_room` and `distance_to_center` using Haversine formula.
- **Categorical Encoding**: Encodes categorical features like `grid`.
- **Feature Scaling**: Standardizes numerical features (mean = 0, variance = 1).

#### **app/main.py**
A FastAPI-based script for online predictions:
- Hosts an API for predicting housing prices based on input variables.
- Example API body is provided in `app/test.json`.
- Deployed on Google Cloud Run. Test the API [here](https://fastapi-housing-predictor-981103843427.europe-west1.run.app/predict/).

#### **app/test.json**
An example JSON file illustrating the request body format for the prediction API.

#### **app/requirements.txt**
Lists all required Python libraries for the project.

#### **app/random_forest_model.pkl**
The trained Random Forest model file used for predictions.

#### **app/Dockerfile**
The Docker configuration file for containerizing the API:
- Hosted in Google Cloud Artifact Registry.
- Deployed on Google Cloud Run.

## Future Improvements
Given more time, the project could be extended with:
1. **Enhanced GeoData Analysis**:
   - Refine geographic features using Google Maps API.
   - Incorporate distance-based features for better predictions.
2. **Streamlit Dashboard**:
   - Build an interactive online dashboard for exploratory data analysis.
   - Integrate prediction capabilities using the API.
3. **Advanced Feature Engineering**:
   - Develop more sophisticated features to capture hidden patterns in the dataset.
4. **Pipeline Automation**:
   - Fully integrate the modeling pipeline with Google Cloud Vertex AI for streamlined training and deployment.

---

For more information, please don’t hesitate to contact the project owner.
