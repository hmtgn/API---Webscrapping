from fastapi import APIRouter, HTTPException
import os
import json
import kaggle
import pandas as pd
import glob  # Utilisé pour rechercher des fichiers spécifiques
import shutil  # Utilisé pour renommer des fichiers
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Form
from fastapi.responses import StreamingResponse
from time import sleep
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving the trained model
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer
import firebase_admin.auth as firebase_auth
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

auth_scheme = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()
templates = Jinja2Templates(directory="src/template")
DATA_FOLDER = "src/data"
JSON_FOLDER = "src/config"
MODELS_FOLDER = "src/models"
DATASETS_JSON = os.path.join(JSON_FOLDER, "dataset.json")
MODEL_PARAMETERS_FILE = os.path.join(JSON_FOLDER, "model_parameters.json")

cred = credentials.Certificate("src/config/datasources-api-montagnon-firebase-adminsdk-gpdo6-9a8eca22db.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
    
def rate_limit_key_func(request: Request):
    """
    Determine the rate-limiting key based on Firebase UID or IP address.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        str: The Firebase UID if authenticated, otherwise the remote IP address.
    """
    # Vérifie si un token Firebase est présent
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            decoded_token = firebase_auth.verify_id_token(token)
            return decoded_token["uid"]  # Utiliser l'UID Firebase comme clé
        except Exception:
            pass
    # Retour par défaut à l'adresse IP si aucun token valide
    return get_remote_address(request)

limiter = Limiter(key_func=rate_limit_key_func)

def verify_firebase_token(token: str = Security(auth_scheme)):
    """
    Verifies a Firebase authentication token and retrieves user details.

    Args:
        token (str): The Firebase authentication token provided by the client.

    Returns:
        dict: Decoded token information if valid.

    Raises:
        HTTPException: If the token is invalid or authentication fails.
    """
    try:
        # Verify the token using Firebase Admin SDK
        decoded_token = firebase_auth.verify_id_token(token.credentials)
        return decoded_token
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


# Fonction pour lire le fichier JSON des datasets
def load_datasets():
    """
    Reads the dataset configuration file (JSON) containing dataset metadata.

    Returns:
        dict: Parsed JSON content of datasets if the file exists.
    """
    if os.path.exists(JSON_FOLDER):
        with open(DATASETS_JSON, "r") as file:
            return json.load(file)
    return {}

# Fonction pour sauvegarder le fichier JSON des datasets
def save_datasets(datasets):
    """
    Saves the dataset metadata to a JSON file.

    Args:
        datasets (dict): Dataset metadata to save.
    """
    with open(DATASETS_JSON, "w") as file:
        json.dump(datasets, file, indent=4)

def load_csv_from_folder(folder_path: str):
    """
    Load all CSV files from a given folder into a dictionary of DataFrames.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        dict: Dictionary where keys are file names and values are DataFrames.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    csv_files = {}
    
    # Vérifier que le dossier existe
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Le dossier '{folder_path}' n'existe pas.")
    
    # Lister tous les fichiers dans le dossier
    for file_name in os.listdir(folder_path):
        # Vérifier si c'est un fichier CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Charger le CSV en DataFrame
                df = pd.read_csv(file_path)
                csv_files[file_name] = df
            except Exception as e:
                print(f"Erreur lors du chargement du fichier {file_name}: {e}")
    
    return csv_files

def load_model_parameters():
    """
    Load machine learning model parameters from the configuration file.

    Returns:
        dict: Dictionary of model parameters if the file exists.
    """
    if os.path.exists(MODEL_PARAMETERS_FILE):
        with open(MODEL_PARAMETERS_FILE, "r") as file:
            return json.load(file)
    return {}


@router.get("/train/ui", response_class=HTMLResponse, tags=["model"])
def get_prediction_ui(request: Request):
    """
    Serve an HTML form to initiate the training process for a machine learning model.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        HTMLResponse: An HTML page containing the training UI.
    """
    return templates.TemplateResponse("full_process.html", {"request": request})

@router.get("/predict/ui", response_class=HTMLResponse, tags=["model"])
def get_prediction_ui(request: Request):
    """
    Serve an HTML form to input features for making predictions using a trained model.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        HTMLResponse: An HTML page containing the prediction input form.
    """
    return templates.TemplateResponse("predict_form.html", {"request": request})

@router.get("/download-dataset/{dataset_name}", tags=["data"])
def download_dataset(dataset_name: str):
    """
        Downloads a dataset from Kaggle and renames the CSV file.

    Description:
        This endpoint allows you to download a dataset from Kaggle based on its name,
        as defined in the `datasets.json` file. The dataset is downloaded, unzipped,
        and the main CSV file is renamed to match the given `dataset_name`.

    Args:
        - dataset_name (str): The name of the dataset to be downloaded. This must match
          one of the keys in the `datasets.json` file.

    Returns:
        - Success (200): JSON object with the following keys:
            - message (str): Confirmation message of the successful download and rename.
            - path (str): File path to the renamed CSV file.
        - Error (404): Dataset not found in `datasets.json`.
        - Error (422): `datasets.json` file is invalid (e.g., malformed JSON).
        - Error (503): Service unavailable due to missing `datasets.json` or
          Kaggle dependency issues.
        - Error (500): Internal server error if processing or file handling fails.
    """
    # Ensure the data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Load datasets.json
    try:
        with open(DATASETS_JSON, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        # Erreur 500 si le fichier JSON n'est pas trouvé
        raise HTTPException(status_code=503, detail="dataset.json file not found")
    except json.JSONDecodeError:
        # Erreur 500 si le fichier JSON est mal formé
        raise HTTPException(status_code=422, detail="datasets.json is not a valid JSON file")

    # Check if the dataset_name exists
    if dataset_name not in datasets:
        # Erreur 404 si le dataset_name n'existe pas dans le JSON
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found in datasets.json"
        )

    # Get the dataset URL
    dataset_url = datasets[dataset_name]["url"]

    # Download the dataset
    try:
        kaggle.api.dataset_download_files(
            dataset_url, path=DATA_FOLDER, unzip=True
        )
    except Exception as e:
        # Erreur 500 si le téléchargement échoue
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")

    # Search for a CSV file in the DATA_FOLDER
    try:
        csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded dataset.")

        # Rename the first CSV file found to match `dataset_name.csv`
        original_csv_path = csv_files[0]
        renamed_csv_path = os.path.join(DATA_FOLDER, f"{dataset_name}.csv")
        shutil.move(original_csv_path, renamed_csv_path)

        return {
            "message": f"Dataset '{dataset_name}' downloaded and renamed successfully.",
            "path": renamed_csv_path
        }
    except FileNotFoundError:
        # Erreur 500 si aucun fichier CSV n'est trouvé
        raise HTTPException(status_code=503, detail="No CSV file found in the dataset.")
    except Exception as e:
        # Erreur 500 pour toute autre erreur
        raise HTTPException(status_code=503, detail=f"Error processing dataset: {str(e)}")

@router.get("/manage-datasets", response_class=HTMLResponse, tags=["data"])
def show_manage_datasets_form():
    """
    Display an HTML form to manage Kaggle datasets.

    Returns:
        HTMLResponse: An HTML page containing the form to add or remove Kaggle datasets.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Manage Datasets</title>
    </head>
    <body>
        <h1>Download Kaggle Dataset</h1>
        <form action="/api/manage-datasets" method="post">
            <label for="dataset_name">Dataset Name:</label><br>
            <input type="text" id="dataset_name" name="dataset_name" required><br><br>
            <label for="dataset_url">Dataset URL:</label><br>
            <input type="text" id="dataset_url" name="dataset_url" required><br><br>
            <button type="submit">Add/Remove Dataset</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/manage-datasets", response_class=HTMLResponse, tags=["data"])
def manage_datasets(dataset_name: str = Form(...), dataset_url: str = Form(...)):
    """
    Manage Kaggle datasets in the dataset configuration JSON file.

    Args:
        dataset_name (str): The name of the dataset to add or remove.
        dataset_url (str): The Kaggle URL of the dataset.

    Returns:
        HTMLResponse: A confirmation message indicating whether the dataset was added or removed.
    """
    datasets = load_datasets()

    dataset_key = dataset_name.lower().replace(" ", "_")
    if dataset_key in datasets:
        # Remove the dataset if it already exists
        del datasets[dataset_key]
        message = f"The dataset '{dataset_name}' has been removed."
    else:
        # Add the dataset if it does not exist
        datasets[dataset_key] = {
            "name": dataset_name,
            "url": dataset_url,
            "csv": dataset_key + ".csv"
        }
        message = f"The dataset '{dataset_name}' has been added successfully."

    save_datasets(datasets)

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h1>{message}</h1>
            <p>{message}</p>
            <a href="/api/manage-datasets">Return to the form</a>
        </body>
        </html>
    """, status_code=200)


@router.get("/datasets", response_class=HTMLResponse, tags=["data"])
def show_datasets():
    """
    Display a table of available datasets.

    Returns:
        HTMLResponse: An HTML page showing a table of datasets and their details.
    """
    datasets = load_datasets()

    if not datasets:
        return HTMLResponse(content="<h1>No datasets available</h1>", status_code=404)

    table_content = "<table border='1' style='width:100%'><tr><th>Name</th><th>URL</th>"
    for dataset_key, dataset_info in datasets.items():
        table_content += f"""
        <tr>
            <td>{dataset_info['name']}</td>
            <td><a href='https://www.kaggle.com/datasets/{dataset_info['url']}' target='_blank'>{dataset_info['url']}</a></td>
        </tr>
        """
    table_content += "</table>"

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Available Datasets</title>
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                button {{
                    padding: 8px 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <h1>Available Datasets</h1>
            {table_content}
            <br><br>
            <a href="/docs">Return to management</a>
        </body>
        </html>
    """, status_code=200)


@router.get("/datasets/iris_species", response_class=HTMLResponse, tags=["data"])
def show_iris_dataset():
    """
    Display the content of the `iris_species.csv` dataset in a table format.

    Returns:
        HTMLResponse: An HTML page containing a table with the dataset's data.

    Raises:
        HTTPException: If the dataset file is not found or cannot be loaded.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail=f"The file '{iris_csv_path}' was not found.")

    try:
        df = pd.read_csv(iris_csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

    table_html = df.to_html(index=False, classes='data-table', border=1)

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>View Data - Iris Species</title>
            <style>
                .data-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 20px;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #f2f2f2;
                }}
                .data-table tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <h1>Dataset Data: Iris Species</h1>
            <p>Displaying the first rows of the iris_species.csv dataset.</p>
            {table_html}
            <br><br>
            <a href='/api/datasets'>Return to the dataset list</a>
        </body>
        </html>
    """, status_code=200)

@router.get("/datasets/iris_species/json", tags=["data"])
def get_iris_dataset_as_json():
    """
    Provide the `iris_species.csv` dataset as a JSON response.

    Returns:
        JSONResponse: The dataset in JSON format.

    Raises:
        HTTPException: If the dataset file is not found or cannot be loaded.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="The file 'iris_species.csv' was not found.")
    try:
        df = pd.read_csv(iris_csv_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@router.get("/datasets/iris_species/preprocessed", tags=["data"])
def preprocess_iris_dataset():
    """
    Preprocess the `iris_species.csv` dataset:
    - Handle missing values.
    - Encode categorical columns.

    Returns:
        JSONResponse: Preprocessed dataset in JSON format.

    Raises:
        HTTPException: If the dataset file is not found or processing fails.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="The file 'iris_species.csv' was not found.")

    try:
        df = pd.read_csv(iris_csv_path)

        # Preprocessing steps
        df.dropna(inplace=True)  # Remove missing values
        if "species" in df.columns:
            df["species"] = df["species"].astype("category").cat.codes  # Encode categorical data

        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@router.api_route("/datasets/split", methods=["GET", "POST"], tags=["data"])
def train_test_split_endpoint(test_size: float = 0.2, random_state: int = 42):
    """
    Perform a train/test split on the `iris_species.csv` dataset.

    Args:
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Yields:
        str: A streaming response with data about the train/test split.

    Raises:
        HTTPException: If the dataset file is not found or an error occurs during processing.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"

    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail=f"The file '{iris_csv_path}.csv' was not found.")

    try:
        sleep(1)
        df = pd.read_csv(iris_csv_path)
        sleep(1)
        df.dropna(inplace=True)
        if "Species" in df.columns:
            df["Species"] = df["Species"].astype("category").cat.codes

        sleep(1)
        X = df.drop("Species", axis=1)
        y = df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_data = pd.concat([X_train, y_train], axis=1).to_json(orient="records")
        test_data = pd.concat([X_test, y_test], axis=1).to_json(orient="records")

        yield f"data: Train Split: {train_data}\n\n"
        yield f"data: Test Split: {test_data}\n\n"
        yield "data: Process completed successfully!\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"


@router.post("/train-model", tags=["model"])
def train_classification_model():
    """
    Train a RandomForestClassifier using the `iris_species.csv` dataset.
    The trained model is saved to the `src/models` folder.

    Returns:
        dict: Details about the trained model, including its save path and parameters.

    Raises:
        HTTPException: If the dataset is missing or if an error occurs during training.
    """
    iris_csv_path = os.path.join(DATA_FOLDER, "iris_species.csv")
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="Processed dataset not found.")

    try:
        df = pd.read_csv(iris_csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the dataset: {str(e)}")

    if "Species" not in df.columns:
        raise HTTPException(status_code=500, detail="The dataset must include the 'Species' column.")

    try:
        X = df.drop("Species", axis=1)
        y = df["Species"]

        model_parameters = load_model_parameters().get("RandomForestClassifier", {})
        model = RandomForestClassifier(**model_parameters)
        model.fit(X, y)

        model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")
        joblib.dump(model, model_path)

        return {
            "message": "Model trained and saved successfully.",
            "model_path": model_path,
            "parameters": model_parameters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


@router.get("/datasets/iris_species/full_process", tags=["data"])
async def full_process_iris_dataset_stream(test_size: float = 0.2, random_state: int = 42):
    """
    Execute the complete pipeline for the `iris_species.csv` dataset:
    - Load data.
    - Preprocess data.
    - Perform train/test split.
    - Train a machine learning model.

    Args:
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        StreamingResponse: A live stream of updates about the pipeline's progress.
    """
    iris_csv_path = os.path.join(DATA_FOLDER, "iris_species.csv")

    async def event_stream():
        try:
            yield "data: Step 1: Loading data...\n\n"
            sleep(1)
            df = pd.read_csv(iris_csv_path)

            yield "data: Step 2: Preprocessing data...\n\n"
            sleep(1)
            df.dropna(inplace=True)
            if "Species" in df.columns:
                df["Species"] = df["Species"].astype("category").cat.codes

            df.to_csv(iris_csv_path, index=False)

            yield "data: Step 3: Splitting data...\n\n"
            sleep(1)
            X = df.drop(columns=["Id", "Species"])  # Exclude 'Id' and 'Species'
            y = df["Species"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            yield "data: Step 4: Training model...\n\n"
            sleep(1)
            model_parameters = load_model_parameters().get("RandomForestClassifier", {})
            model = RandomForestClassifier(**model_parameters)
            model.fit(X_train, y_train)

            model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")
            joblib.dump(model, model_path)

            yield f"data: Model trained and saved at {model_path}\n\n"
            yield "data: Full process completed successfully!\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
@router.post("/predict", tags=["model"])
def predict_with_model(features: dict):
    """
    Predict outcomes using a trained RandomForest model.

    Args:
        features (dict): A dictionary containing input features for prediction.

    Returns:
        dict: Predictions and their probabilities for the given input features.

    Raises:
        HTTPException: If the trained model is missing or if the input features are invalid.
    """
    model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained model not found. Train the model first.")

    try:
        model = joblib.load(model_path)

        input_features = np.array(list(features.values())).reshape(1, -1)

        if input_features.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Model expects {model.n_features_in_} features, but received {input_features.shape[1]}"
            )

        predictions = model.predict(input_features)
        probabilities = model.predict_proba(input_features)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@router.post("/predict/ui", response_class=HTMLResponse, tags=["model"])
async def predict_from_ui(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    """
    Process input from a prediction form and predict outcomes using a trained model.

    Args:
        request (Request): The HTTP request object.
        sepal_length (float): Sepal length value provided by the user.
        sepal_width (float): Sepal width value provided by the user.
        petal_length (float): Petal length value provided by the user.
        petal_width (float): Petal width value provided by the user.

    Returns:
        HTMLResponse: A rendered HTML page showing the prediction results.
    """
    input_features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    predictions = None
    try:
        model_response = predict_with_model(input_features)
        predictions = model_response
    except HTTPException as e:
        predictions = {"error": e.detail}

    return templates.TemplateResponse(
        "predict_result.html",
        {"request": request, "predictions": predictions, "features": input_features}
    )


@router.post("/create-parameters", tags=["firestore"])
def create_firestore_parameters():
    """
    Create a Firestore document containing default model parameters.

    Returns:
        dict: A confirmation message and the parameters saved.

    Raises:
        HTTPException: If an error occurs while creating the document.
    """
    try:
        parameters = {
            "n_estimators": 100,
            "criterion": "gini"
        }

        doc_ref = db.collection("parameters").document("parameters")
        doc_ref.set(parameters)

        return {"message": "Firestore document 'parameters' created successfully", "parameters": parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating Firestore document: {str(e)}")


@router.get("/retrieve-parameters", tags=["firestore"])
def retrieve_firestore_parameters():
    """
    Retrieve model parameters from a Firestore document.

    Returns:
        dict: Retrieved parameters.

    Raises:
        HTTPException: If the document is not found or an error occurs during retrieval.
    """
    try:
        doc_ref = db.collection("parameters").document("parameters")
        doc = doc_ref.get()

        if doc.exists:
            parameters = doc.to_dict()
            return {"message": "Parameters retrieved successfully", "parameters": parameters}
        else:
            raise HTTPException(status_code=404, detail="Document 'parameters' not found in Firestore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parameters: {str(e)}")


@router.get("/protected-endpoint")
@limiter.limit("5/minute")  # Limit to 5 requests per minute per user
async def protected_endpoint(request: Request):
    """
    A rate-limited endpoint accessible by users.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        dict: A success message indicating the endpoint is accessible.
    """
    return {"message": "This endpoint is rate-limited per user!"}
