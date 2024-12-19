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



router = APIRouter()

templates = Jinja2Templates(directory="src/template")
DATA_FOLDER = "src/data"
JSON_FOLDER = "src/config"
MODELS_FOLDER = "src/models"
DATASETS_JSON = os.path.join(JSON_FOLDER, "dataset.json")
MODEL_PARAMETERS_FILE = os.path.join(JSON_FOLDER, "model_parameters.json")

# Fonction pour lire le fichier JSON des datasets
def load_datasets():
    if os.path.exists(JSON_FOLDER):
        with open(DATASETS_JSON, "r") as file:
            return json.load(file)
    return {}

# Fonction pour sauvegarder le fichier JSON des datasets
def save_datasets(datasets):
    with open(DATASETS_JSON, "w") as file:
        json.dump(datasets, file, indent=4)

def load_csv_from_folder(folder_path: str):
    """
    Charge tous les fichiers CSV présents dans un dossier et les retourne sous forme de dictionnaire
    avec le nom du fichier comme clé et le DataFrame comme valeur.
    
    :param folder_path: Chemin vers le dossier contenant les fichiers CSV
    :return: Dictionnaire où chaque clé est le nom du fichier CSV et chaque valeur est le DataFrame
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
    Load model parameters from the JSON file.
    """
    if os.path.exists(MODEL_PARAMETERS_FILE):
        with open(MODEL_PARAMETERS_FILE, "r") as file:
            return json.load(file)
    return {}

@router.get("/train/ui", response_class=HTMLResponse, tags=["model"])
def get_prediction_ui(request: Request):
    """
    Serve an HTML form for entering prediction parameters.
    """
    return templates.TemplateResponse("full_process.html", {"request": request})

@router.get("/predict/ui", response_class=HTMLResponse, tags=["model"])
def get_prediction_ui(request: Request):
    """
    Serve an HTML form for entering prediction parameters.
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

    Inputs:
        - dataset_name (str): The name of the dataset to be downloaded. This must match
          one of the keys in the `datasets.json` file.

    Outputs:
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
    Affiche un formulaire HTML pour entrer une URL de dataset Kaggle.
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
    Gère les datasets dans le fichier JSON : ajouter ou supprimer un dataset basé sur l'URL.
    
    Args:
        dataset_name (str): Le nom du dataset.
        dataset_url (str): L'URL du dataset Kaggle.
    
    Returns:
        HTMLResponse: Confirmation ou message d'erreur.
    """
    datasets = load_datasets()

    # Vérifier si le dataset existe déjà
    dataset_key = dataset_name.lower().replace(" ", "_")
    
    if dataset_key in datasets:
        # Si le dataset existe déjà, le supprimer
        del datasets[dataset_key]
        message = f"Le dataset '{dataset_name}' a été supprimé."
    else:
        # Si le dataset n'existe pas, l'ajouter
        datasets[dataset_key] = {
            "name": dataset_name,
            "url": dataset_url,
            "csv": dataset_key + ".csv"
        }
        message = f"Le dataset '{dataset_name}' a été ajouté avec succès."
    
    # Sauvegarde des datasets dans le fichier JSON
    save_datasets(datasets)

    # Retour à l'utilisateur avec un message de confirmation
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h1>{message}</h1>
            <p>{message}</p>
            <a href="/api/manage-datasets">Retour au formulaire</a>
        </body>
        </html>
    """, status_code=200)

@router.get("/datasets", response_class=HTMLResponse, tags=["data"])
def show_datasets():
    """
    Affiche un tableau HTML avec les datasets disponibles.
    """
    datasets = load_datasets()

    if not datasets:
        return HTMLResponse(content="<h1>Aucun dataset disponible</h1>", status_code=404)

    # Créer un tableau HTML pour afficher les datasets
    table_content = "<table border='1' style='width:100%'><tr><th>Nom</th><th>URL</th>"

    # Ajouter chaque dataset au tableau
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
            <h1>Datasets Disponibles</h1>
            {table_content}
            <br><br>
            <a href="/docs">Retour à la gestion</a>
        </body>
        </html>
    """, status_code=200)


@router.get("/datasets/iris_species", response_class=HTMLResponse, tags=["data"])
def show_iris_dataset():
    """
    Charge et affiche les données du fichier iris_species.csv dans un tableau HTML.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    # Vérifier si le fichier CSV existe
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail=f"Le fichier '{iris_csv_path}' n'a pas été trouvé.")
    
    # Charger le fichier CSV dans un DataFrame pandas
    try:
        df = pd.read_csv(iris_csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des données : {str(e)}")

    # Convertir les premières lignes du DataFrame en tableau HTML
    table_html = df.to_html(index=False, classes='data-table', border=1)

    # Retourner la réponse HTML avec le tableau des données
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Voir les données - Iris Species</title>
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
            <h1>Données du dataset: Iris Species</h1>
            <p>Affichage des premières lignes du dataset iris_species.csv</p>
            {table_html}
            <br><br>
            <a href='/api/datasets'>Retour à la liste des datasets</a>
        </body>
        </html>
    """, status_code=200)

@router.get("/datasets/iris_species/json", tags=["data"])
def get_iris_dataset_as_json():
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="Le fichier 'iris_species.csv' n'a pas été trouvé.")
    try:
        df = pd.read_csv(iris_csv_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des données : {str(e)}")


@router.get("/datasets/iris_species/preprocessed", tags=["data"])
def preprocess_iris_dataset():
    """
    Prétraite le dataset Iris : gestion des valeurs manquantes, encodage, etc.
    """
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="Le fichier 'iris_species.csv' n'a pas été trouvé.")
    
    try:
        df = pd.read_csv(iris_csv_path)

        # Exemple de prétraitement
        # 1. Suppression des valeurs manquantes
        df.dropna(inplace=True)
        
        # 2. Encodage des colonnes catégoriques (si nécessaire)
        if "species" in df.columns:
            df["species"] = df["species"].astype("category").cat.codes
        
    
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des données : {str(e)}")


@router.api_route("/datasets/split", methods=["GET", "POST"], tags=["data"])
def train_test_split_endpoint(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Effectue un train/test split sur un dataset donné.
    Les paramètres peuvent être passés via une requête GET ou un formulaire POST.
    """
    
    iris_csv_path = DATA_FOLDER + "/iris_species.csv"

    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail=f"Le fichier '{iris_csv_path}.csv' n'a pas été trouvé.")

    try:
        # Étape 1 : Charger les données
        sleep(1)
        df = pd.read_csv(iris_csv_path)

        # Étape 2 : Prétraitement
        sleep(1)
        df.dropna(inplace=True)
        if "Species" in df.columns:
            df["Species"] = df["Species"].astype("category").cat.codes

        # Étape 3 : Division des données
        sleep(1)
        X = df.drop("Species", axis=1)
        y = df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Préparer les résultats
        train_data = pd.concat([X_train, y_train], axis=1).to_json(orient="records")
        test_data = pd.concat([X_test, y_test], axis=1).to_json(orient="records")

        # Étape finale : Envoi des résultats
        yield f"data: Train Split: {train_data}\n\n"
        yield f"data: Test Split: {test_data}\n\n"
        yield "data: Processus terminé avec succès !\n\n"
    except Exception as e:
        yield f"data: Erreur : {str(e)}\n\n"

@router.post("/train-model", tags=["model"])
def train_classification_model():
    """
    Entraîne un modèle de classification à partir du dataset traité.
    Enregistre le modèle dans le dossier src/models.
    """
    iris_csv_path = os.path.join(DATA_FOLDER, "iris_species.csv")
    if not os.path.exists(iris_csv_path):
        raise HTTPException(status_code=404, detail="Le dataset traité est introuvable.")

    # Charger le dataset
    try:
        df = pd.read_csv(iris_csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset : {str(e)}")
    
    # Vérifier que la colonne cible 'Species' est présente
    if "Species" not in df.columns:
        raise HTTPException(status_code=500, detail="Le dataset doit inclure la colonne 'Species'.")

    try:
        # Séparation des features et de la cible
        X = df.drop("Species", axis=1)
        y = df["Species"]

        # Charger les paramètres du modèle depuis model_parameters.json
        model_parameters = load_model_parameters().get("RandomForestClassifier", {})
        model = RandomForestClassifier(**model_parameters)

        # Entraîner le modèle
        model.fit(X, y)

        # Sauvegarder le modèle entraîné
        model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")
        joblib.dump(model, model_path)

        return {
            "message": "Modèle entraîné et sauvegardé avec succès.",
            "model_path": model_path,
            "parameters": model_parameters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement du modèle : {str(e)}")

@router.get("/datasets/iris_species/full_process", tags=["data"])
async def full_process_iris_dataset_stream(test_size: float = 0.2, random_state: int = 42):
    """
    Exécute le processus complet avec des événements en streaming :
    - Chargement des données
    - Prétraitement
    - Train/Test Split
    - Entraînement du modèle
    """
    iris_csv_path = os.path.join(DATA_FOLDER, "iris_species.csv")

    async def event_stream():
        try:
            # Étape 1 : Chargement des données
            yield "data: Étape 1 : Chargement des données...\n\n"
            sleep(1)
            df = pd.read_csv(iris_csv_path)

            # Étape 2 : Prétraitement
            yield "data: Étape 2 : Prétraitement...\n\n"
            sleep(1)
            df.dropna(inplace=True)
            if "Species" in df.columns:
                df["Species"] = df["Species"].astype("category").cat.codes

            # Sauvegarde des données prétraitées
            df.to_csv(iris_csv_path, index=False)

            # Étape 3 : Division des données
            yield "data: Étape 3 : Division des données...\n\n"
            sleep(1)
            X = df.drop(columns=["Id", "Species"])  # Exclude 'Id' and 'Species'
            y = df["Species"]
            yield f"data: X : {X}\n\n"

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Préparer les résultats
            train_data = pd.concat([X_train, y_train], axis=1).to_json(orient="records")
            test_data = pd.concat([X_test, y_test], axis=1).to_json(orient="records")
            if train_data :
                yield f"data: Train Split: ok \n\n"
            if test_data:
                yield f"data: Test Split: ok \n\n"

            # Étape 4 : Entraînement du modèle
            yield "data: Étape 4 : Entraînement du modèle avec train split...\n\n"
            sleep(1)
            model_parameters = load_model_parameters().get("RandomForestClassifier", {})
            model = RandomForestClassifier(**model_parameters)
            model.fit(X_train, y_train)  # Utilisation des splits d'entraînement

            # Sauvegarde du modèle
            model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")
            joblib.dump(model, model_path)

            yield f"data: Modèle entraîné sur train split et sauvegardé à {model_path}\n\n"

            # Fin du processus
            yield "data: Processus complet terminé avec succès !\n\n"
        except Exception as e:
            yield f"data: Erreur : {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/predict", tags=["model"])
def predict_with_model(features: dict):
    """
    Endpoint to make predictions with the trained RandomForest model.

    Args:
        features (dict): Input features for prediction, as a JSON object.

    Returns:
        JSON: Predictions for the input features.
    """
    model_path = os.path.join(MODELS_FOLDER, "iris_random_forest_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained model not found. Train the model first.")

    try:
        model = joblib.load(model_path)

        # Convert to NumPy array
        input_features = np.array(list(features.values())).reshape(1, -1)

        if input_features.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,

                detail=f"Model expects {model.n_features_in_} features, but received {input_features.shape[1]}"
            )

        # Make predictions
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
    Process input from the prediction form and send it to the /predict endpoint.
    """
    input_features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    # Send data to the /predict endpoint
    predictions = None
    try:
        model_response = predict_with_model(input_features)
        predictions = model_response
    except HTTPException as e:
        predictions = {"error": e.detail}

    # Render the response in HTML
    return templates.TemplateResponse(
        "predict_result.html",
        {"request": request, "predictions": predictions, "features": input_features}
    )
