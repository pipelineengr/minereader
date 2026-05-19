# config.py
# Central place for all dataset column mappings and training hyperparameters.
# Changing a column name here propagates everywhere — no hardcoded strings in model files.

DEVICE = "cuda"  # will fall back to cpu in prepare_graph.py if cuda unavailable

DATASET_CONFIGS = {
    "marvin": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "grade": "@AU",         # primary prediction target
        "density": "%Density",  
        "aux_grade": "@CU",     
    },
    "mclaughlin": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "grade": "AU",          
        "density": "density",   
        "aux_grade": None,      # No copper grade present in McLaughlin, can be added if it's present in another dataset
    },
}

# Graph construction
KNN_K = 8                       # each block connects to its 8 nearest spatial neighbours
                                # 6 face-neighbours in a cubic grid + 2 diagonal — a reasonable starting point

# Training
LEARNING_RATE = 1e-3
BATCH_SIZE = 32                 # number of graphs per batch (used in DataLoader)
EPOCHS = 100
TRAIN_SPLIT = 0.8               # 80% train, 20% test for the applied ML experiment

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RUNS_DIR = "runs"