from src.pipelines.load_data import dataRaw
from src.pipelines.preprocessing_pipeline import preprocessing
from src.pipelines.clustering_pipeline import nCluster, clustering
from src.pipelines.inverse_pipeline import inverse
from src.pipelines.classification_pipeline import classify
from src.pipelines.evaluation_pipeline import evaluation


def main():
    print("\n==========================")
    print("  TRAINING PIPELINE START ")
    print("==========================\n")

    # 1. Load Raw Data
    print("[1] Loading raw data...")
    df_raw = dataRaw()
    print(f"    ✔ Raw data loaded: {df_raw.shape[0]} rows")

    # 2. Preprocessing
    print("[2] Running preprocessing...")
    preprocessing()
    print("    ✔ Preprocessing finished")

    # 3. Clustering (Elbow + KMeans)
    print("[3] Running clustering...")
    nCluster()
    clustering()
    print("    ✔ Clustering finished")

    # 4. Inverse Transform
    print("[4] Running inverse transformation...")
    inverse()
    print("    ✔ Inverse data generated")

    # 5. Classification
    print("[5] Running classification...")
    classify()
    print("    ✔ Classification finished")

    # 6. Evaluation
    print("[6] Running evaluation...")
    model_paths = {
        'DecisionTreeClassifier': 'src/models/DecisionTreeClassifier_best.pkl',
        'LogisticRegression': 'src/models/LogisticRegression_best.pkl',
        'RandomForestClassifier': 'src/models/RandomForestClassifier_best.pkl'
    }
    data_path = "data/04-inverse/data_inverse.csv"

    eval = evaluation(model_paths, data_path)
    eval.plt_confusion_matrix()
    eval.plt_classification_report()

    print("    ✔ Evaluation saved")

    print("\n==========================")
    print("  TRAINING PIPELINE DONE! ")
    print("==========================\n")

if __name__ == "__main__":
    main()