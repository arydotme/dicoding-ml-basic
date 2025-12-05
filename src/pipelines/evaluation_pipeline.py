from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns

class evaluation:

    def __init__(self, model_path, data_path):

        self.model = {name: joblib.load(path) for name, path in model_path.items()}

        df = pd.read_csv(data_path)

        self.X = df.drop(columns = ['Cluster'])
        self.y = df['Cluster']

    def plt_confusion_matrix(self):

        for name, model in self.model.items():
            y_pred = self.model[name].predict(self.X)
            cm = confusion_matrix(self.y, y_pred)

            plt.figure(figsize = (8,6))
            sns.heatmap(cm, annot = True, cmap = 'YlGnBu', fmt = 'd')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(f"src/imgs/confusion_matrix/confusion_matrix_{name}.png")
            plt.close()

    def plt_classification_report(self):
        for name, model in self.model.items():
            y_pred = self.model[name].predict(self.X)

            cr = classification_report(self.y, y_pred)

            text = f"Classification Report {name}\n\n{cr}"

            plt.text(0.5, 0.5, text, fontsize = 12, fontfamily = 'monospace')
            plt.axis('off')
            plt.savefig(f"src/imgs/classification_report/classification_report_{name}.png", bbox_inches='tight')
            plt.close()

def main():
    model_path = {
        'DecisionTreeClassifier': 'src/models/DecisionTreeClassifier_best.pkl',
        'LogisticRegression': 'src/models/LogisticRegression_best.pkl',
        'RandomForestClassifier': 'src/models/RandomForestClassifier_best.pkl'
    }
    data_path = "data/04-inverse/data_inverse.csv"

    eval = evaluation(model_path, data_path)

    eval.plt_confusion_matrix()
    eval.plt_classification_report()

if __name__ == '__main__':
    main()