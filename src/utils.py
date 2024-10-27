from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

def report(predictions, y_test):
    print('Accuracy: %s' % accuracy_score(y_test, predictions))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, predictions, labels=[0,1,2]))
    print('Classification Report:')
    print(classification_report(y_test, predictions))

def create_prediction(predictions, test_data_path, output_path):
    data = pd.read_json(test_data_path)
    ans = pd.DataFrame(columns=["id", "rating"])
    
    for id, rating in enumerate(predictions):
        ans.loc[id] = [data['metadata'][id]['id'], int(rating)]
    
    ans.to_csv(output_path, index=False)
