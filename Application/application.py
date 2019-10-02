from ImageRetrieval.Ranking import RankingSimilar
from definitions import QSD1_PATH, BBDD_PATH, K, CORRESPONDANCE_FILE
import pickle
import os
import ml_metrics as metrics


def find_k_most_similar(query):
    candidates = []
    for file in os.listdir(BBDD_PATH):
        if file.endswith(".jpg"):
            candidates.append(os.path.join(BBDD_PATH, file))

    ranking = RankingSimilar(query, candidates, K)
    return ranking.findKMostSimilar("chi")


def evaluate(actual, predicted):
    return metrics.apk(actual, predicted)


if __name__ == "__main__":
    files_q1   = []
    results_q1 = []
    for file in os.listdir(QSD1_PATH):
        if file.endswith(".jpg"):
            path = os.path.join(QSD1_PATH, file)
            files_q1.append(path)

    with open(CORRESPONDANCE_FILE, 'rb') as f:
        correspondance = pickle.load(f)

    correspondance_dict = {}
    for tuple in correspondance:
        correspondance_dict.update(dict(tuple))

    print(correspondance_dict)

    for (i, file_q1) in enumerate(sorted(files_q1), 1):
        similars = find_k_most_similar(file_q1)
        results_index = [index for index, path in similars]
        print(results_index)
        print(evaluate([correspondance_dict[i]], results_index))



    '''with open('QS1_RESULTS_2.pkl', 'wb') as f:
        pickle.dump(results, f)'''
