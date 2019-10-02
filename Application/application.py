from ImageRetrieval.Ranking import RankingSimilar
from definitions import QSD1_PATH, BBDD_PATH, K
import pickle
import os


def find_k_most_similar(query):
    candidates = []
    for file in os.listdir(BBDD_PATH):
        if file.endswith(".jpg"):
            candidates.append(os.path.join(BBDD_PATH, file))

    ranking = RankingSimilar(query, candidates, K)
    return ranking.findKMostSimilar()


if __name__ == "__main__":
    results = []
    for file in os.listdir(QSD1_PATH):
        if file.endswith(".jpg"):
            query = os.path.join(QSD1_PATH, file)
            similars = find_k_most_similar(query)
            results.append(similars)
    print(results)

    with open('QS1_RESULTS_2.pkl', 'wb') as f:
        pickle.dump(results, f)
