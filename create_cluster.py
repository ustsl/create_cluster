from typing import Iterable
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


class CreateCluster:
    # На входе передаем  датасет,
    # параметр кластеризации,
    # колонку, по которой будем вести кластеризацию

    # В result получаем результат после исполнения clasterize для создания кластеров
    # Дополнительно при исполнении get_clasters_name для получения нормальных имен кластеров
    # В статикметод вынесена функция, которую можно использовать отдельно от класса

    def __init__(
        self, df: pd.DataFrame, n_clusters: Iterable[int], clasters_column: str
    ):
        self.n_clusters = n_clusters
        self.df = df
        self.clasters_column = clasters_column
        self.result = None

    def clasterize(self):
        # Вычислите TF-IDF векторы для всех фраз
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.df[self.clasters_column])

        # Определите параметры для сеточного поиска
        param_grid = {
            "n_clusters": self.n_clusters,
            "init": ["k-means++", "random"],
        }

        # Определите пользовательскую функцию оценки, возвращающую отрицательное значение инерции
        def neg_inertia_score(estimator, X):
            return -estimator.inertia_

        neg_inertia_scorer = make_scorer(neg_inertia_score)

        # Кластеризация с использованием KMeans и сеточного поиска для определения оптимальных параметров
        kmeans = KMeans(random_state=42)
        grid_search = GridSearchCV(kmeans, param_grid, scoring=neg_inertia_scorer, cv=3)
        grid_search.fit(X)

        # Примените лучшую модель на данных
        best_kmeans = grid_search.best_estimator_
        self.df["cluster"] = best_kmeans.predict(X)
        self.result = self.df

    def get_clasters_names(self):
        if self.result is not None:
            data = self.result
            map_name = {}

            for cluster in data.cluster.unique():
                segment = tuple(data[data["cluster"] == cluster][self.clasters_column])
                name = self.get_claster_name(
                    current_names=tuple(map_name.values()), claster_body=segment
                )
                map_name[cluster] = name

            def add_claster_name(row):
                return map_name[row["cluster"]]

            data["cluster"] = data.apply(add_claster_name, axis=1)
            self.result = data

    @staticmethod
    def get_claster_name(
        current_names: Iterable[str], claster_body: Iterable[str]
    ) -> str:
        for item in claster_body:
            if item not in current_names:
                return item
