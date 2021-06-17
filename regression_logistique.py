"""Module qui implémente la modèle de regression logistique multiclasse"""
from collections import defaultdict
from math import sqrt
from math import exp
import random as rn
from outils import get_features, read_corpus

class RegressionLogistique :
    """Cette classe implémente la régression logistique multiclasse

    méthodes :
    --------
    - init_parametres :
        fonctions qui initialise les paramètres du modèle selon le nombre de classes des données
    - score :
        applique la fonction de score linéaire entre deux vecteurs
    - probabilite :
        estime la probabilité qu'un vecteur d'observation soit associé à une classe particulière
    - classifie : fontion qui classifie un vecteur d'observation à partir \
    d'un ensemble de vecteurs de paramètres
    - dgs : méthode qui estime les paramètres du modèle en utilisant \
    la méthode d'optimisation numérique de descente de gradient.

    attributs :
    ---------
    - classes : set() :
        l'ensemble des classes possibles
    - instances : list[tuples]
        instances à entraîner. Sous formes de (caractéristiques, gold)
    """

    def __init__(self : object, classes: set, instances: list) -> object:
        self.classes = classes
        self.instances = instances

    def init_parametres(self: object) -> defaultdict :
        """
        fonction qui initialise pour chaque classe un vecteur de paramètre

        Paramaters :
        -----------

        Returns :
        - parametres : defaultdict
            les paramètres initialisés
        """
        parameters = defaultdict(str)
        for classe in self.classes :
            parameters[classe] = defaultdict(float)
        return parameters

    #pylint: disable=R0201
    def score(self: object, w_y: dict, fts: dict) -> float :
        """
        Fonction qui fait le produit scalaire entre un vecteur \
        de caractéristiques et un vecteur de features

        Parameters :
        ----------
        - w_y : dict :
            vecteur de paramètre de la classe y
        - features : dict :
            caractéristiques d'une instance particulière

        Returns :
        -------
        - score : float
            score linéaire que les caractéristiques soient dans la classe des paramètres
        """
        return exp(sum(w_y[c] * fts[c] for c in fts))

    def probabilite(self : object, parametres : defaultdict, classe : str, fts: dict) -> float :
        """
        Fonction calcule la probabilité qu'une feature appartienne à la classe en paramètre

        Parameters :
        ----------
        - parametres : l'ensemble des vecteur de paramètres
        - classe : la classe supposée de la features
        - features : caractéristiques d'une instance particulière

        Returns :
        -------
        probabilite : float:
            la proba de la clase sachant les caractéristiques
        """
        parametres[classe] = rn.randint(0.01, 0.1) \
        if parametres[classe] == 0 else parametres[classe]
        return self.score(parametres[classe], fts) \
        / sum(self.score(parametres[y], fts) for y in parametres)

    def classify(self: object, parametres: defaultdict, fts: dict) -> str:
        """
        Fonction qui classifie en appliquant la règle de décision : la classe \
        du vecteur qui a le meilleure score est choisi;

        Parameters :
        ----------
        - parametres : defaultdict
            vecteurs de paramètres
        - features : dict
            caractéristiques d'une instance à classifier

        Returns :
        -------
        classe : str
            la classe prédite
        """
        scores = {label : self.score(parametres[label], fts) for label in parametres}
        return max((score, label) for label, score in scores.items())[1]

    def precision(self: object, instances: object, parametres: defaultdict) -> float :
        """
        fonction qui calcule la précision du système sur des instances

        Paramaters :
        ----------
        - instances : dict
            caractéristiques
        - parametres : defaultdict
            paramètres estimées

        Returns :
        --------
        - precision : float
            la précision du modèle sur les instances
        """
        bons = 0.0
        total = 0.0
        for inst, tag in instances :
            total += 1
            if self.classify(parametres, inst) == tag :
                bons += 1
        return bons / total

    def dgs(self: object, alpha0: float, iterations: float) -> defaultdict :
        """
        Fonction qui estime les paramètres grâce à la descente de gradient stochastique

        Parameters :
        ----------
        - alpha0 : float
            pas de gradient de départ
        - iterations : int
            nombre d'itération sur le corpus d'instances
        - batch : int
            taille de batch à considérer

        Returns :
        -------
        - parametres : defaultdict
            les paramètres estimés
        """
        rn.shuffle(self.instances)
        parametres = self.init_parametres()
        alpha = alpha0
        for i in range(iterations) :
            for fts, tag in self.instances :
                normalisation = sum(self.score(parametres[y], fts) for y in parametres)
                for classe in parametres :
                    prob = self.score(parametres[classe], fts) / normalisation
                    grad = {feat : alpha * ((int(tag == classe) - prob) \
                    * fts[feat]) for feat in fts}
                    for feat in grad :
                        parametres[classe][feat] += grad[feat]
            alpha = alpha0 / sqrt(i + 1)
            prec = self.precision(self.instances, parametres)
            print("it : {}, precision sur le train : {}".format(i + 1, prec))
        return parametres

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)

    args = parser.parse_args()
    train = list(read_corpus(open(args.train)))
    test = list(read_corpus(open(args.test)))

    ftr = get_features(train)
    fte = get_features(test)

    classes_set = set(tag for prase, tags in train for tag in tags) # classe

    lr = RegressionLogistique(classes_set, ftr)
    parametres_estimes = lr.dgs(0.5, 20)
    print('précision sur le test : ', lr.precision(fte, parametres_estimes))
