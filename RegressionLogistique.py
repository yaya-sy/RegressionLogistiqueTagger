from outils import *
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from math import sqrt
from math import exp, log
from itertools import chain
import random as rn

class RegressionLogistique :
    """
    Cette classe implémente la régression logistique multiclasse

    méthodes :
    --------
    - init_parametres :
        fonctions qui initialise les paramètres du modèle selon le nombre de classes des données
    - score :
        applique la fonction de score linéaire entre deux vecteurs
    - probabilite :
        estime la probabilité qu'un vecteur d'observation soit associé à une classe particulière
    - classifie : fontion qui classifie un vecteur d'observation à partir d'un ensemble de vecteurs de paramètres
    - dgs : méthode qui estime les paramètres du modèle en utilisant la méthode d'optimisation numérique de descente de gradient.

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
        - w : defaultdict
            les paramètres initialisés
        """
        w = defaultdict(str)
        for classe in self.classes :
            w[classe] = defaultdict(float)
        return w

    def score(self: object, w_y: dict, features:dict) -> float :
        """
        Fonction qui fait le produit scalaire entre un vecteur de caractéristiques et un vecteur de features

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
        return exp(sum(w_y[c] * features[c] for c in features))

    def probabilite(self : object, w : defaultdict, classe : str, features: dict) -> float :
        """
        Fonction calcule la probabilité qu'une feature appartienne à la classe en paramètre

        Parameters :
        ----------
        - w : l'ensemble des vecteur de paramètres
        - classe : la classe supposée de la features
        - features : caractéristiques d'une instance particulière

        Returns :
        -------
        probabilite : float:
            la proba de la clase sachant les caractéristiques
        """
        w[classe] = random.randint(0.01, 0.1) if w[classe] == 0 else w[classe]
        return self.score(w[classe], features) / sum(self.score(w[y], features) for y in w)

    def classify(self: object, w: defaultdict, features: dict) -> str:
        """
        Fonction qui classifie en appliquant la règle de décision : la classe du vecteur qui a le meilleure score est choisi;

        Parameters :
        ----------
        - w : defaultdict
            vecteurs de paramètres
        - features : dict
            caractéristiques d'une instance à classifier

        Returns :
        -------
        classe : str
            la classe prédite
        """
        scores = {label : self.score(w[label], features)  for label in w}
        return max((score, label) for label, score in scores.items())[1]

    def precision(self: object, instances: object, w: defaultdict) -> float :
        """
        fonction qui calcule la précision du système sur des instances

        Paramaters :
        ----------
        - instances : dict
            caractéristiques
        - w : defaultdict
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
            if self.classify(w, inst) == tag :
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
        - w : defaultdict
            les paramètres estimés
        """
        rn.shuffle(self.instances)
        w = self.init_parametres()
        alpha = alpha0
        for i in range(iterations) :
            for features, tag in self.instances :
                z = sum(self.score(w[y], features) for y in w)
                for classe in w :
                    p = self.score(w[classe], features) / z # self.probabilite(w, classe, features)
                    grad = {f : alpha * ((int(tag == classe) - p) * features[f]) for f in features}
                    for f in grad :
                        w[classe][f] += grad[f]
            alpha = alpha0 / sqrt(i + 1)
            prec = self.precision(self.instances, w)
            print("it : {}, precision sur le train : {}".format(i + 1, prec))
        return w

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

    classes = set(tag for prase, tags in train for tag in tags)

    lr = RegressionLogistique(classes, ftr)
    w = lr.dgs(0.5, 20)                                         #wo : 0.2, 10 = 0.914, ru : 0.42, 10, fr : 0.42, 40
    print('précision sur le test : ', lr.precision(fte, w))
