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
    ------ méthodes :
    - init_parametres : fonctions qui initialise les paramètres du modèle selon le nombre de classes des données
    - score : applique la fonction de score linéaire entre deux vecteurs
    - probabilite : estime la probabilité qu'un vecteur d'observation soit associé à une classe particulière
    - classifie : fontion qui classifie un vecteur d'observation à partir d'un ensemble de vecteurs de paramètres
    - dgs : méthode qui estime les paramètres du modèle en utilisant la méthode d'optimisation numérique de descente de gradient.
    ------ attributs :
    - classes : l'ensemble des classes possibles
    - instances : instances à entraîner. Sous formes de (caractéristiques, gold)
    """

    def __init__(self, classes, instances) :
        self.classes = classes
        self.instances = instances

    def init_parametres(self) :
        """
        fonction qui initialise pour chaque classe un vecteur de paramètre
        ----- paramètres :
        - classes : l'ensemble des classes
        """
        w = defaultdict(str)
        for classe in self.classes :
            w[classe] = defaultdict(float)
        return w

    def score(self, w_y, features) :
        """
        Fonction qui fait le produit scalaire entre un vecteur de caractéristiques et un vecteur de features
        ------ parametres :
        - w_y : vecteur de paramètre de la classe y
        - features : caractéristiques d'une instance particulière
        """
        return exp(sum(w_y[c] * features[c] for c in features))

    def probabilite(self, w, classe, features) :
        """
        Fonction calcule la probabilité qu'une feature appartienne à la classe en paramètre
        ------ paramètres :
        - w : l'ensemble des vecteur de paramètres
        - classe : la classe supposée de la features
        - features : caractéristiques d'une instance particulière
        """
        w[classe] = random.randint(0.01, 0.1) if w[classe] == 0 else w[classe]
        return self.score(w[classe], features) / sum(self.score(w[y], features) for y in w)

    def classify(self, w, features) :
        """
        Fonction qui classifie en appliquant la règle de décision : la classe du vecteur qui a le meilleure score est choisi;
        ------- paramètres :
        - w : vecteurs de paramètres
        - features : caractéristiques d'une instance à classifier
        """
        scores = {label : self.score(w[label], features)  for label in w}
        return max((score, label) for label, score in scores.items())[1]

    def precision(self,instances, w) :
        """
        fonction qui calcule la précision du système sur des instances
        - instances : caractéristiques
        - w : paramètres estimées
        """
        bons = 0.0
        total = 0.0
        for inst, tag in instances :
            total += 1
            if self.classify(w, inst) == tag :
                bons += 1
        return bons / total

    def dgs(self, alpha0, iterations, partition = None) :
        """
        Fonction qui estime les paramètres grâce à la descente de gradient stochastique
        ------ paramètres :
        - alpha0 : pas de gradient de départ
        - iterations : nombre d'itération sur le corpus d'instances
        - partition : nombre d'instances à considérer aléatoirement à chaque itération
        """
        if partition == None :
            partition = len(self.instances)
        w = self.init_parametres()
        for i in range(iterations) :
            alpha = alpha0 / sqrt(i + 1)
            insts = rn.choices(self.instances, k=partition)
            for features, tag in insts :
                z = sum(self.score(w[y], features) for y in w)
                for classe in w :
                    p = self.score(w[classe], features) / z #self.probabilite(w, classe, features)
                    grad = {f : alpha * ((int(tag == classe) - p) * features[f]) for f in features}
                    for f in grad :
                        w[classe][f] += grad[f]
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
    print(len(ftr) * 15)

    classes = set(tag for prase, tags in train for tag in tags)

    lr = RegressionLogistique(classes, ftr)
    w = lr.dgs(1, 500, 10)                                         #wo : 0.2, 10 = 0.914, ru : 0.42, 10, fr : 0.42, 40
    print('précision sur le test : ', lr.precision(fte, w))
