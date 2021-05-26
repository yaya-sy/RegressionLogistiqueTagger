
from collections import Counter, defaultdict
from itertools import chain

def features(phrase, tags, span_w, span_s, position) :
    """
    Fonction qui extrait les features  en créant un dictionnaire : pour chaque mot (par exemple 'désinfection'), on regarde
        - pref1 (son préfixe de longueur 1) : 'd'
        - pref2 (son préfixe de longueur 2) : 'dé'
        - etc
        - suff1 (son suffixe de longueur 1) : 'n'
        - suff2 (son suffixe de longueur 2) : 'on'
        - suff3 (son suffixe de longueur 3) : 'ion'
        - etc.
    ------ parametre :
    - phrase : liste des phrase en mots
    - tags : liste des étiquettes alignées avec les mots des phrases
    - span_w : longueur max des affixes à regarder sur le mot
    - span_s : span à extraire de la phrase
    """
    l = len(phrase)
    p = ["BEG"] * (span_s + 1) + phrase + ["END"] * (span_s + 1)
    t = ["BEG"] * (span_s + 1) + tags + ["END"] * (span_s + 1)
    features = defaultdict(str)

    if p[position] not in ["BEG", "END"] : #and mot in mc and mot not in mots :
        for spann in chain.from_iterable([(-i, i) for i in range(1, span_s + 1)]) :
            features["w_" + str(spann) + " = " + p[position + spann]] = 1

        for span in chain.from_iterable([(-i, i) for i in range(1, span_w + 1)]):
            if span < 0 :
                if len(p[position]) >= abs(span) :
                    features["pref" + str(abs(span)) + "_" + p[position][:-span]] = 1
                else :
                    features["pref" + str(abs(span)) + "_" + "HayDara"] = 1
            else :
                if len(p[position]) >= abs(span) :
                    features["suff" + str(span) + "_" + p[position][-span:]] = 1
                else :
                    features["suff" + str(abs(span)) + "_" + "HayDara"] = 1

        features["maj{}".format(p[position][0].isupper())] = 1
        features["mot = {}".format(p[position])] = 1
        features["Maj{}".format(p[position].isupper())] = 1
        features["num{}".format(p[position].isnumeric())] = 1
        features["num{}".format(len(p[position]) > 7)] = 1
    if bool(features) :
        return features, t[position]


def get_features(corpus) :
    """
    Fonction qui extrait les caractéristiques sur tout un corpus
    ----- parametres :
    - corpus : corpus sur lequel on veut extraire les caractéristiques
    """
    fts = []
    for phrase, labels in corpus :
        for i in range(len(phrase)) :
            f = features(phrase, labels, 4, 2, i)
            if f != None :
                fts.append(f)
    return fts

def accuracy(instances, w) :
    """
    fonction qui calcule la précision du système sur des instances
    - instances : caractéristiques
    - w : paramètres estimées
    """
    bons = 0.0
    total = 0.0
    for inst, tag in instances :
        total += 1
        if classify(w, inst) == tag :
            bons += 1
    return bons / total

def read_corpus(corpus):
    """
    Fonction qui lit un corpus universal dépendencies
    """
    for sentence in corpus.read().split("\n\n"):
        if not sentence:
            continue
        data = (line.split("\t")[:4] for line in sentence.split("\n") if not line.startswith("#"))
        data = ((word, pos) for indice, word, _, pos in data if "-" not in indice and "." not in indice)

        wds, poss = zip(*((w, pos) for w, pos in data)) #le mot clé yield va nous permettre de réduire la complexité en espace en retournant

        yield [list(wds), list(poss)]
