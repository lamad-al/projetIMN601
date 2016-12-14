# projetIMN601

## Main

Pour l’exécution des algorithmes de classificateur, il faut exécuter le « main.py ». 
Option: 
	Avec ou sans GridSearch

##Réseau à neurone

Il y a 4 fichiers pour les réseaux à neurone. 
	- mplMnist.py contient le code pour le multilayer perceptron appliquée à Mnist
	- mplCifar10.py contient le code pour le multilayer perceptron appliquée à Cifar10
	- convNetworkMnist.py contient le code pour le réseau à convolution pour Mnist
	- convNetworkCifar10.py contient le code pour le réseau à convolution pour Cifar10
Les arguments tel que le nombre de filtres ou d'epoch doivent être changés manuellement à l'intérieur du code.

##Donnée d'entrainement, de validation et test

Chaque fichier de code contient une section permettant de spécifier le pourcentage du dataset voulu
Il suffit d'indiquer le dataset, soit Mnist() ou Cifar10(), et le slice 
images = Images(Mnist(), slice=0.1)
