Simulation d'une invasion de zombies à partir de la ville de Rize en Turquie. En combien de temps arriveront-ils à Brest ?
-> Présentation1.pdf : présentation ppt qui explique la démarche

Problèmes de preprocessing rencontrés : 
	* Les deux maps (élévation+densité de population) qui ne sont pas superposables -> usage du machine learning, computer vision/homographie
	* Comment ordonner les couleurs par ordre croissant pour faire la correspondance numérique avec la densité de population ou l'élévation -> usage du machine learning (Nearest Neighbours)

-> withGraph.py : implémentation en utilisant un graphe
-> challenge3_skeleton1.py : implémentation en orienté objet
L'implémentation en utilisant un graphe est beaucoup plus rapide à exécuter.

Détails sur l'énoncé : zombin.pdf