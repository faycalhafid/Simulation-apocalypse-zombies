Simulation d'une invasion de zombies � partir de la ville de Rize en Turquie. En combien de temps arriveront-ils � Brest ?
-> Pr�sentation1.pdf : pr�sentation ppt qui explique la d�marche

Probl�mes de preprocessing rencontr�s : 
	* Les deux maps (�l�vation+densit� de population) qui ne sont pas superposables -> usage du machine learning, computer vision/homographie
	* Comment ordonner les couleurs par ordre croissant pour faire la correspondance num�rique avec la densit� de population ou l'�l�vation -> usage du machine learning (Nearest Neighbours)

-> withGraph.py : impl�mentation en utilisant un graphe
-> challenge3_skeleton1.py : impl�mentation en orient� objet
L'impl�mentation en utilisant un graphe est beaucoup plus rapide � ex�cuter.

D�tails sur l'�nonc� : zombin.pdf