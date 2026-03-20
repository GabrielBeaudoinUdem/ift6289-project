Data coming from the 
[French Lexical Network (fr-LN)](https://www.ortolang.fr/market/lexicons/lexical-system-fr/v3.2)

To re-extract, put fr-LN in the `./src/data` folder with this format: `./src/data/fr-LN-V3_2/ls-fr-V3.2`

To run the extraction script: `python ./src/data/extraction.py`





Ideas for latter
Quelle est la valeur de fct {} pour le mot {} (POS{}) : Exemple d'utulisation du mot: {}


Avec Exemple de la fct demandée
Sans Exemple de la fct demandée

- remove ndoes without lf
- Thinking of only keeping the lfs with at least 100 (or 50) occurences. This would have to be done on the train set...
