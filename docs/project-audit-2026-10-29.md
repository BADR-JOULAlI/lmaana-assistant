# Audit du projet et prompt exécuté — 29 octobre 2026

## Constats issus du code

Le projet possède déjà un pipeline modulaire : ingestion de sources revues,
index Qdrant versionné, normalisation multilingue, recherche, génération locale,
validation des citations et transcription Lmaana 2.4 avec worker WSL. Les composants
communiquent par des contrats Python et HTTP. La priorité est de fiabiliser et
mesurer ce chemin existant.

Le graphe AST existant relie notamment `Settings` à `create_app`, `make_asr` et
`LocalLmaanaASR`. Sa lecture a orienté la vérification des configurations et du
contrat ASR, recoupée avec les sources actuelles. L'exécutable Graphify est bloqué
par Windows Application Control lors de cet audit ; le JSON déjà produit reste
lisible. Il ne couvre pas les nouveaux modules et ne constitue pas une analyse
automatiquement actualisée de tout le dépôt.

| Constat avant modification | Conséquence | Traitement |
|---|---|---|
| CI avec `pip install -e .[dev]` malgré `uv.lock` | Versions variables à chaque installation ; UI absente de l'environnement CI | Installation verrouillée avec les extras `dev` et `ui` |
| `doctor` dépend du démarrage des imports applicatifs et des réglages de service | Inspection des paquets difficile en cas de dépendance manquante ou de configuration incorrecte | Commande indépendante `lmaana-env`, JSON et code de sortie |
| Imports directs de Pydantic couverts seulement par des dépendances transitives | Contrat d'installation implicite | Dépendance Pydantic déclarée explicitement |
| `.env.example` active l'ASR et des chemins propres à une machine | Configuration copiée incohérente avec le démarrage minimal annoncé | ASR désactivé par défaut, chemins documentés en commentaires |
| Ports locaux invalides acceptés par les réglages | Échec plus tard lors de la construction du client HTTP | Validation au chargement de la configuration |
| Contrat ASR acceptant du texte blanc, des segments inversés/hors durée ou des temps non finis | Transcription invalide transmise à l'interface | Validation du résultat et erreur de dépendance du pont HTTP |

## Dépendances et rôle de chaque profil

| Profil | Dépendances | Usage |
|---|---|---|
| API et contrats | FastAPI, Uvicorn, Pydantic, pydantic-settings, python-multipart | HTTP, configuration, validation des entrées/sorties, audio envoyé |
| Ingestion | Beautiful Soup, pypdf, HTTPX, truststore | Extraction HTML/PDF et téléchargement HTTPS |
| Recherche | qdrant-client | Index vectoriel local et versions du corpus |
| Diagnostic | packaging | Analyse des contraintes de versions et marqueurs des dépendances |
| `ui` | Streamlit et ses dépendances transitives | Interface de questions écrites et vocales |
| `models` | Sentence Transformers ; Torch et Transformers transitifs | Adaptateur Qwen3-Embedding optionnel |
| `dev` | pytest, Ruff | Tests et qualité du code |
| ASR natif séparé | Torch/CUDA, fairseq2, fork OmniASR, checkpoint et tokenizer Lmaana 2.4 | Environnement WSL existant, géré indépendamment du runtime texte |
| Génération séparée | Service Ollama ou llama.cpp et poids Qwen3 | Appels HTTP depuis les adaptateurs du projet |

Le lockfile contient 114 distributions au total, en incluant les dépendances
optionnelles et de plateforme. La sélection `dev` + `ui` installée pour cet audit
contient 66 distributions. Ces nombres décrivent des ensembles différents.
La modification ajoute les déclarations directes `packaging` et `pydantic` ; les
versions déjà verrouillées sont conservées.

`lmaana-env` vérifie la version Python, les dépendances directes du paquet installé
et les extras demandés. Les marqueurs de plateforme/Python sont pris en compte.
Il signale les paquets absents, versions incompatibles, métadonnées de version
invalides et références de source non vérifiées. Il n'importe pas les modèles et
ne contacte aucun service. Un rapport satisfaisant ne vérifie pas les DLL/CUDA,
les poids, les dépendances transitives ou la qualité linguistique.

La CI applique aussi `uv pip check` aux dépendances transitives installées.
Les étapes suivent les commandes décrites dans la
[documentation uv](https://docs.astral.sh/uv/guides/integration/github/).
Le diagnostic utilise l'[analyseur de contraintes packaging](https://packaging.pypa.io/en/stable/requirements.html).

## Prompt exécuté

> Analyse les dépendances et les frontières du pipeline Lmaana Assistant. Fiabilise
> l'installation et le diagnostic : utilise uv.lock en CI avec les profils dev/ui,
> ajoute une commande indépendante d'inspection des métadonnées des dépendances,
> et déclare les imports directs nécessaires. Corrige les ports locaux invalides,
> les incohérences de .env.example et la validation des résultats ASR : texte
> non blanc, segments compris dans la durée audio, temps finis et positifs ou nuls.
> Conserve le texte original des transcriptions. Vérifie ces modifications avec
> des réponses HTTP simulées et les tests du projet. Documente les résultats et
> les prochains ajouts à prioriser. Préserve Lmaana/lmaana-2.4, le corpus actif,
> les réglages locaux et les contrôles Windows. N'annonce des mesures de modèles
> que si leur inférence a réellement été exécutée.

## Exécution reproductible

```powershell
uv sync --locked --extra dev --extra ui
uv pip check
uv run --no-sync python -m lmaana_assistant.diagnostics --profile ui --profile dev
uv run --no-sync python -m pytest -q
uv run --no-sync python -m ruff check src tests apps
uv run --no-sync python -m ruff format --check src tests apps
```

Pour cet audit local, `UV_PROJECT_ENVIRONMENT=work/project-audit/.venv` a permis
de vérifier le lockfile dans un nouvel environnement Python 3.12.10. Les commandes
de diagnostic peuvent aussi être lancées directement avec le Python de cet
environnement. Les modèles et les services locaux ne sont pas requis par ces tests.

Résultats locaux : **384 tests réussis** sur Windows/Python 3.12.10, Ruff et
formatage conformes, lockfile à jour et compatibilité des 66 distributions installées
validée par `uv pip check`. Le diagnostic `ui` + `dev` retourne
`requirements_satisfied: true`. La suite émet un avertissement de dépréciation
Starlette concernant son intégration HTTPX ; il reste à suivre lors d'une évolution
du runtime de tests. La matrice Linux/Windows et Python 3.11/3.12 est configurée
dans la CI ; ces quatre exécutions distantes ne sont pas des résultats de l'essai local.

## Suite exécutée — validation des vecteurs

La continuation implémente aussi la validation numérique à la frontière Qdrant.
La même fonction vérifie et normalise désormais les vecteurs pour le stockage et
le benchmark. L'ingestion contrôle tous les embeddings avant de créer une
collection : nombre, dimension, valeurs numériques finies et direction non nulle.
Une ingestion invalide conserve le manifeste actif et ne crée aucune collection.

Les vecteurs valides sont normalisés avant la conversion interne en float32 ;
des tests avec des amplitudes `1e300` et `1e-300` vérifient des scores finis.
Une requête nulle renvoie une liste vide sans classement Qdrant. Une requête
malformée produit une erreur de dépendance HTTP 503 et libère le créneau
d'inférence ; le test vérifie qu'une requête valide suivante peut réussir.

**395 tests passent** après ce lot ; Ruff et le formatage sont conformes. Le
diagnostic lexical étendu a été rejoué avec le même hash de données et les
classements/scores correspondent exactement au rapport local du 22 octobre :
Recall@1 = 17/35 et 3/5 négatifs avec un candidat au seuil 0,18. Ces mesures
concernent toujours la fixture logicielle, sans génération de réponses.
Rapport local : `outputs/retrieval-lexical-expanded-vector-guard.json`.

## Prochains ajouts à prioriser

1. **Jeu d'évaluation documentaire réel et revu.** Ajouter des questions et
   annotations avec sources exactes, réponses attendues, négatifs et variantes
   Darija/Arabizi/français/arabe/anglais. Séparer développement et test final par
   scénario/source. Les 40 questions logicielles actuelles servent au développement.
2. **Comparaison sémantique mesurée.** Reprendre les deux modes du benchmark Qwen
   après validation du composant Windows bloqué. Comparer rappel, rangs, négatifs
   et latence à corpus identique, puis calibrer sur un jeu distinct. Le dernier
   essai documenté du 22 octobre n'a pas exécuté l'inférence Qwen.
3. **Mesures ASR sur audio pertinent.** Références transcrites et revues, séparation
   des locuteurs, comparaison Lmaana 2.4/Whisper, WER/CER brut et protocole normalisé
   explicitement distinct, temps de chargement et d'inférence.
4. **Recherche hybride et reranking.** Évaluer leur intérêt après le baseline
   sémantique, notamment pour les noms, acronymes et questions avec code-switching.
   Les conserver uniquement si leur gain mesuré justifie le coût supplémentaire.
5. **Couverture humaine des réponses.** Vérifier les affirmations, conditions
   omises, qualité Darija et raisons d'abstention sur des sorties annotées.
6. **Exploitation.** Mesurer mémoire et coexistence GPU, temps de disponibilité,
   renouvellement des revues de sources, puis définir les besoins d'historique,
   export ou TTS à partir d'usages observés.

Le pipeline actuel fournit déjà l'orchestration nécessaire. L'ajout de LangChain,
LangGraph, d'une base externe ou de conteneurs peut être réévalué si un besoin
mesuré le demande. La taille actuelle du code ne justifie pas à elle seule ces
dépendances supplémentaires.
