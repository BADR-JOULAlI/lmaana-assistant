from course_data import page, p, h, note, code, table, fig, ar, refs

page('20 · LOGICIEL', 'FastAPI, HTTP et contrats typés', 'Un accord explicite entre interface et moteur',
     p('Une <b>API</b> expose des opérations à un autre programme. HTTP fournit notamment une méthode, un chemin, un corps et un code de réponse. GET sert ici à lire l’état du service ; POST transmet une question à traiter. JSON représente les données, sans transporter directement un objet Python en mémoire.'),
     p('FastAPI reçoit le corps de requête et l’associe à des modèles Pydantic. Le contrat AnswerRequest exige une question non vide, limitée à 1 500 caractères, et une préférence de langue autorisée. Un contrat invalide est rejeté avant la recherche. La documentation interactive permet d’explorer le schéma réel. [R11, R12]'),
     code('POST http://127.0.0.1:8000/v1/answers\nContent-Type: application/json\n\n{"question": "Quels documents pour l’inscription ?",\n "response_language": "fr"}'),
     table(['Code HTTP', 'Sens dans le projet', 'Réaction appropriée'], [
         ['200', 'Traitement terminé, réponse ou abstention.', 'Lire outcome et generation_failed.'],
         ['422', 'Entrée non conforme.', 'Corriger la requête.'],
         ['429', 'Créneau d’inférence occupé.', 'Respecter Retry-After avant de réessayer.'],
         ['503', 'Corpus ou dépendance indisponible.', 'Consulter readiness et la configuration.'],
         ['504', 'Délai de génération dépassé.', 'Vérifier le runtime et l’état de l’API.']]),
     p('Un HTTP 200 ne dit pas que la réponse est factuellement correcte. Il indique que l’API a traité la requête selon son protocole. De même, readiness vérifie des préconditions techniques ; cela ne prédit pas parfaitement une future allocation GPU ou la qualité de la langue générée.'),
     note('La réponse inclut request_id, versions, citations, avertissements et timings. Ces champs sont essentiels pour relier un défaut d’interface à une exécution précise sans devoir deviner quelle configuration était active.'),
     refs('À lire : api/app.py ; contracts.py. Interface API locale : http://127.0.0.1:8000/docs.'))

page('21 · LOGICIEL', 'Concurrence et cycle de vie', 'Plus de requêtes ne veut pas dire plus de GPU',
     p('Deux utilisateurs peuvent envoyer une question presque simultanément. La <b>concurrence</b> consiste à gérer plusieurs travaux qui se chevauchent ; le <b>parallélisme</b> exécute réellement plusieurs calculs en même temps. Une fonction async n’augmente pas la VRAM disponible et ne rend pas automatiquement un modèle parallèle.'),
     fig('concurrency', 'Le verrou applicatif n’autorise qu’une inférence à la fois. Après un timeout, le service se protège d’un calcul distant qui pourrait continuer.'),
     p('Le projet utilise un verrou non bloquant : si le créneau est pris, une nouvelle requête reçoit 429. La route synchrone s’exécute dans un pool de threads. Le verrou est conservé jusqu’à la fin du travail ; la déconnexion du navigateur n’est pas une preuve que le runtime a arrêté le calcul.'),
     p('Un <b>timeout</b> borne l’attente côté client HTTP. Il ne garantit pas l’annulation immédiate côté Ollama. Après un timeout, l’API marque l’inférence indisponible jusqu’au redémarrage. Cette stratégie conservatrice évite d’empiler de nouvelles générations sur une ancienne encore active, mais demande un diagnostic opérationnel.'),
     h('Vie du processus et disponibilité'),
     p('/health/live répond à « le processus vit-il ? ». /health/ready répond à « les conditions requises sont-elles réunies ? ». Au démarrage, les dépendances et le corpus sont ouverts ; à l’arrêt, les ressources sont fermées. Le cycle de vie évite de recréer le stockage et le client réseau pour chaque question.'),
     note('Ne multiplie pas les workers avec le stockage Qdrant local partagé. Passer à plusieurs utilisateurs exige une conception explicite du stockage, de la file d’attente, de l’annulation et des limites de charge.'),
     refs('À lire : lifespan, inference_lock et readiness dans api/app.py.'))

page('22 · LOGICIEL', 'Streamlit, interface et cache', 'L’interface fait partie de la vérité du système',
     p('Streamlit exécute le script de l’application selon son modèle de réexécution lors des interactions. Le <b>state</b> de session conserve ce qui doit survivre entre les interactions d’une même session. Le <b>cache</b> réutilise un calcul déjà obtenu selon une clé. Ces mécanismes sont différents et ne doivent pas faire passer un ancien résultat pour une nouvelle réponse. [R13]'),
     p('L’interface du projet est cliente HTTP : elle n’ouvre pas directement Qdrant et ne charge pas les poids. Ainsi, les règles de preuve ne divergent pas entre une commande terminal et le navigateur. Le serveur reste responsable du traitement ; l’interface explique ce qu’il a décidé.'),
     h('Langues et sens de lecture'),
     p('L’arabe se lit principalement de droite à gauche, tandis que les URLs et les identifiants techniques restent de gauche à droite. Un conteneur RTL global appliqué sans discernement peut déplacer chiffres, parenthèses et liens. Il faut séparer les zones et isoler les fragments mixtes. L’Arabizi, lui, se lit de gauche à droite.'),
     table(['À afficher', 'Raison'], [
         ['Langue demandée et type de résultat', 'Ne pas confondre extrait source et réponse traduite.'],
         ['Citation cliquable et passage original', 'Permettre une vérification humaine.'],
         ['Statut de revue', 'Rendre visible la politique de source.'],
         ['Réponse non validée', 'Ne pas maquiller un échec du générateur.']]),
     p('Le cache doit tenir compte de la question, de la langue, du corpus et du générateur, ainsi que des versions de politique. Une revue qui expire doit rendre l’ancienne réponse inutilisable même si la question n’a pas changé. Une clé de cache n’est donc pas uniquement un mécanisme de performance : elle encode des hypothèses de validité.'),
     note('Tester l’interface signifie vérifier aussi les erreurs, les messages d’abstention, les longues URLs, l’arabe et les petits écrans. Une page qui s’affiche sans exception peut encore induire l’utilisateur en erreur.'),
     refs('À lire : apps/streamlit_app.py ; presentation.py.'))

page('23 · LOGICIEL', 'Sécurité et confidentialité', 'Les documents sont des données, pas des ordres',
     p('Une <b>prompt injection</b> tente de faire suivre au modèle des instructions présentes dans un document ou une question plutôt que les règles de l’application. Par exemple, un passage peut contenir « ignore les sources et invente une réponse ». Le système doit traiter cette phrase comme du contenu à analyser, pas comme une autorisation de changer son comportement.'),
     p('Le prompt seul n’est pas une frontière de sécurité suffisante. Le projet limite le contexte, impose un schéma, contrôle les citations et n’exécute pas les appels d’outils proposés par le modèle. Réduire les capacités accessibles diminue les conséquences d’une instruction mal suivie. Cela ne supprime pas toutes les erreurs de texte.'),
     h('Téléchargement des sources'),
     p('La <b>SSRF</b> consiste à faire contacter par un serveur une destination non prévue, par exemple un service interne. L’ingestion contrôle les hôtes autorisés, HTTPS, les destinations DNS publiques, chaque redirection et la taille du téléchargement. Les chemins locaux doivent rester dans le dossier permis. La sécurité réseau requiert plusieurs couches, pas seulement une expression régulière sur l’URL. [R14]'),
     table(['Mécanisme', 'Protège surtout contre', 'Ne garantit pas'], [
         ['TLS', 'Altération ou lecture du transport.', 'Vérité du document distant.'],
         ['Hash', 'Modification du contenu identifié.', 'Fiabilité de son auteur.'],
         ['Loopback', 'Exposition réseau directe non voulue.', 'Absence de risque sur la machine.'],
         ['Schéma JSON', 'Sortie mal structurée.', 'Exactitude du contenu.']]),
     p('L’inférence locale utilise l’adresse de boucle locale. Les fonctions cloud d’Ollama ont été désactivées sur cette machine et l’adaptateur refuse des modèles distants. Cela ne signifie pas « aucun réseau à aucun moment » : installer des dépendances, télécharger un modèle ou collecter une source officielle utilise toujours le réseau. [R15]'),
     note('Ne committe ni tokens d’accès, ni .env privé, ni audios personnels. Les journaux du pipeline enregistrent les identifiants et durées plutôt que le texte des questions. Vérifie séparément les logs du runtime et du système avant toute promesse globale de confidentialité.'))

page('24 · LOGICIEL', 'Reproductibilité et organisation du code', 'Pouvoir expliquer ce qui a changé',
     p('<b>Git</b> conserve l’historique des fichiers ; <b>GitHub</b> héberge un dépôt et des outils de collaboration. Un fichier modifié localement n’est pas automatiquement committé ni publié. Un commit décrit un état suivi ; une branche permet un historique de travail distinct. Le fichier .gitignore exclut des fichiers du suivi normal, mais ne retire pas un secret déjà enregistré dans l’historique.'),
     p('Un <b>environnement virtuel</b> isole les paquets Python du projet. pyproject.toml déclare les dépendances et les options ; uv.lock fige leur résolution. Une installation verrouillée réduit les variations involontaires, sans garantir à elle seule des résultats numériques identiques entre CPU, GPU et pilotes. [R16, R17]'),
     h('Les contrats rendent les modules remplaçables'),
     p('Un adaptateur traduit l’interface d’un service vers celle attendue par le pipeline. L’injection de dépendances fournit un embedder ou un générateur à la construction au lieu de les figer partout. Un test peut donc utiliser un faux générateur prévisible. Ce faux vérifie le logiciel autour du modèle ; il ne mesure pas l’intelligence du vrai modèle.'),
     table(['Famille de fichiers', 'Rôle à retrouver dans le code'], [
         ['config.py / contracts.py', 'Configuration et formes des données.'],
         ['ingestion/ / source_policy.py', 'Préparer et autoriser les sources.'],
         ['retrieval/ / pipeline.py', 'Rechercher puis orchestrer le traitement.'],
         ['generation/ / language.py', 'Produire, contrôler et orienter la langue.'],
         ['api/ / apps/ / tests/', 'Exposer, afficher et vérifier.']]),
     p('Les paramètres d’expérience à conserver comprennent le commit, les dépendances, les hashes de modèles, la release documentaire, les politiques et les réglages de décodage. Sans cette fiche, une amélioration apparente peut venir d’un changement involontaire de données.'),
     note('Graphify a servi à naviguer dans les relations du code. Son graphe ancien a été recoupé avec les fichiers actuels. Ce graphe de code n’est ni la base Qdrant de l’assistant ni une implémentation de GraphRAG dans le produit.'))

page('25 · ÉVALUATION', 'Tests logiciels et tests de qualité', 'Deux types de preuves complémentaires',
     fig('tests', 'Chaque étage répond à une question différente. Un grand nombre de tests unitaires ne remplace pas une évaluation linguistique.'),
     p('Un <b>test unitaire</b> cible une règle isolée : refuser un numéro inventé, identifier une langue ou calculer un hash. Un <b>test d’intégration</b> vérifie une liaison réelle entre composants, par exemple ingestion et stockage local. Un <b>test de contrat</b> utilise une réponse simulée pour vérifier la structure attendue d’une API externe.'),
     p('Un <b>test de bout en bout</b> parcourt un scénario utilisateur. Un test navigateur peut inspecter citations, directions de texte et messages d’erreur. Un <b>smoke test</b> est un petit contrôle de bon fonctionnement ; il n’est pas conçu pour estimer avec précision une performance générale.'),
     h('Ce que les outils apportent'),
     table(['Outil', 'Utilité', 'Ce qu’il ne mesure pas'], [
         ['pytest', 'Exécuter les assertions des tests.', 'Fidélité linguistique sans cas annotés.'],
         ['Ruff', 'Détecter certains défauts et uniformiser le code.', 'Exactitude métier ou absence de bugs.'],
         ['pip check', 'Détecter des incompatibilités déclarées.', 'Compatibilité pratique de tous les runtimes.'],
         ['CI', 'Rejouer des contrôles sur des environnements définis.', 'Qualité réelle sur tout matériel utilisateur.']]),
     p('Le précédent passage de la suite a donné 199 tests hors ligne réussis. Le cours cite ce résultat enregistré ; il ne prétend pas avoir réexécuté les tests de l’application pour sa rédaction. Ces 199 cas couvrent des comportements de logiciel et non 199 réponses Darija jugées par des locuteurs.'),
     note('Pour chaque bug, ajoute un test qui échoue avant correction et réussit après. Mais garde aussi des cas non utilisés pendant l’optimisation : sinon tu apprends seulement à passer ta collection de tests connue.'),
     refs('À lire : tests/ ; .github/ ; README, section de validation locale.'))

page('26 · ÉVALUATION', 'WER, CER et protocole ASR', 'Mesurer des erreurs de transcription comparables',
     p('La <b>distance d’édition</b> compte le nombre minimal de substitutions, suppressions et insertions pour transformer une transcription en référence. Le WER travaille sur des mots ; le CER travaille sur des caractères. La convention de découpage fait donc partie de la métrique.'),
     code('WER = (S + D + I) / N\nS : substitutions ; D : suppressions ; I : insertions\nN : nombre de mots de la référence'),
     table(['Exemple fictif', 'Séquence'], [
         ['Référence', 'je veux une carte'],
         ['Hypothèse', 'je veux carte maintenant'],
         ['Alignement', 'Suppression de « une », insertion de « maintenant ».'],
         ['WER', '(0 + 1 + 1) / 4 = 0,50, soit 50 %.']]),
     p('Le WER peut dépasser 100 % quand les insertions sont nombreuses. Une référence vide exige une convention spéciale ; ne divise pas par zéro. Pour un corpus, additionner S, D, I et N avant la division pondère par le nombre de mots. La moyenne des WER par clip répond à une autre question : elle donne autant de poids à un clip court qu’à un clip long.'),
     h('Particularités de la Darija'),
     p('Plusieurs graphies peuvent exprimer le même mot. Il est utile de conserver une mesure brute et une mesure normalisée, avec des règles publiées. Si la normalisation supprime une négation ou transforme trop de mots en une même forme, elle rend le score artificiellement bon. CER et WER sont complémentaires, pas interchangeables.'),
     p('Le protocole doit séparer locuteurs, domaines et conditions audio ; stratifier les résultats par langue mêlée, bruit, durée et accent ; puis examiner les erreurs critiques comme chiffres et négations. Un petit progrès de WER moyen peut coexister avec une dégradation importante des montants reconnus.'),
     note('Évaluer Lmaana contre Whisper exige les mêmes références et le même calcul. Le module evaluation/ est encore réservé : ces métriques constituent le protocole à implémenter, pas un benchmark ASR déjà exécuté dans le projet.'),
     refs('Définition usuelle des erreurs d’édition ; protocole propre à la release Lmaana : [R6].'))

page('27 · ÉVALUATION', 'Recall@K, Hit@K et MRR', 'Mesurer la recherche avant de juger la rédaction',
     p('Un jeu de recherche annoté associe chaque question aux passages réellement pertinents. Il faut décider si une preuve partielle compte, comment traiter les doublons et si une source périmée reste pertinente pour le sujet mais non utilisable pour répondre. Sans ces conventions, deux scores identiques peuvent décrire des évaluations différentes.'),
     code('Recall@K = pertinents retrouvés dans les K premiers / pertinents totaux\nHit@K = 1 si au moins un pertinent est dans le top-K, sinon 0\nRR = 1 / rang du premier pertinent ; RR = 0 si aucun\nMRR = moyenne des RR sur les questions'),
     table(['Question fictive', 'Pertinents', 'Top-3', 'Recall@3', 'RR'], [
         ['Q1', 'A, C', 'B, A, C', '2/2 = 1', '1/2'],
         ['Q2', 'D', 'E, F, G', '0/1 = 0', '0'],
         ['Q3', 'H, I', 'H, X, Y', '1/2', '1']]),
     p('Sur ces trois questions, le rappel moyen vaut (1 + 0 + 0,5)/3 = 0,5. Hit@3 vaut 2/3. MRR, ici limité aux résultats présentés, vaut (0,5 + 0 + 1)/3 = 0,5. Le premier cas possède un rappel parfait à 3 mais pas un premier résultat pertinent : les métriques éclairent des aspects différents.'),
     h('Faire un test utile au RAG'),
     p('Mesure la recherche avec la question correcte, puis avec sa transcription ASR. La différence révèle l’impact de la parole. Compare lexical, alias, puis embeddings sémantiques sur le même jeu gelé. Inclue des questions sans réponse afin d’étudier les faux rapprochements, pas seulement la capacité à retrouver quelque chose.'),
     p('Un reranker ne peut pas récupérer une preuve absente de sa liste d’entrée. Avant d’optimiser le classement final, assure-toi que les candidats contiennent les bons passages. Inversement, un rappel élevé avec cinquante passages ne garantit pas un bon résultat si seulement quatre peuvent entrer dans le contexte.'),
     note('Le rappel porte sur la récupération d’éléments pertinents, pas sur la vérité des phrases produites par Qwen. Évalue séparément la recherche et la génération.'),
     refs('Référence sur l’évaluation de recherche : [R18].'))

page('28 · ÉVALUATION', 'Fidélité, langue et abstention', 'Lire correctement les résultats du projet',
     p('La <b>faithfulness</b>, ou fidélité aux preuves, demande si les affirmations sont soutenues par les passages fournis. La <b>correctness</b> demande si elles sont correctes par rapport à une référence fiable. La <b>complétude</b> demande si les éléments indispensables sont présents. La <b>qualité linguistique</b> demande si le texte respecte la langue, l’usage et la lisibilité attendus.'),
     table(['Critère', 'Exemple de défaut'], [
         ['Fidélité', 'Le modèle ajoute un montant absent du contexte.'],
         ['Correction', 'La source utilisée est obsolète.'],
         ['Complétude', 'Une exception importante disparaît.'],
         ['Langue', 'Arabe standard ou mélange incohérent au lieu de Darija.'],
         ['Abstention', 'Le système refuse une question pourtant bien documentée.']]),
     h('Le petit essai réel enregistré'),
     p('Dans outputs/Lmaana-Ollama-Checks.json, 14 scénarios ont été testés. Dix ont eu l’issue attendue ; quatre tentatives de réponse arabe/Darija/Arabizi ont été rejetées après validation, y compris une demande de Darija depuis une question française. Les réponses documentaires française et anglaise ont été générées avec citations. Les autres succès incluent des abstentions et une clarification attendues.'),
     p('Il serait faux d’annoncer « 71 % de précision en Darija » ou « dix réponses factuelles réussies ». Ce petit ensemble n’est ni représentatif ni annoté comme un benchmark complet. Le rejet des quatre sorties est une protection utile, mais pas une réussite de génération. Il peut aussi contenir des faux rejets à examiner humainement.'),
     h('Construire une vraie grille'),
     p('Décompose les réponses en affirmations ; associe chaque affirmation à ses preuves ; annote support complet, partiel, contradictoire ou absent ; examine les conditions et la langue avec des locuteurs compétents. Fais juger un sous-ensemble par deux personnes pour mesurer les désaccords. Conserve aussi le taux de réponse et les abstentions erronées.'),
     note('Une meilleure fidélité obtenue en refusant presque toutes les questions peut rendre le produit inutilisable. Rapporte simultanément la qualité des réponses acceptées et la couverture : combien de questions reçoivent effectivement une réponse ?'))

page('29 · ÉVALUATION', 'Latence et performance', 'Mesurer le chemin entier, pas seulement le modèle',
     p('La <b>latence</b> est la durée observée pour une requête. Le <b>débit</b> mesure combien de requêtes ou tokens sont traités par unité de temps. Une application peut générer vite après chargement, mais sembler lente au premier appel à cause de la lecture des poids et de l’initialisation GPU.'),
     code('latence totale = préparation + embedding + recherche\n                 + génération + contrôles + transport + affichage'),
     p('Le pipeline mesure normalisation, embedding, retrieval, generation et total. Les limites de chronométrage sont celles du code : le transport navigateur et tout coût extérieur à la méthode ne figurent pas nécessairement dans ce total. Le champ generation inclut les tentatives et certains traitements finaux ; il ne faut pas le présenter comme du temps GPU pur.'),
     h('Pourquoi les percentiles ?'),
     p('La médiane P50 décrit une requête typique ; P95 décrit un seuil sous lequel se situent environ 95 % des observations, selon la convention de calcul. Une moyenne seule masque les requêtes très lentes. Les percentiles issus de quelques essais restent instables : utilise assez de mesures et publie la taille de l’échantillon.'),
     table(['Expérience', 'Ce qu’il faut isoler'], [
         ['Démarrage à froid', 'Modèle non chargé.'],
         ['Exécution chaude', 'Modèle déjà chargé, même configuration.'],
         ['Cache de réponse', 'Pas de nouvelle génération pour le même résultat.'],
         ['Requête avec retry', 'Coût des sorties invalides inclus.'],
         ['Charge concurrente', 'Attente, 429 et mémoire maximale.']]),
     p('Le temps jusqu’au premier token est utile pour un affichage en streaming, mais ce projet attend actuellement la réponse complète validée. Réduire le temps total peut passer par moins de contexte bruité, un budget de sortie adapté ou une meilleure précision de recherche, plutôt que simplement changer de GPU.'),
     note('Ne compare pas un cas en cache et un cas fraîchement généré. Enregistre modèle, contexte, longueur de question, nombre de preuves, longueur de sortie, état chaud/froid et matériel. Les chiffres deviennent interprétables seulement avec ces conditions.'))

page('30 · PRATIQUE', 'Explorer le projet sans le modifier', 'Un premier laboratoire en lecture seule',
     p('Ce laboratoire suit la requête réelle. Lance les commandes depuis la racine du dépôt avec l’environnement du projet. Les commandes ci-dessous n’ingèrent rien et ne téléchargent aucun modèle. Le POST final demande seulement une réponse au service local déjà lancé ; il peut consommer du temps de calcul.'),
     code('.\\.venv\\Scripts\\python.exe -m lmaana_assistant.cli doctor\nInvoke-RestMethod http://127.0.0.1:8000/health/live\nInvoke-RestMethod http://127.0.0.1:8000/health/ready'),
     p('Compare live et ready. Note les backends, la release et le nombre de sources approuvées. Puis lis Settings : distingue les valeurs par défaut du dépôt et celles chargées depuis .env. Le dépôt peut démarrer en mode extrait tandis que cette machine utilise explicitement Ollama.'),
     code('$requeteCours = @{\n  question = "Quels documents pour devenir auto-entrepreneur ?"\n  response_language = "fr"\n} | ConvertTo-Json\nInvoke-RestMethod -Method Post `\n  -Uri http://127.0.0.1:8000/v1/answers `\n  -ContentType "application/json; charset=utf-8" `\n  -Body ([Text.Encoding]::UTF8.GetBytes($requeteCours))'),
     h('Ce qu’il faut observer'),
     p('Suis original, normalized et search_text. Vérifie detected, reply et method. Lis outcome et generation_failed avant de lire answer. Compare les citations avec les passages originaux, puis examine les timings. Une réponse différente de l’ancien rapport peut être normale si la revue a expiré, le modèle a changé ou le décodage a produit une autre sortie.'),
     p('Enfin, ouvre pipeline.py et retrouve ces étapes dans le même ordre. Lis ensuite le test qui correspond à l’erreur qui t’intéresse. Cette lecture guidée est plus efficace que d’examiner tous les fichiers sans question précise.'),
     note('Si l’API est arrêtée, une erreur de connexion ne dit rien sur la qualité du modèle. Si elle répond 503, commence par son motif de readiness. N’essaie pas une réingestion avec l’API ouverte sur le même Qdrant local.'))

page('31 · PRATIQUE', 'Les prochaines améliorations', 'Transformer les limites observées en expériences',
     p('La prochaine étape doit répondre à un défaut mesuré. Ajouter un framework, un modèle plus gros ou une base différente n’est pas une amélioration en soi. Définis d’abord le critère de réussite et garde une baseline pour savoir si le changement a servi.'),
     table(['Chantier', 'Expérience proposée', 'Preuve de réussite'], [
         ['Darija générée', 'Comparer prompts/modèles sur questions gelées.', 'Fidélité et langue jugées, couverture conservée.'],
         ['ASR Lmaana 2.4', 'Exécuter un clip puis un test tenu à part.', 'Transcriptions, WER/CER, mémoire et latence.'],
         ['Recherche sémantique', 'Comparer lexical et Qwen embeddings.', 'Recall/MRR et abstentions sur le même jeu.'],
         ['Corpus', 'Ajouter une source revue avec cas négatifs.', 'Extraction fidèle et réponses traçables.'],
         ['Usage vocal', 'Ajouter correction de transcription.', 'Erreurs reconnues et récupérables par l’utilisateur.']]),
     h('Une expérience, une variable principale'),
     p('Pour étudier l’embedding, conserve autant que possible les questions, les sources et le générateur. Pour étudier la génération, fournis exactement les mêmes preuves. Pour étudier l’ASR, évalue d’abord les transcriptions sans RAG. Une ablation enlève ou remplace un mécanisme, par exemple les alias, afin de mesurer son apport.'),
     p('Les essais doivent prévoir un lot de développement pour ajuster les règles, un lot de validation pour choisir une configuration et un test final tenu à part. Ne réutilise pas continuellement le test final comme tableau de bord d’optimisation : il finirait par influencer toutes tes décisions.'),
     h('Ce que LangChain ou LangGraph changeraient'),
     p('Ces outils pourraient structurer des chaînes ou des graphes d’états, notamment avec boucles et outils. Le pipeline actuel est du Python explicite. Introduire un framework ne fournit pas automatiquement de meilleures sources, une meilleure Darija ou des métriques. La complexité ajoutée doit avoir un bénéfice concret de maintenance ou de comportement.'),
     note('La cible est un assistant utile, pas une démonstration de noms de bibliothèques. Tu dois pouvoir expliquer chaque ajout par une erreur qu’il corrige et une mesure qui le confirme.'))
