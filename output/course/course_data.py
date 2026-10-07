"""Contenu original du cours, commun aux éditions PDF et HTML."""

PAGES = []

def page(part, title, subtitle, *blocks):
    PAGES.append(dict(part=part, title=title, subtitle=subtitle, blocks=list(blocks)))

def p(text): return ('p', text)
def h(text): return ('h', text)
def note(text): return ('note', text)
def code(text): return ('code', text)
def table(headers, rows): return ('table', headers, rows)
def fig(name, caption): return ('fig', name, caption)
def ar(text, translation): return ('ar', text, translation)
def refs(text): return ('refs', text)

page('COURS ILLUSTRÉ', 'Comprendre et construire\nLmaana Assistant',
     'De la voix en Darija à une réponse documentée',
     p('Un cours de bout en bout pour comprendre les modèles, les données, le code et les décisions de qualité derrière ton projet. Chaque notion est reliée à un problème concret : reconnaître une phrase, retrouver une preuve, conserver une condition ou expliquer un refus de réponse.'),
     fig('cover', 'Trois capacités distinctes : reconnaître, retrouver et répondre. Aucune ne garantit à elle seule les deux autres.'),
     h('Ce que tu sauras faire'),
     p('Expliquer le rôle de Lmaana 2.4, Qwen3, Ollama, Qdrant, FastAPI et Streamlit ; calculer des métriques simples ; diagnostiquer une mauvaise réponse ; lire les principaux modules ; concevoir des tests qui séparent un logiciel fonctionnel d’un assistant réellement fiable.'),
     note('Niveau : débutant motivé à intermédiaire. Prérequis : variables, fonctions et listes Python ; notions élémentaires de terminal. Les formules sont expliquées. Les exemples chiffrés fictifs sont signalés.'),
     p('<b>Photographie du projet : 3 octobre 2026.</b> Le chemin texte fonctionne avec un modèle local. L’entrée vocale Lmaana 2.4 et la synthèse vocale ne sont pas intégrées. La qualité de génération en Darija reste à améliorer. Ce cours n’est pas un guide administratif ni une certification de réponses.'),
     refs('Établi à partir du code local, du README, des rapports de vérification et des références primaires listées en fin de cours.'))

page('MODE D’EMPLOI', 'Comment étudier ce cours', 'Comprendre les responsabilités avant de mémoriser les noms',
     p('Lis d’abord les chapitres 01 à 05 pour construire une carte mentale. Poursuis avec la voix et les langues, puis la préparation des sources et la recherche. Les derniers chapitres transforment ces composants en service testable. Ne cherche pas à retenir tous les paramètres dès la première lecture : demande-toi toujours quelle erreur chaque mécanisme évite.'),
     table(['Parcours', 'Chapitres', 'Question directrice'], [
         ['Fondations', '01 - 05', 'Quel composant fait quel travail ?'],
         ['Voix et langues', '06 - 09', 'Comment conserver ce que l’utilisateur veut dire ?'],
         ['Connaissance et recherche', '10 - 16', 'Quelle preuve peut réellement soutenir la réponse ?'],
         ['Réponse et logiciel', '17 - 24', 'Comment servir et contrôler le résultat ?'],
         ['Évaluation et pratique', '25 - 31', 'Comment savoir si une modification améliore le projet ?'],
         ['Révision', 'Annexes', 'Exercices, corrigés, glossaire et références.']]),
     h('Trois statuts à ne jamais mélanger'),
     p('<b>Actif</b> désigne un chemin exécuté sur cette machine. <b>Adaptateur présent</b> indique que du code et des tests de contrat existent, sans démonstration complète avec le vrai modèle. <b>Prévu</b> décrit la cible future. Un dessin d’architecture peut contenir ces trois statuts ; il doit les rendre visibles.'),
     h('La méthode de travail'),
     p('Après chaque chapitre, explique la notion à voix haute avec un exemple différent. Puis ouvre le fichier indiqué et retrouve la responsabilité correspondante. Enfin, imagine un contre-exemple : une langue ambiguë, un PDF tronqué, un modèle indisponible. Si tu sais prédire la réaction du système, tu as compris davantage qu’une simple définition.'),
     note('Les durées d’apprentissage dépendent de ton expérience. Travaille par petites séances : une notion, un calcul, un test. Les exercices finaux utilisent des données fictives pour éviter de confondre apprentissage technique et conseil administratif.'))

page('01 · FONDATIONS', 'L’architecture de bout en bout', 'Le produit est un système, pas un modèle unique',
     p('Lmaana Assistant assemble plusieurs programmes spécialisés. Streamlit reçoit la question et affiche le résultat. FastAPI vérifie la requête et appelle un pipeline Python. Le pipeline prépare le texte, retrouve des passages dans Qdrant, filtre les sources, puis demande à Qwen de rédiger via Ollama. Des contrôles décident si cette rédaction peut être affichée.'),
     fig('architecture', 'Le trajet actif est textuel. La voix est une extension prévue, pas une fonctionnalité déjà disponible.'),
     p('La <b>séparation des responsabilités</b> permet de remplacer un composant sans réécrire tout le produit. Une mauvaise transcription relève de l’ASR ; un document introuvable relève de la recherche ; un chiffre inventé relève de la génération ou de ses contrôles. Sans cette séparation, chaque échec devient un vague « problème d’IA ».'),
     p('L’<b>ingestion</b> prépare les documents avant les questions. L’<b>inférence</b> utilise les modèles lorsqu’une requête arrive. Ces deux chemins n’ont pas le même rythme : on n’extrait pas tous les PDF à chaque question, et on n’entraîne pas Qwen lorsqu’un utilisateur clique sur Envoyer.'),
     note('Actuellement : recherche lexicale hachée + Qdrant local + Qwen3-4B via Ollama. Qwen3-Embedding a un adaptateur optionnel non validé en exécution réelle ici. LangChain, LangGraph, FAISS et React étaient des options évoquées, pas la pile implémentée.'),
     refs('À lire : src/lmaana_assistant/pipeline.py ; api/app.py ; apps/streamlit_app.py.'))

page('02 · FONDATIONS', 'Ce qu’est un modèle de langage', 'Des poids appris et une prédiction de séquence',
     p('Un modèle est une fonction paramétrée. Ses <b>poids</b> sont des nombres ajustés pendant l’entraînement. Dans un modèle de langage causal, la fonction attribue des probabilités aux tokens suivants à partir des tokens déjà présents. Un token peut correspondre à un mot, une partie de mot, un signe ou une portion d’octets selon le tokenizer.'),
     code('P(réponse | contexte) = produit des P(token_t | contexte, tokens précédents)'),
     p('Cette règle très simple, répétée sur des réseaux et des données considérables, permet des comportements complexes : reformulation, traduction, raisonnement partiel, rédaction. Elle ne donne pas au modèle une base de faits garantie. Une continuation plausible peut être fausse, notamment si les données sont anciennes ou si une condition est perdue.'),
     h('L’idée du Transformer'),
     p('L’attention combine des représentations de positions différentes pour construire le contexte utile à chaque token. Dans un décodeur causal, une position ne doit pas voir les tokens futurs. Des couches successives transforment ces représentations ; la sortie finale sert à choisir le prochain token. Attention n’est donc pas synonyme de vérification des sources. [R2]'),
     p('Dans « les banques partenaires », le mot « partenaires » modifie une restriction essentielle. Un système peut écrire une phrase grammaticalement parfaite en supprimant cette restriction. Voilà pourquoi la qualité linguistique, la compréhension des conditions et la vérité factuelle doivent être évaluées séparément.'),
     h('Trois notions pratiques'),
     table(['Notion', 'Signification'], [
         ['Préentraînement', 'Apprendre de larges régularités de langue et de données.'],
         ['Post-entraînement', 'Adapter le comportement, par exemple au suivi d’instructions.'],
         ['Inférence', 'Utiliser les poids existants pour produire une sortie.']]),
     note('Qwen3-4B est le modèle génératif retenu. Le nom « 4B » indique un ordre de grandeur de quatre milliards de paramètres, pas quatre milliards de documents ni une taille de mémoire fixe. [R3]'))

page('03 · FONDATIONS', 'Ollama, Qwen et Lmaana', 'Un moteur, un modèle de texte et un modèle de parole',
     fig('roles', 'Ces noms appartiennent à des niveaux différents. Changer de moteur ne garantit pas une meilleure compétence linguistique.'),
     p('<b>Ollama</b> est le service d’exécution : il gère un modèle local et expose une API. <b>Qwen3-4B</b> représente les poids utilisés pour générer du texte. <b>Lmaana 2.4</b> est la cible de reconnaissance vocale, avec un autre runtime. Ollama n’est donc ni le cerveau complet du RAG, ni la transcription Lmaana, ni le moteur de recherche documentaire.'),
     p('Le service local est interrogé sur le port 11434. L’application utilise /api/tags et /api/show pour contrôler la présence et les métadonnées du modèle, puis /api/chat pour la génération. Le paramètre de format permet de demander une sortie structurée. Ces appels ne remplacent pas les contrôles métier du projet. [R4, R5]'),
     h('Nom, version et identité'),
     p('Un <b>tag</b> comme qwen3:4b est pratique mais peut désigner un artefact différent après une mise à jour. Un <b>digest</b> identifie précisément un manifeste. Épingler ce digest évite un changement silencieux. Un GGUF importé et un modèle du registre peuvent différer par les poids, les métadonnées ou le template de conversation : leurs résultats doivent être comparés, pas supposés identiques.'),
     note('Dans ce projet, Ollama a effectivement exécuté Qwen sur GPU. L’adaptateur llama.cpp autonome existe, mais son exécution locale reste non validée. L’installation d’un runtime ne prouve ni la fidélité de ses réponses ni sa maîtrise de la Darija.'),
     refs('À lire : generation/ollama.py ; generation/adapters.py ; configs/Modelfile.qwen3.'))

page('04 · FONDATIONS', 'Tokens, contexte et décodage', 'La place disponible et la manière de choisir les mots',
     p('La <b>fenêtre de contexte</b> limite la séquence manipulée pendant une génération. Le prompt comprend les instructions, la question, les passages et le formatage de conversation. Il faut aussi réserver de la place à la réponse. Une limite exprimée en tokens n’est pas un nombre de mots constant : le français, l’arabe et l’Arabizi ne se découpent pas nécessairement de la même façon.'),
     code('instructions + question + preuves + template + sortie <= fenêtre de contexte'),
     p('Le profil local configuré réserve 8 192 tokens de contexte et jusqu’à 1 024 tokens de sortie. L’adaptateur Ollama estime prudemment la taille à partir des octets UTF-8 avec une réserve de template. Ce n’est pas un comptage exact. Si nécessaire, il retire des passages entiers de moindre rang ; il ne doit pas tronquer silencieusement une condition au milieu d’une phrase.'),
     h('Les paramètres de génération'),
     table(['Paramètre', 'Intuition', 'Attention'], [
         ['Température', 'Modifie la concentration des probabilités.', 'Plus faible ne signifie pas plus vrai.'],
         ['top-k', 'Garde les k candidats les plus probables.', 'Un k trop petit réduit les possibilités.'],
         ['top-p', 'Garde un ensemble couvrant une masse de probabilité.', 'Il dépend de la distribution courante.'],
         ['Limite de sortie', 'Arrête une réponse devenue trop longue.', 'Une sortie coupée peut être invalide.']]),
     p('Les réglages actifs sont température 0,7, top-p 0,8 et top-k 20. Ils sont des choix de configuration, pas des garanties. Le projet demande think=false et stream=false : pas de raisonnement séparé demandé, et récupération de la réponse complète avant validation. Un mode de réflexion activé n’a pas résolu les défauts linguistiques observés dans l’essai diagnostique local.'),
     note('Le modèle ne reçoit pas automatiquement tout l’historique de l’interface. Une conversation mémorisée exige une politique explicite : quelles anciennes questions transmettre, pendant combien de temps, et avec quel budget ? Le chemin actuel traite chaque question avec ses preuves.'),
     refs('À lire : generation/ollama.py. Références : [R3, R4].'))

page('05 · FONDATIONS', 'RAM, VRAM et quantification', 'Pourquoi « le fichier tient » ne suffit pas',
     p('La <b>RAM</b> est la mémoire principale du système ; la <b>VRAM</b> est la mémoire du GPU. La machine de ce projet possède environ 32 Go de RAM et un GPU RTX 5070 Laptop avec environ 8 Go de VRAM. Ce sont deux budgets distincts. Un modèle peut tenir dans la RAM sans tenir entièrement sur le GPU.'),
     fig('memory', 'Budget conceptuel de mémoire, non à l’échelle. Les activations et le cache s’ajoutent toujours aux poids.'),
     p('La <b>quantification</b> représente les poids avec moins de bits. À titre d’ordre de grandeur : quatre milliards de paramètres à 4 bits donnent environ 2 milliards d’octets bruts. Il faut ajouter échelles, métadonnées et parties stockées différemment. Le GGUF Q4_K_M officiel téléchargé occupe environ 2,5 Go ; cela ne prédit pas à lui seul la mémoire totale d’exécution. [R3]'),
     code('mémoire des poids, approximation = paramètres × bits / 8'),
     p('Le <b>cache KV</b> conserve des représentations utiles à l’attention pour les tokens déjà traités. À architecture et précision fixées, il augmente avec la longueur du contexte et le nombre de séquences concurrentes. S’ajoutent les activations temporaires, les buffers du runtime et la mémoire utilisée par l’affichage.'),
     p('L’<b>offload</b> répartit le calcul entre CPU et GPU. Il peut permettre un modèle plus gros, mais les transferts et le CPU peuvent ralentir la réponse. La bonne stratégie est de mesurer chaque modèle seul, puis leur coexistence. Charger simultanément ASR et LLM n’est pas une promesse raisonnable sur la seule base des tailles de fichiers.'),
     note('Le relevé précédent d’Ollama indiquait environ 3,9 Go chargés pour Qwen et un placement 100 % GPU. C’est une observation datée pour une configuration, pas une garantie de consommation sous toute charge.'))

page('06 · VOIX ET LANGUES', 'De l’onde sonore au texte', 'ASR, prétraitement audio et erreurs en cascade',
     p('L’<b>ASR</b>, automatic speech recognition, transforme un signal audio en transcription. Le microphone produit une suite d’échantillons. La fréquence d’échantillonnage indique combien de mesures sont stockées par seconde ; la profondeur de quantification indique leur précision. Des canaux stéréo peuvent devoir être convertis en mono selon le modèle.'),
     h('Préparer sans déformer'),
     p('Le prétraitement doit respecter la recette du modèle : fréquence attendue, amplitude, format et segmentation. Réétiqueter un fichier 48 kHz comme 16 kHz sans rééchantillonnage change sa durée apparente et sa hauteur. Un volume saturé détruit de l’information ; un filtre de bruit trop agressif peut effacer des consonnes utiles.'),
     p('La <b>VAD</b>, détection d’activité vocale, repère les segments susceptibles de contenir de la parole. Elle peut limiter les silences et les coûts, mais couper une syllabe si ses seuils sont mauvais. La <b>diarisation</b> cherche qui parle ; elle n’est pas la reconnaissance du contenu. La <b>TTS</b> synthétise une voix à partir du texte, donc réalise une autre tâche encore.'),
     table(['Étape', 'Erreur possible', 'Conséquence'], [
         ['Segmentation', 'Fin de mot coupée', 'Transcription incomplète.'],
         ['Transcription', 'Négation omise', 'Intention inversée.'],
         ['Recherche', 'Mauvais terme administratif', 'Mauvais passage sélectionné.'],
         ['Réponse', 'Reformulation assurée', 'Erreur d’origine masquée.']]),
     p('Un assistant vocal fiable laisse l’utilisateur relire et corriger la transcription. Il conserve aussi une séparation entre le texte brut reconnu et sa forme normalisée pour la recherche. Ainsi, on peut retrouver à quel étage une erreur a été introduite. L’audio personnel ne doit pas être conservé par défaut sans besoin explicite.'),
     note('Dans Lmaana Assistant, ce chapitre décrit l’extension à construire. Il n’existe pas encore de bouton microphone relié à une transcription validée. Une variable de configuration ASR n’est pas une implémentation ASR.'),
     refs('À lire : docs/lmaana-2.4-integration.md ; asr/__init__.py.'))

page('07 · VOIX ET LANGUES', 'Lmaana 2.4 et le principe CTC', 'Comprendre le décodage sans confondre les runtimes',
     p('La fiche de Lmaana 2.4 décrit un modèle Darija fondé sur OmniASR CTC 3B v2, avec fairseq2 et décodage CTC glouton. Le checkpoint est natif, pas un paquet Transformers chargeable tel quel avec from_pretrained(). L’accès aux fichiers nécessite l’acceptation des conditions sur Hugging Face. Ces contraintes doivent guider l’intégration. [R6]'),
     fig('ctc', 'Exemple pédagogique CTC : fusionner d’abord les répétitions consécutives, puis supprimer les blancs. Un blanc entre deux a permet de conserver deux a.'),
     p('CTC traite le décalage entre beaucoup de pas temporels audio et une transcription plus courte. Plusieurs alignements peuvent correspondre au même texte. À l’entraînement, l’objectif additionne les probabilités des alignements compatibles. Au décodage glouton, on choisit le symbole le plus probable à chaque pas, puis on applique la réduction. Cette voie n’est pas forcément la transcription globalement la plus probable. [R7]'),
     h('Pourquoi comparer à Whisper ?'),
     p('Whisper-large-v3-turbo sert de <b>baseline</b> prévue : un point de comparaison clairement défini, pas un remplaçant silencieux. Sa fiche présente une variante accélérée du modèle large-v3. Pour comparer utilement, les deux systèmes doivent transcrire exactement les mêmes audios de test, avec un protocole de normalisation et de calcul identique. [R8]'),
     p('La séparation entraînement/validation/test est cruciale. Si des enregistrements MoulSot ont servi à adapter Lmaana, les réutiliser pour annoncer sa qualité sur de nouveaux locuteurs serait trompeur. Vérifie les splits, les locuteurs et les doublons acoustiques. Les chiffres d’une fiche de modèle restent ceux de son protocole, pas ceux de notre application.'),
     note('Cible imposée dans ce dépôt : Lmaana/lmaana-2.4. Pas sailu4/lmaana-2.1. Les poids ASR n’ont pas été exécutés dans le projet ; le support natif Windows du runtime reste à vérifier.'))

page('08 · VOIX ET LANGUES', 'Détecter la langue et choisir la réponse', 'Reconnaître une langue ne prouve pas qu’on sait la parler',
     p('La <b>language identification</b> estime la langue de la question. Le routage décide ensuite la langue souhaitée pour la réponse. Ces décisions sont indépendantes de la langue du document : une question en Darija peut être soutenue par un passage français. La traduction doit préserver les conditions, pas seulement le sujet.'),
     fig('language', 'Le choix explicite de l’utilisateur prévaut sur la détection. La génération et ses contrôles restent nécessaires après le routage.'),
     table(['Code du projet', 'Sens'], [['fr', 'Français'], ['ar', 'Arabe standard'], ['ary', 'Darija en caractères arabes'], ['ary-Latn', 'Darija latine / Arabizi'], ['en', 'Anglais'], ['und', 'Langue indéterminée']]),
     p('Les règles actuelles utilisent des indices de caractères et de vocabulaire. Les mots étrangers, URL et noms de produits peuvent perturber une détection naïve ; le code cherche à éviter certains de ces pièges. Le <b>code-switching</b> est le mélange de langues dans une même question. Il n’impose pas nécessairement de répondre dans un mélange identique.'),
     p('La confiance low/medium/high du projet est qualitative. Ce n’est pas une probabilité calibrée. Pour une entrée très courte comme « 123 », mieux vaut demander une langue que fabriquer une certitude. Le sélecteur de réponse est une échappatoire importante quand les règles se trompent.'),
     note('Le mode extrait conserve le texte de la source et signale answer_language="source". Un extrait français n’est jamais présenté comme une réponse Darija réussie. Dans le mode LLM, une sortie rejetée après validation reste un échec même si la langue d’entrée a été correctement détectée.'),
     refs('À lire : language.py ; contracts.py ; tests/test_language.py.'))

page('09 · VOIX ET LANGUES', 'Normaliser sans perdre le sens', 'Trois représentations de la même question',
     ar('شنو الوثائق لي خاصني؟', 'Exemple Darija : « Quels documents me faut-il ? »'),
     p('La forme <b>originale</b> est ce que l’utilisateur a réellement écrit. La forme <b>normalisée</b> retire certains écarts superficiels. La forme de <b>recherche</b> ajoute éventuellement des alias utiles pour retrouver les documents. Garder ces trois champs facilite le diagnostic et évite de confondre la phrase de l’utilisateur avec une reformulation produite par des règles.'),
     p('Le tatweel est un caractère d’allongement visuel en arabe. Il peut être retiré pour la recherche sans changer le mot attendu. Les espaces PDF irréguliers peuvent être harmonisés. En revanche, effacer une négation, confondre des chiffres ou supprimer systématiquement toute distinction orthographique peut changer le sens. La normalisation doit être testée avec des contre-exemples.'),
     h('Alias et intentions'),
     p('Des alias rapprochent des termes comme wra9, documents et pièces. Cela aide une recherche lexicale, mais ne transforme pas un dictionnaire en compréhension générale. Une liste finie ne couvre pas toutes les paraphrases, dialectes régionaux ou fautes de frappe.'),
     table(['Question', 'Intention à distinguer'], [
         ['Quels papiers préparer ?', 'Documents requis.'],
         ['Combien cela coûte ?', 'Frais.'],
         ['Quand déposer le dossier ?', 'Délai de dépôt.'],
         ['Quand recevoir la carte ?', 'Délai de délivrance.']]),
     p('Le <b>topic</b> décrit le sujet, par exemple auto-entrepreneur ; l’<b>intent</b> décrit l’information demandée. Deux questions peuvent partager le même sujet sans avoir la même réponse. Un délai de dépôt ne répond pas à un délai de fabrication, même si le passage contient « jours » et « carte ».'),
     note('Les règles de ce dépôt sont versionnées. Les changer peut modifier les vecteurs et le classement : une nouvelle ingestion est nécessaire quand le contrat de l’index devient incompatible. Versionner la normalisation fait donc partie de la reproductibilité.'),
     refs('À lire : normalization.py ; QUERY_RULES_VERSION dans contracts.py.'))
