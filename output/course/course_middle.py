from course_data import page, p, h, note, code, table, fig, ar, refs

page('10 · CONNAISSANCE', 'Le RAG : consulter avant de répondre', 'Une mémoire documentaire externe aux poids',
     p('RAG signifie retrieval-augmented generation : génération augmentée par recherche. L’application récupère des passages pertinents, les place dans le contexte du modèle et lui demande de formuler une réponse soutenue par ces passages. Le travail fondateur distingue une mémoire dans les paramètres d’une mémoire externe consultable. Notre pipeline reprend ce principe général sans reproduire son entraînement conjoint. [R1]'),
     table(['Mécanisme', 'Ce qui change', 'Usage typique'], [
         ['Prompt', 'Instructions et contexte de la requête.', 'Choisir langue, format et règles.'],
         ['RAG', 'Documents recherchés à la demande.', 'Apporter des faits traçables et actualisables.'],
         ['Fine-tuning', 'Poids appris sur des exemples.', 'Adapter un comportement ou une compétence.'],
         ['Entraînement initial', 'Poids à grande échelle.', 'Construire les capacités de base.']]),
     p('Ajouter un PDF à Qdrant ne réentraîne pas Qwen. Le modèle peut utiliser ce PDF pendant la requête parce que l’application lui en transmet une portion. Une autre requête ne reçoit pas automatiquement le même passage. Cette distinction explique pourquoi on peut mettre à jour un corpus sans recalculer des milliards de paramètres.'),
     h('Le RAG n’annule pas les erreurs'),
     p('Un document peut être périmé, un passage mal extrait, la recherche peut manquer une exception, ou le modèle peut mal traduire une preuve correcte. Ces erreurs s’additionnent. Une bonne réponse exige une chaîne de conditions : source utilisable, passage pertinent, interprétation correcte, rédaction fidèle et restitution lisible.'),
     p('Il faut aussi autoriser l’<b>abstention</b>. Quand le corpus ne contient pas les frais d’une procédure, le modèle ne doit pas combler le vide avec ses souvenirs. L’absence de preuve doit être un résultat normal du produit. Le refus doit toutefois préciser sa cause : information absente, source à revoir ou sortie générée non validée.'),
     note('Le RAG est utile pour accéder à une connaissance externe. Le fine-tuning peut aider un style ou une langue, mais ne remplace pas une politique de sources à jour. On ne choisit pas entre les deux sans identifier d’abord le problème mesuré.'))

page('11 · CONNAISSANCE', 'La provenance et la fraîcheur', 'Une source téléchargée aujourd’hui peut être ancienne',
     p('La <b>provenance</b> répond à quatre questions : qui publie, quel fichier a été utilisé, où se situe le passage et quand son usage a été vérifié ? Le domaine officiel est un indice important, mais pas une preuve que chaque phrase est exhaustive, actuelle ou applicable à chaque situation.'),
     fig('source', 'La date de téléchargement ne remplace pas une revue. Le hash identifie le contenu ; il ne prouve pas sa vérité.'),
     p('Le manifeste conserve notamment l’URL, l’éditeur, les pages, la date de collecte et un statut. <b>reviewed=true</b> autorise l’ingestion après inspection ; cela n’autorise pas encore une réponse affirmative. La politique d’utilisation exige une revue datée, une échéance et un hash correspondant au fichier approuvé.'),
     p('Un <b>SHA-256</b> est une empreinte calculée à partir des octets. Une modification du fichier change normalement l’empreinte. Cela aide à détecter une divergence entre le contenu approuvé et celui utilisé. Un document mensonger possède lui aussi un hash valide : intégrité et exactitude sont deux propriétés différentes.'),
     h('Exemple de calendrier du projet'),
     p('La section d’inscription du guide DGI 2026 a été revue le 3 octobre avec un réexamen interne fixé au 2 novembre 2026. Ce créneau représente la politique du projet, pas une durée de validité légale. Une source historique ou non vérifiée reste consultable comme référence séparée, mais ne soutient pas une réponse actuelle.'),
     note('Une API peut être prête avec zéro source approuvée. Elle doit alors expliquer pourquoi elle ne répond pas. Confondre « prêt techniquement » et « dispose d’une preuve actuelle » crée des interfaces trompeuses.'),
     refs('À lire : source_policy.py ; docs/source-quality.md ; tests/test_source_policy.py. Exemple documentaire, pas conseil administratif.'))

page('12 · CONNAISSANCE', 'Extraire et découper les documents', 'Le chunk doit conserver les conditions utiles',
     p('L’<b>extraction</b> transforme HTML, PDF ou texte en contenu exploitable. BeautifulSoup sélectionne des zones HTML ; pypdf lit le texte accessible d’un PDF. Un PDF scanné peut nécessiter un OCR supplémentaire : extraire du texte et reconnaître des pixels sont deux tâches différentes. L’OCR n’est pas intégré au chemin actuel.'),
     fig('chunks', 'Exemple fictif : séparer « nécessaire » de « sauf pour… » fabrique un passage trompeur. Une frontière sémantique vaut mieux qu’une coupe mécanique.'),
     p('Un <b>chunk</b> est une portion recherchable. Trop petit, il perd les conditions ; trop grand, il dilue les mots utiles et consomme le contexte. L’<b>overlap</b> répète une partie entre deux chunks pour préserver les liaisons, au prix de doublons. Le profil lexical découpe en unités séparées par espaces : ses 350 unités ne sont pas 350 tokens Qwen.'),
     p('Le projet utilise une taille générale de 350 avec recouvrement de 50, mais traite les sections PDF sensibles par frontières explicites. Une section sur deux pages est assemblée avec des marqueurs vérifiés. Elle doit tenir entière ; si elle dépasse la limite, l’ingestion échoue plutôt que de couper discrètement une condition.'),
     h('Nettoyage contrôlé'),
     p('Les en-têtes et numéros de page ne doivent être retirés que lorsque leur forme attendue est connue. Une règle trop large pourrait effacer une ligne du corps. Les numéros de pages PDF et les folios imprimés peuvent différer. Il faut enregistrer le repère réellement utilisé pour que l’utilisateur retrouve le passage.'),
     note('Teste les marqueurs absents, répétés ou déplacés. Une extraction qui « réussit » en retournant le mauvais paragraphe est plus dangereuse qu’un échec explicite. La fidélité commence avant tout appel au LLM.'),
     refs('À lire : ingestion/sources.py ; tests/test_pdf_sections.py.'))

page('13 · RECHERCHE', 'Les vecteurs et la similarité', 'Transformer une comparaison de textes en calcul',
     p('Un <b>vecteur</b> est une liste ordonnée de nombres. Pour rechercher, on représente la question et les passages dans le même espace. Une métrique produit un score de proximité. La similarité cosinus compare leur direction : le produit scalaire est divisé par le produit des longueurs des deux vecteurs.'),
     code('cos(q, d) = (q · d) / (||q|| × ||d||)'),
     fig('vectors', 'Exemple fictif à trois dimensions. Le score mesure une proximité de représentation, pas la probabilité que le passage soit vrai.'),
     p('Prenons q=(1,1,0), d1=(1,1,0), d2=(1,0,1). Le cosinus entre q et d1 vaut 1 ; entre q et d2, il vaut 1/2. Après normalisation L2, chaque vecteur a une longueur de 1 et le cosinus devient simplement le produit scalaire. Un vecteur nul exige un traitement spécial pour éviter une division par zéro.'),
     h('La baseline réellement active'),
     p('LexicalEmbedder compte des termes normalisés. Pour chaque terme, SHA-256 choisit une case parmi 4 096 ; la contribution vaut 1 + log(fréquence), avec la fréquence entière du terme. La somme est ensuite normalisée en L2. C’est un sac de termes hachés, pas une représentation neuronale du sens. L’ordre exact des mots y est largement perdu.'),
     p('Deux mots différents peuvent tomber dans la même case : c’est une <b>collision</b>. Le pipeline exige donc aussi un recouvrement réel de termes pour les candidats lexicaux. Cette précaution ne résout ni les synonymes absents des alias, ni les contradictions contenant les mêmes mots.'),
     note('Le seuil 0,18 du projet est un réglage de départ non calibré. Il ne signifie ni « 18 % de confiance » ni « réponse correcte ». Une similarité élevée est une invitation à examiner la preuve, pas un verdict.'),
     refs('À lire : retrieval/embeddings.py. Métriques Qdrant : [R9].'))

page('14 · RECHERCHE', 'Embeddings sémantiques et Qdrant', 'L’encodeur crée les vecteurs ; la base les recherche',
     p('Un <b>embedding neuronal</b> résulte d’un modèle entraîné à produire une représentation utile du texte. Selon l’entraînement, des formulations différentes peuvent être rapprochées. Ce comportement doit être mesuré dans notre domaine et nos langues : un modèle multilingue annoncé ne garantit pas une bonne recherche Darija-français sur nos documents.'),
     p('L’adaptateur optionnel Qwen3-Embedding-0.6B produit ici des vecteurs de 1 024 dimensions normalisés et ajoute une instruction à la question. La fiche du modèle décrit son usage pour la représentation et la recherche de textes. L’adaptateur est présent, mais son bénéfice réel n’a pas encore été mesuré dans cette application. [R10]'),
     table(['Élément', 'Responsabilité'], [
         ['Encodeur', 'Transformer un texte en vecteur.'],
         ['Collection Qdrant', 'Regrouper des points recherchables.'],
         ['Point', 'Associer un identifiant, un vecteur et des métadonnées.'],
         ['Payload', 'Conserver le passage, sa source, son sujet et sa localisation.'],
         ['Pipeline', 'Décider quels résultats peuvent soutenir une réponse.']]),
     p('Qdrant organise les vecteurs et leurs métadonnées. Dans ce dépôt, il s’exécute en mode local embarqué. Une collection doit respecter son contrat de dimension et de distance. Mélanger 4 096 composantes lexicales et 1 024 composantes neuronales n’est pas une migration valide. Même deux modèles de même dimension peuvent représenter le texte dans des espaces incompatibles. [R9]'),
     h('FAISS, BM25, reranker : où se situent-ils ?'),
     p('FAISS est une bibliothèque de recherche vectorielle évoquée comme alternative, pas le stockage actif. BM25 est une méthode de classement lexical pondérant notamment fréquence des termes et longueur du document. Un reranker reclasse une petite liste de candidats en comparant plus finement question et passage. Une recherche hybride combine signaux lexicaux et sémantiques. Ces pistes sont distinctes et non implémentées ici.'),
     note('Changer l’encodeur impose de recalculer les vecteurs du corpus et ceux des questions avec la même convention. Remplacer seulement le modèle côté question peut produire des nombres plausibles et des résultats inutiles.'))

page('15 · RECHERCHE', 'Du top-k aux preuves utilisables', 'La pertinence thématique ne suffit pas',
     p('La recherche retourne un classement de candidats. <b>Top-k</b> signifie conserver les k premiers selon le score. Le pipeline demande jusqu’à 10 candidats, applique des filtres et conserve au maximum 4 chunks de preuve avant le budget final du générateur. Demander davantage de passages peut augmenter le rappel, mais aussi le bruit et le coût.'),
     table(['Filtre du projet', 'Pourquoi il existe', 'Limite'], [
         ['Sujet', 'Écarter un domaine manifestement différent.', 'Classification conservatrice.'],
         ['Score minimal', 'Retirer des rapprochements faibles.', 'Seuil non calibré.'],
         ['Doublons', 'Éviter de répéter la même preuve.', 'Des paraphrases restent possibles.'],
         ['Actualité de la revue', 'Ne pas utiliser une source non approuvée.', 'Pas de vérité garantie entre deux revues.'],
         ['Indices de réponse', 'Exiger des indices compatibles avec l’intention.', 'Ce ne sont pas des preuves sémantiques complètes.']]),
     p('Imagine une question sur le coût et un passage qui dit seulement où déposer un dossier. Le mot « inscription » peut donner un bon score, sans qu’aucun montant ne soit présent. La question est dans le bon domaine, mais le détail recherché manque. C’est un problème d’<b>answerability</b> : peut-on réellement répondre avec ce contexte ?'),
     h('Deux erreurs opposées'),
     p('Un faux positif de recherche présente un passage inutile comme pertinent. Un faux négatif manque une preuve valable. Monter le seuil peut réduire le premier et aggraver le second. Il faut donc calibrer sur un jeu annoté, en observant les deux types d’erreurs et pas seulement quelques exemples réussis.'),
     p('La preuve finale doit aussi conserver son identité à travers les étapes. Si le générateur retire un chunk pour respecter le contexte, les citations autorisées doivent correspondre aux chunks effectivement transmis. Citer un passage récupéré mais jamais vu par le modèle rendrait l’explication du résultat trompeuse.'),
     note('Une base vectorielle ne comprend pas seule les exigences du produit. La politique de preuve appartient à l’application : filtres, versions, exceptions, abstention et journal d’évaluation.'),
     refs('À lire : AnswerPipeline.answer dans pipeline.py ; generation/ollama.py.'))

page('16 · RECHERCHE', 'Versionner et activer un corpus', 'Une mise à jour doit être identifiable et récupérable',
     p('Un <b>manifeste</b> décrit le corpus : sources, versions de transformation, modèle d’embedding, dimension, métrique et identifiant de release. Il permet de savoir avec quoi une réponse a été produite. Sans lui, deux réponses différentes pourraient venir d’une mise à jour de données, d’une règle de normalisation ou d’un modèle, sans moyen de trancher.'),
     h('Construire avant de basculer'),
     p('L’ingestion crée une nouvelle collection immuable, vérifie qu’elle est cohérente, puis remplace le pointeur actif. L’<b>activation atomique</b> signifie que les lecteurs voient l’ancien ou le nouveau pointeur, pas un demi-fichier. Si le téléchargement ou l’extraction échoue avant activation, l’ancien corpus reste actif.'),
     code('sources -> extraction -> chunks -> vecteurs -> nouvelle collection\ncontrôles réussis -> remplacement atomique de active.json\néchec avant activation -> ancienne release conservée'),
     p('L’atomicité ne signifie pas que toute panne imaginable est impossible. Il faut toujours conserver les anciennes collections, les fichiers source et les informations de version nécessaires à une récupération. Le projet conserve les anciennes collections ; il ne les supprime pas automatiquement.'),
     table(['Changement', 'Réingestion ?'], [
         ['Nouvel embedding ou nouvelle révision', 'Oui, espace de représentation modifié.'],
         ['Nouvelles règles affectant les vecteurs', 'Oui, contrat de l’index modifié.'],
         ['Nouvelles frontières de chunks', 'Oui, unités documentaires modifiées.'],
         ['Autre générateur sur les mêmes vecteurs', 'Pas nécessaire pour cette seule raison.']]),
     p('Le mode local de Qdrant a un propriétaire unique dans ce workflow. Il faut arrêter l’API avant l’ingestion puis la redémarrer. Lancer plusieurs processus sur le même stockage ne crée pas un cluster : cela crée une concurrence non prévue et des erreurs de verrouillage.'),
     note('Le versionnement applicatif 0.2.0, la version des règles, le digest du modèle et la release du corpus sont des identités différentes. Un numéro de version Python ne décrit pas à lui seul la totalité d’une expérience.'),
     refs('À lire : ingestion/pipeline.py ; retrieval/store.py ; initialisation de AnswerPipeline.'))

page('17 · GÉNÉRATION', 'Prompt, JSON et citations', 'Donner une structure contrôlable au résultat',
     p('Un <b>prompt système</b> définit le rôle et les contraintes : utiliser les preuves fournies, respecter les conditions, répondre dans la langue demandée et s’abstenir si nécessaire. Les passages sont des données non fiables en tant qu’instructions. Une phrase trouvée dans un PDF ne doit pas pouvoir modifier le rôle de l’assistant.'),
     p('Le projet demande un objet JSON conforme à un schéma. Chaque affirmation dispose d’un texte et d’identifiants de citations. Les identifiants autorisés sont limités aux preuves réellement sélectionnées. Cela évite notamment des références inventées ou des UUID mal recopiés. Ollama peut contraindre le format, puis Pydantic vérifie le contrat côté application. [R5, R12]'),
     code('Exemple fictif, identifiant abrégé pour la lecture :\n{\n  "outcome": "answered",\n  "statements": [{\n    "text": "Le dossier de démonstration exige une photo.",\n    "citation_ids": ["preuve-A"]\n  }]\n}'),
     h('Trois niveaux de validation'),
     table(['Niveau', 'Question posée', 'Exemple de défaut'], [
         ['Syntaxe', 'Est-ce du JSON lisible ?', 'Accolade manquante.'],
         ['Contrat', 'Les champs et valeurs sont-ils permis ?', 'Statut inconnu ou citation absente du contexte.'],
         ['Sens', 'La preuve soutient-elle l’affirmation ?', 'Photo transformée en passeport.']]),
     p('Les deux premiers niveaux sont automatisables de manière assez stricte. Le troisième est plus difficile : « dans trente jours » et « au plus tard dans trente jours » ne disent pas exactement la même chose. Une citation valide prouve uniquement que l’identifiant existe ; elle ne prouve pas l’implication logique entre le passage et la phrase.'),
     note('Le JSON facilite les contrôles et l’interface, mais ne rend pas la réponse vraie. La structure doit servir l’inspection : affirmation par affirmation, preuve par preuve, condition par condition.'),
     refs('À lire : generation/adapters.py ; GeneratedAnswer dans contracts.py.'))

page('18 · GÉNÉRATION', 'Valider, réessayer, s’abstenir', 'Un refus explicite peut être le bon comportement',
     fig('validation', 'Deux tentatives maximum pour une sortie invalide. La seconde ne transforme pas une absence de preuve en autorisation d’inventer.'),
     p('Les contrôles rejettent un format incorrect, une référence inconnue ou certains écarts de langue. Des règles ciblées examinent aussi les erreurs rencontrées : document d’identité inventé, chiffre absent des preuves, signature ou condition importante omise. Elles sont déclenchées par le texte source et l’intention, pas par une réponse administrative universelle codée en dur.'),
     p('Ces garde-fous restent <b>heuristiques</b>. Ils peuvent refuser une bonne paraphrase qui utilise un vocabulaire inattendu, ou accepter une erreur qui contient tous les mots recherchés. Leur réussite sur un test de régression ne les transforme pas en système général de vérification sémantique.'),
     table(['Résultat observable', 'Interprétation correcte'], [
         ['insufficient_evidence', 'Les éléments disponibles ne permettent pas la réponse.'],
         ['verification_required', 'Des références pertinentes nécessitent une revue.'],
         ['generation_failed=true', 'La sortie générée a échoué aux contrôles.'],
         ['clarification_required', 'Une précision, notamment de langue, est nécessaire.']]),
     p('L’API conserve actuellement insufficient_evidence comme outcome après un échec de génération, et ajoute generation_failed=true pour distinguer la cause. L’interface doit lire ce drapeau et afficher « Réponse non validée ». Ne pas faire cette distinction ferait croire que les documents sont absents alors que le problème vient de la rédaction.'),
     note('Réessayer une fois donne une chance à une autre sortie ; ce n’est pas une boucle infinie jusqu’à obtenir un texte acceptable. Le budget, la latence et le taux d’abstention doivent inclure le coût des tentatives échouées.'),
     refs('À lire : generation/grounding.py ; pipeline.py ; presentation.py.'))

page('19 · GÉNÉRATION', 'Pourquoi la réponse semblait fausse', 'Étudier le défaut de la chaîne, pas seulement la phrase',
     p('Tu avais signalé une réponse arabe reprenant un passage sur le dépôt du dossier dans une agence bancaire. Le problème pédagogique est que la présence d’une citation ou d’une phrase authentique ne suffit pas à produire une réponse adaptée, actuelle et complète. Il faut reconstituer la chaîne de provenance et de transformation.'),
     h('Quatre questions de diagnostic'),
     table(['Contrôle', 'Ce qu’il fallait examiner'], [
         ['Provenance', 'Le passage venait-il d’une brochure historique ou d’une source revue ?'],
         ['Portée', 'La formulation perdait-elle une restriction sur les agences concernées ?'],
         ['Extraction', 'Les conditions continuaient-elles sur la page suivante ?'],
         ['Restitution', 'Était-ce un extrait arabe standard ou une vraie réponse en Darija ?']]),
     p('Un extrait littéral peut être fidèle à un vieux document tout en n’étant pas une réponse actuelle sûre. Une traduction peut aussi remplacer une notion précise par un terme trop large. L’amélioration ne consiste donc pas seulement à écrire un prompt plus autoritaire : elle commence par le choix de source, sa revue et la conservation des conditions.'),
     p('Le corpus actuel conserve une section complète du guide DGI 2026 et met à part les références historiques ou non vérifiées. La politique documente les restrictions et les dates de revue. Ce cours décrit le traitement informatique de ce cas ; pour une démarche réelle, il faut consulter les sources officielles actuelles et confirmer les modalités pratiques.'),
     h('Le test qui empêche le retour du défaut'),
     p('Un bon test de régression n’attend pas une phrase exacte du modèle. Il vérifie les invariants : pas de citation historique utilisée comme preuve actuelle, pas de condition tronquée, pas de délai de dépôt présenté comme délai de réception, langue affichée honnêtement et cause d’abstention explicite.'),
     note('La question essentielle n’est pas « est-ce que la réponse sonne bien ? », mais « quelle preuve autorise chaque partie de cette réponse, dans ce contexte et à cette date ? »'),
     refs('À lire : docs/source-quality.md ; tests/test_pdf_sections.py ; tests/test_grounding.py.'))
