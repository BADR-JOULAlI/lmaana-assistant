# Diagnostic de recherche sur un snapshot documentaire

`prepare-retrieval` construit un jeu local à partir d'une copie de source et
d'annotations versionnées. Il réutilise les règles d'extraction du manifeste,
vérifie le hash des octets avant extraction et découpe des passages par marqueurs
uniques. Chaque passage reste une sous-chaîne exacte de sa section extraite.

La commande écrit le jeu compatible avec `benchmark-retrieval` et un fichier
`.provenance.json` contenant URL, hash source, emplacement, offsets de caractères,
hash du texte, hash de la spécification, hash du manifeste validé et familles de
questions. Le hash du jeu relie les annotations, les passages et les rapports.
Le préparateur n'ouvre pas l'index actif et ne contacte pas de service.

## Préparer la copie locale

La spécification `configs/retrieval-dgi-snapshot.dev.json` référence la source
`dgi-ae-2026-registration` de `configs/sources.example.json`. Obtenir la copie du
PDF depuis l'URL officielle inscrite dans ce manifeste et la conserver localement,
par exemple dans `work/dgi-ae-2026.pdf`. La copie doit correspondre à :

```text
SHA-256 : 7fa82a063e2f74001ecd03f4c2803e86295b35b514d4eef4f8adaee695bab9cd
```

Le dépôt conserve les marqueurs, questions et labels. Le PDF et les passages
extraits restent dans les dossiers locaux ignorés par Git, conformément à la
politique de sources. Si le site ne fournit plus ces octets, la préparation doit
échouer : une nouvelle édition nécessite une nouvelle revue des passages et labels.

```powershell
uv run --locked lmaana prepare-retrieval configs/retrieval-dgi-snapshot.dev.json --manifest configs/sources.example.json --source work/dgi-ae-2026.pdf --output outputs/dgi-registration-snapshot.dev.json
uv run --locked lmaana benchmark-retrieval outputs/dgi-registration-snapshot.dev.json --query-mode raw --output outputs/dgi-lexical-raw.dev.json
uv run --locked lmaana benchmark-retrieval outputs/dgi-registration-snapshot.dev.json --query-mode expanded --output outputs/dgi-lexical-expanded.dev.json
```

Ces commandes utilisent l'environnement du projet. Avec un environnement de
vérification séparé, lancer les mêmes commandes avec son Python et
`-m lmaana_assistant.cli`. La préparation lit uniquement les trois fichiers fournis.
Les rapports ne peuvent pas écraser la spécification, le manifeste ou le fichier
source, y compris lorsqu'un chemin existant désigne le même fichier.

## Périmètre des annotations initiales

Le jeu contient trois passages de la section d'inscription, conservant la
continuité entre les pages PDF 11–12. Il propose 50 questions : sept familles
positives et trois familles négatives, chacune en français, arabe standard,
Darija arabe, Arabizi et anglais. Les 35 positives couvrent les éléments de
recherche présents dans les passages sélectionnés ; les 15 négatives concernent
des informations absentes de cette sélection. Le reste du PDF n'est pas indexé
dans cette expérience.

Les labels et formulations sont **des brouillons rédigés pour le développement**.
La vérification structurelle et le contrôle du passage source ont été réalisés ;
une relecture humaine indépendante de la pertinence et des langues reste à faire.
Le statut `draft` est conservé dans les métadonnées et les notes du rapport.

Les traductions d'une famille partagent les mêmes labels. Conserver chaque
snapshot de source et toutes les traductions d'une famille dans un seul split.
Les questions actuelles partagent un unique document : elles ne constituent pas
50 observations indépendantes, ni un test final représentatif des services publics.
Une mesure robuste nécessite davantage de sources, de distracteurs et de scénarios
annotés indépendamment, puis un jeu final tenu à l'écart du développement.

La préparation d'un snapshot ne renouvelle pas les revues de sources de
l'application. Le 29 octobre 2026, l'accès au PDF par l'outil web a échoué ; la
copie locale a été contrôlée par hash et les deux pages pertinentes inspectées
visuellement. Aucune actualité administrative supplémentaire n'en est déduite.

## Résultats locaux du 29 octobre 2026

Profil `lexical-sha256-4096-v3`, règles `darija-rules-v5`, seuil cosinus `0.18`.
Hash canonique du jeu préparé :
`bca9b033b0bc0cd3ef9fc856f1c34097937107b164d1ecbb418d7f15f5c0293f`.
Les résultats ci-dessous utilisent les labels brouillons décrits plus haut.

| Entrée | Positives avec bon premier passage, après seuil | Recall@1 avant seuil | MRR@3 après seuil | Négatives avec candidat, après seuil |
|---|---:|---:|---:|---:|
| Question brute | 5/35 (14,3 %) | 16/35 (45,7 %) | 0,1571 | 4/15 |
| Question étendue | 3/35 (8,6 %) | 5/35 (14,3 %) | 0,1286 | 8/15 |

Le seuil écarte plusieurs passages correctement classés, mais les rangs sans
seuil se dégradent aussi avec l'expansion. Cela justifie de mesurer les deux modes
dans les expériences suivantes. Le seuil commun est un contrôle expérimental ;
ne pas le recalibrer sur ces brouillons et annoncer ensuite une performance finale
sur les mêmes questions.

Recall@3 avant seuil vaut 100 % dans les deux modes : avec seulement trois passages
et des requêtes non nulles, cette valeur ne démontre pas une bonne recherche.
Les candidats négatifs ne sont pas des réponses affichées : les filtres de sujet,
de fraîcheur et d'answerability de l'application ne sont pas exécutés ici.
Ce diagnostic ne mesure ni génération, ni ASR, ni qualité Darija d'un modèle.

## Vérifications logicielles

La suite passe à **411 tests réussis** sous Windows/Python 3.12.10. Les nouveaux
cas couvrent la modification de source, les désaccords de hashes, les références
non uniques, les marqueurs manquants/ambigus/inversés, les chevauchements, la
cohérence des familles et la protection des fichiers d'entrée. Ruff est conforme.
Le hash de l'ancienne fixture logicielle reste inchangé. Le blocage du runtime
Qwen documenté dans `evaluation.md` n'a pas été retenté.
