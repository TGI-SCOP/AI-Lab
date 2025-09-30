# AI-Lab

Un laboratoire ouvert pour l’expérimentation et la recherche en IA.

## Objectifs
- Itérer vite avec des expériences courtes et traçables.
- Garder la reproductibilité (scripts, versions, CI).
- Capitaliser les résultats (métriques + modèles sauvegardés).

## Structure
```
data/            # non versionné par défaut (volumineux / sensible)
models/          # artefacts de modèles (.pkl, .pt, …)
experiments/     # scripts + métriques JSON par expérience
notebooks/       # explorations Jupyter
src/ai_lab/      # code réutilisable (fonctions utilitaires, modules)
tests/           # tests unitaires
.github/workflows/ci.yml  # CI
```

## Démarrer
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/train_classifier.py
```

## Résultats
- `experiments/iris_logreg_metrics.json` : métriques d’évaluation
- `models/iris_logreg.pkl` : modèle entraîné (pickle)

## Roadmap courte (v0.2)
- Ajout d’un suivi d’expériences (simple logger CSV/JSON).
- Jeux de données externes (Hugging Face Datasets).
- Baselines vision & NLP (CPU friendly).
