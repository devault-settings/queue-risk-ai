# Notebooks

Name notebooks `NN_owner_topic.ipynb` (for example, `01_audrey_eda.ipynb`) so they sort and don't collide.

- Import shared code with `from queue_risk import config, data`. Don't copy loaders into notebooks.
- Clear outputs before committing large notebooks, or keep them small.
- Code that other workstreams depend on belongs in `src/queue_risk/`.
