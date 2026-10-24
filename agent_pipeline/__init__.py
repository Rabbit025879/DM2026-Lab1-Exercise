# Importing helpers points NLTK_DATA / SCIKIT_LEARN_DATA at the project folder.
# The agentic notebooks never import helpers directly, and student workspace
# tools are loaded before premade_tools, so do it as soon as this package loads.
import helpers  # noqa: F401
