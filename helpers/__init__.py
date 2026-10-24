__all__ = ["emo_data_helpers","emo_cnn_helpers", "vowel_helpers", "deepir","word2vec_helpers","pickle_helpers", \
"liwc_helpers", "cnn_data_helpers", "cnn_model_helpers", "word_embeddings"]

# Keep the NLTK and scikit-learn dataset downloads inside the project folder
# (./nltk_data, ./scikit_learn_data) instead of the user's home directory.
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Set (not setdefault) so an NLTK_DATA / SCIKIT_LEARN_DATA already exported in
# the shell can't send the downloads back to the home directory. Subprocesses
# started from the notebook (e.g. the agent-dev pytest runs) inherit these too.
os.environ["SCIKIT_LEARN_DATA"] = os.path.join(_PROJECT_ROOT, "scikit_learn_data")
os.environ["NLTK_DATA"] = os.path.join(_PROJECT_ROOT, "nltk_data")
# nltk only downloads into a directory that already exists, and it fixes both
# its search path and its download target when it is first imported -- so
# create the folder before importing it, and patch both in case it was
# imported earlier.
os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)

import nltk
import nltk.downloader

if os.environ["NLTK_DATA"] not in nltk.data.path:
    nltk.data.path.insert(0, os.environ["NLTK_DATA"])
nltk.downloader._downloader.download_dir = os.environ["NLTK_DATA"]
