from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> Dict[str, Any]:
        """Build the global Document-Term Matrix (DTM) from session.dataframe text.

        Fits CountVectorizer, updates session.feature_matrix, session.feature_names,
        session.artifacts['count_vectorizer'], sets category labels, and reports sparsity.

        Args:
            max_features: Maximum number of terms to include in vocabulary (default: None).
            ngram_range: Lower and upper boundary of range of n-values for n-grams as [min_n, max_n] (default: [1, 1]).
            min_df: Ignore terms with document frequency strictly lower than threshold (default: 1).
            max_df: Ignore terms with document frequency strictly higher than threshold (default: 1.0).

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, sample_feature_names, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Please run load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "error": "DataFrame does not contain a 'text' column."
            }

        # Convert List[int] ngram_range to tuple (min_n, max_n) inside body
        if len(ngram_range) != 2:
            return {
                "error": f"ngram_range must be a list of 2 integers [min_n, max_n], got {ngram_range}."
            }
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        # Handle text series with missing values filled as empty strings
        text_series = df["text"].fillna("")

        # Fit CountVectorizer
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
        )

        dtm_matrix = vectorizer.fit_transform(text_series)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = dtm_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels in session if category/label column exists
        label_col = "category_name" if "category_name" in df.columns else "label"
        if label_col in df.columns:
            session.set_labels(df[label_col].values)

        # Sparsity calculation
        n_rows, n_cols = dtm_matrix.shape
        non_zero = int(dtm_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        args_dict = {
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
        }

        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "sample_feature_names": feature_names[:10],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", args_dict, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Optional limit on the number of top frequent terms to return (default: None).

        Returns:
            Dict containing result_id, total_terms, and term_frequencies dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-Term Matrix not built. Please run build_dtm_tool first."
            }

        # Calculate column sums (frequency per term across all documents)
        col_sums = np.array(session.feature_matrix.sum(axis=0)).flatten()

        term_freq_pairs = [
            (term, int(freq))
            for term, freq in zip(session.feature_names, col_sums)
        ]

        # Sort descending by total frequency
        term_freq_pairs.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n > 0:
            term_freq_pairs = term_freq_pairs[:top_n]

        term_frequencies = {term: freq for term, freq in term_freq_pairs}

        result_id = session.next_result_id("term_frequency")
        args_dict = {"top_n": top_n}
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": term_frequencies,
        }

        session.store_result("term_frequency_tool", args_dict, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generate a heatmap slice of the raw Document-Term Matrix (DTM).

        Args:
            n_terms: Number of terms (columns) to slice from existing vocabulary order (default: 20).
            n_documents: Number of documents (rows) to slice from existing row order (default: 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, terms, and dtm_slice array.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-Term Matrix not built. Please run build_dtm_tool first."
            }

        if n_terms <= 0 or n_documents <= 0:
            return {
                "error": "Parameters 'n_terms' and 'n_documents' must be positive integers."
            }

        total_docs, total_terms = session.feature_matrix.shape
        rows_to_take = min(n_documents, total_docs)
        cols_to_take = min(n_terms, total_terms)

        matrix_slice = session.feature_matrix[:rows_to_take, :cols_to_take].toarray()
        sliced_terms = session.feature_names[:cols_to_take]

        # Render Heatmap Plot
        fig, ax = plt.subplots(figsize=(max(6, cols_to_take * 0.8), max(4, rows_to_take * 0.5)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=list(range(rows_to_take)),
            ax=ax,
        )
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        ax.set_title("Document-Term Matrix Heatmap")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        args_dict = {"n_terms": n_terms, "n_documents": n_documents}

        summary = {
            "result_id": result_id,
            "n_documents": rows_to_take,
            "n_terms": cols_to_take,
            "terms": sliced_terms,
            "dtm_slice": matrix_slice.tolist(),
        }

        session.store_result("dtm_heatmap_tool", args_dict, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
