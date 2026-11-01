from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> Dict[str, Any]:
        """Compute cosine similarity between two documents in the document-term matrix.

        Retrieves row vectors for doc_index_1 and doc_index_2 from session.feature_matrix,
        calculates cosine similarity using raw term counts, and handles empty/zero rows cleanly.

        Args:
            doc_index_1: Index of the first document (0-based).
            doc_index_2: Index of the second document (0-based).

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, and similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found. Please run build_dtm_tool first."}

        dtm = session.feature_matrix
        n_samples = dtm.shape[0]

        if not (0 <= doc_index_1 < n_samples):
            return {"error": f"doc_index_1 ({doc_index_1}) out of range [0, {n_samples - 1}]."}
        if not (0 <= doc_index_2 < n_samples):
            return {"error": f"doc_index_2 ({doc_index_2}) out of range [0, {n_samples - 1}]."}

        vec1 = dtm[doc_index_1]
        vec2 = dtm[doc_index_2]

        sim_matrix = cosine_similarity(vec1, vec2)
        sim_val = float(sim_matrix[0, 0])

        if np.isnan(sim_val):
            sim_val = 0.0

        result_id = session.next_result_id("cosine_similarity")
        args_dict = {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}

        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(sim_val, 4),
        }

        session.store_result("cosine_similarity_tool", args_dict, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Compute feature correlation matrix and plot heatmap for top terms by variance.

        Selects top_k terms with highest variance from session.feature_matrix, computes
        their Pearson correlation matrix, and renders an annotated heatmap set on session.pending_figure.

        Args:
            top_k: Number of highest-variance terms to include in correlation matrix (default: 20).

        Returns:
            Dict containing result_id, terms list, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Please run build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = np.array(session.feature_names)

        if hasattr(dtm, "toarray"):
            dense_dtm = dtm.toarray()
        else:
            dense_dtm = np.asarray(dtm)

        variances = dense_dtm.var(axis=0, ddof=0)
        n_features = len(feature_names)

        k_actual = min(top_k, n_features)
        top_indices = np.argsort(variances)[::-1][:k_actual]

        top_terms = feature_names[top_indices].tolist()
        top_dtm_slice = dense_dtm[:, top_indices]

        # Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(top_dtm_slice, rowvar=False)

        # Handle NaNs if constant terms are present
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Build annotated correlation heatmap figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Feature Correlation Matrix (Top-{k_actual} Terms by Variance)")
        fig.tight_layout()

        # Set matplotlib figure on session.pending_figure per spec requirement
        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("feature_correlation")
        args_dict = {"top_k": top_k}

        # Convert corr_matrix to rounded JSON-serializable list
        corr_list = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result("feature_correlation_matrix_tool", args_dict, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
