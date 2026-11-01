from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Perform dimensionality reduction (PCA, t-SNE, or UMAP) on the document-term matrix.

        Applies the specified reduction algorithm on session.feature_matrix and generates
        a 2D scatter plot colored by document category, setting session.pending_figure.

        Args:
            method: Reduction algorithm: "pca", "tsne", or "umap" (default: "pca").
            n_components: Target number of dimensions, usually 2 (default: 2).
            perplexity: Perplexity parameter for t-SNE (default: 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default: 15).
            random_state: Random seed for reproducible results (default: 42).

        Returns:
            Dict containing result_id, method, coordinates, and metadata like explained_variance_ratio.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found. Please run build_dtm_tool first."}

        dtm = session.feature_matrix
        if hasattr(dtm, "toarray"):
            dense_dtm = dtm.toarray()
        else:
            dense_dtm = np.asarray(dtm)

        method_clean = method.lower()
        explained_variance = None

        if method_clean == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
            explained_variance = reducer.explained_variance_ratio_.tolist()
        elif method_clean == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
        elif method_clean == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
        else:
            return {"error": f"Unknown method '{method}'. Supported methods are 'pca', 'tsne', and 'umap'."}

        # Store reduced coordinates in session.artifacts
        session.artifacts[f"reduced_{method_clean}"] = coords

        # Generate mandatory scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))

        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "label" in session.dataframe.columns:
                labels = session.dataframe["label"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()

        if labels is None and session.labels is not None:
            labels = list(session.labels)

        if labels is not None and len(labels) == coords.shape[0]:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=labels,
                palette="Set2",
                s=100,
                ax=ax,
            )
            ax.legend(title="Category")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="steelblue", s=100)

        ax.set_title(f"{method_clean.upper()} Dimensionality Reduction")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set matplotlib figure on session.pending_figure per spec requirement
        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("reduce_dimensions")
        args_dict = {
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }

        summary = {
            "result_id": result_id,
            "method": method_clean,
            "coordinates": coords.tolist(),
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result("reduce_dimensions_tool", args_dict, summary)
        return summary

    return [reduce_dimensions_tool]
