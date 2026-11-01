from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filter terms in the global Document-Term Matrix (DTM) based on count variance.

        Calculates variance across documents for each feature in session.feature_matrix.
        Identifies near-constant features below threshold for removal and creates a full
        report sorted by variance for downstream visualization tools.

        Args:
            threshold: Variance threshold below which features are flagged for removal (default: 0.0).

        Returns:
            Dict containing result_id, total_features, kept_features_count, removed_features_count,
            threshold, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-Term Matrix not built. Please run build_dtm_tool first."
            }

        # Convert sparse matrix to dense array for feature-wise variance calculation
        dense_dtm = session.feature_matrix.toarray()
        variances = dense_dtm.var(axis=0, ddof=0)

        feature_names = np.array(session.feature_names)

        # Build report dataframe sorted by variance descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False)

        # Determine kept vs removed features
        kept_mask = report_df["variance"] >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        args_dict = {"threshold": threshold}

        summary = {
            "result_id": result_id,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_terms),
            "removed_features_count": len(removed_terms),
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        # Store full report for visualize_result_tool (1st col term, 2nd col variance)
        session.store_result("variance_filter_tool", args_dict, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Filter terms based on Pearson correlation with a target binary class indicator.

        Calculates Pearson linear correlation r between each term's count and a binary target
        indicator (1 for target_class, 0 otherwise).

        Args:
            target_class: Name of target class to correlate terms against.
            threshold: Minimum absolute correlation threshold |r| to keep terms (default: 0.0).

        Returns:
            Dict containing result_id, target_class, total_features, threshold, correlations,
            kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-Term Matrix not built. Please run build_dtm_tool first."
            }

        if session.labels is None:
            if session.dataframe is not None and "category_name" in session.dataframe.columns:
                session.labels = session.dataframe["category_name"].values
            elif session.dataframe is not None and "label" in session.dataframe.columns:
                session.labels = session.dataframe["label"].values
            else:
                return {
                    "error": "Document labels not found in session."
                }

        # Create binary target indicator y (1 if target_class else 0)
        y = (np.array(session.labels) == target_class).astype(float)
        
        dense_dtm = session.feature_matrix.toarray()
        n_features = dense_dtm.shape[1]

        pearson_r_list = []
        for i in range(n_features):
            x = dense_dtm[:, i].astype(float)
            # Calculate Pearson r using np.corrcoef
            if np.std(x) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                corr_matrix = np.corrcoef(x, y)
                r_val = float(corr_matrix[0, 1])
            pearson_r_list.append(r_val)

        feature_names = list(session.feature_names)

        # Build report dataframe sorted by absolute correlation descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_list
        })
        report_df["abs_r"] = report_df["pearson_r"].abs()
        report_df = report_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])

        # Determine kept vs removed features based on absolute threshold
        kept_mask = report_df["pearson_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()
        correlations = {term: round(r, 4) for term, r in zip(feature_names, pearson_r_list)}

        result_id = session.next_result_id("pearson_filter")
        args_dict = {"target_class": target_class, "threshold": threshold}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "threshold": threshold,
            "correlations": correlations,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        # Store full report for visualize_result_tool (1st col term, 2nd col pearson_r)
        session.store_result("pearson_filter_tool", args_dict, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Filter terms based on Spearman rank correlation with a target binary class indicator.

        Calculates Spearman monotonic correlation r between each term's count rank and a binary
        target indicator (1 for target_class, 0 otherwise).

        Args:
            target_class: Name of target class to correlate terms against.
            threshold: Minimum absolute correlation threshold |r| to keep terms (default: 0.0).

        Returns:
            Dict containing result_id, target_class, total_features, threshold, correlations,
            kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "Document-Term Matrix not built. Please run build_dtm_tool first."
            }

        if session.labels is None:
            if session.dataframe is not None and "category_name" in session.dataframe.columns:
                session.labels = session.dataframe["category_name"].values
            elif session.dataframe is not None and "label" in session.dataframe.columns:
                session.labels = session.dataframe["label"].values
            else:
                return {
                    "error": "Document labels not found in session."
                }

        # Create binary target indicator y (1 if target_class else 0)
        y = (np.array(session.labels) == target_class).astype(float)

        dense_dtm = session.feature_matrix.toarray()
        n_features = dense_dtm.shape[1]

        spearman_r_list = []
        for i in range(n_features):
            x = dense_dtm[:, i].astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                res = spearmanr(x, y)
                r_val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                if np.isnan(r_val):
                    r_val = 0.0
            spearman_r_list.append(r_val)

        feature_names = list(session.feature_names)

        # Build report dataframe sorted by absolute correlation descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r_list
        })
        report_df["abs_r"] = report_df["spearman_r"].abs()
        report_df = report_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])

        # Determine kept vs removed features based on absolute threshold
        kept_mask = report_df["spearman_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()
        correlations = {term: round(r, 4) for term, r in zip(feature_names, spearman_r_list)}

        result_id = session.next_result_id("spearman_filter")
        args_dict = {"target_class": target_class, "threshold": threshold}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "threshold": threshold,
            "correlations": correlations,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        # Store full report for visualize_result_tool (1st col term, 2nd col spearman_r)
        session.store_result("spearman_filter_tool", args_dict, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
