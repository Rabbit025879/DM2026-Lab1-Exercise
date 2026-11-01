import os
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth_algo
import PAMI.frequentPattern.topk.FAE as topk_algo
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth_algo


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 10,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns for a specific category using PAMI algorithms.

        Filters text for category_name, builds a fresh CountVectorizer(stop_words="english"),
        applies vocabulary percentile filtering (variance, tfidf, or term_frequency), converts
        surviving terms to a transaction database, and mines patterns using FPGrowth, TopK (FAE),
        or MaxFPGrowth.

        Args:
            category_name: Name of category to mine.
            filtering_method: Vocabulary filter method: "variance", "tfidf", or "term_frequency" (default: "tfidf").
            algorithm: Mining algorithm: "fpgrowth", "topk", or "maxfpgrowth" (default: "fpgrowth").
            min_sup: Minimum support threshold (required for "fpgrowth" and "maxfpgrowth").
            k: Top-k patterns count (required for "topk").

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, patterns list,
            and total_patterns found.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        df = session.dataframe
        label_col = "category_name" if "category_name" in df.columns else "label"
        if label_col not in df.columns or "text" not in df.columns:
            return {"error": "DataFrame missing required columns ('text' and category column)."}

        # Step 1: Filter to specific category
        cat_df = df[df[label_col] == category_name]
        if cat_df.empty:
            return {"error": f"No records found for category '{category_name}'."}

        texts = cat_df["text"].fillna("")

        # Fit fresh category-local CountVectorizer with English stopwords removed
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle empty vocabulary case
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "total_patterns": 0,
                "note": "no terms survived vectorization",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        dense_dtm = dtm.toarray()

        if dense_dtm.shape[1] == 0:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "total_patterns": 0,
                "note": "no terms survived vectorization",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Apply chosen vocabulary filtering method
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = dense_dtm.var(axis=0, ddof=0)
            else:
                scores = dense_dtm.sum(axis=0)

            low = np.percentile(scores, 5)
            high = np.percentile(scores, 95)
            kept_indices = np.where((scores > low) & (scores < high))[0]
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dtm).toarray()
            mean_tfidf = tfidf_matrix.mean(axis=0)
            cutoff = np.percentile(mean_tfidf, 20)
            kept_indices = np.where(mean_tfidf >= cutoff)[0]
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        # Step 3: Handle zero terms survived
        if len(kept_indices) == 0:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "total_patterns": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = feature_names[kept_indices]
        filtered_dtm = dense_dtm[:, kept_indices]

        # Convert DTM array to DataFrame for PAMI conversion
        binary_df = pd.DataFrame((filtered_dtm >= 1).astype(int), columns=kept_terms)

        # Drop rows with no active items
        binary_df = binary_df[binary_df.sum(axis=1) > 0]
        if binary_df.empty:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "total_patterns": 0,
                "note": "no valid transactions found",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert to transactional DB and mine via PAMI
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "transactional.csv")

        converter = db_converter.DF2DB(binary_df)
        converter.convert2TransactionalDatabase(oFile=db_path, condition=">=", thresholdValue=1)

        patterns = []
        if algorithm == "fpgrowth":
            s_val = min_sup if min_sup is not None else 1
            obj = fpgrowth_algo.FPGrowth(iFile=db_path, minSup=s_val)
            obj.startMine()
            p_df = obj.getPatternsAsDataFrame()
            if p_df is not None and not p_df.empty:
                for _, row in p_df.iterrows():
                    pattern_str = str(row["Patterns"]).strip()
                    p_tuple = tuple(pattern_str.split())
                    patterns.append({"pattern": p_tuple, "support": int(row["Support"])})
        elif algorithm == "topk":
            k_val = k if k is not None else 10
            obj = topk_algo.FAE(iFile=db_path, k=k_val)
            obj.startMine()
            p_df = obj.getPatternsAsDataFrame()
            if p_df is not None and not p_df.empty:
                for _, row in p_df.iterrows():
                    pattern_str = str(row["Patterns"]).strip()
                    p_tuple = tuple(pattern_str.split())
                    patterns.append({"pattern": p_tuple, "support": int(row["Support"])})
        elif algorithm == "maxfpgrowth":
            s_val = min_sup if min_sup is not None else 1
            obj = maxfpgrowth_algo.MaxFPGrowth(iFile=db_path, minSup=s_val)
            obj.startMine()
            p_df = obj.getPatternsAsDataFrame()
            if p_df is not None and not p_df.empty:
                for _, row in p_df.iterrows():
                    pattern_str = str(row["Patterns"]).strip()
                    p_tuple = tuple(pattern_str.split())
                    patterns.append({"pattern": p_tuple, "support": int(row["Support"])})
        else:
            return {"error": f"Unknown algorithm '{algorithm}'."}

        # Store pattern list in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        args_dict = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns,
            "total_patterns": len(patterns),
        }

        session.store_result("mine_patterns_tool", args_dict, summary)
        return summary

    return [mine_patterns_tool]
