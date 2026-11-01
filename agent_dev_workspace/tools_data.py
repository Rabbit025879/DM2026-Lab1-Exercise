import os
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


def make_tools(session: Any) -> list:
    """Factory function that constructs tools with access to the shared session object.

    Args:
        session: Shared SessionState instance.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize document text in session.dataframe into unigram tokens.

        Lowercases each document's text and splits it into individual word tokens using
        nltk.word_tokenize. Stores the token lists in session.dataframe['unigrams'] and
        session.artifacts['unigrams'].

        Returns:
            Dict containing result_id, n_documents, and sample_tokens.
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

        # Tokenize each row's text using nltk.word_tokenize on lowercased text
        unigrams = []
        for text in df["text"]:
            if not isinstance(text, str) or not text.strip():
                unigrams.append([])
            else:
                tokens = nltk.word_tokenize(text.lower())
                unigrams.append(tokens)

        # Update dataframe and artifacts in session
        df["unigrams"] = unigrams
        session.artifacts["unigrams"] = unigrams

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "total_tokens": sum(len(toks) for toks in unigrams),
            "sample_tokens": unigrams[:3] if unigrams else [],
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List files and directories in a specified relative subdirectory of the workspace.

        Args:
            subdirectory: Subdirectory path relative to workspace/repo root (default: "").

        Returns:
            Dict containing result_id, subdirectory, directories, and files.
        """
        target_dir = Path(subdirectory) if subdirectory else Path(".")

        if not target_dir.exists():
            return {
                "error": f"Directory '{subdirectory}' does not exist."
            }

        if not target_dir.is_dir():
            return {
                "error": f"Path '{subdirectory}' is not a directory."
            }

        directories = []
        files = []

        for entry in target_dir.iterdir():
            # Skip hidden files and directories
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, max_text_length: int = 150) -> Dict[str, Any]:
        """Inspect the first n rows and column metadata of session.dataframe.

        Truncates long text fields to max_text_length characters, provides text_length metadata,
        and excludes redundant derived columns such as 'unigrams' from the preview.

        Args:
            n: Number of sample rows to preview (default: 5). Must be a positive integer.
            max_text_length: Maximum characters to display for long text fields (default: 150).

        Returns:
            Dict containing result_id, total_rows, total_cols, columns, dtypes, and head preview.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Please run load_dataset_tool first."
            }

        if n <= 0:
            return {
                "error": f"Parameter 'n' must be a positive integer, got {n}."
            }

        if max_text_length <= 0:
            return {
                "error": f"Parameter 'max_text_length' must be a positive integer, got {max_text_length}."
            }

        df = session.dataframe
        preview_df = df.head(n).copy()

        # Exclude redundant derived unigrams column if present
        if "unigrams" in preview_df.columns:
            preview_df = preview_df.drop(columns=["unigrams"])

        preview_records: List[Dict[str, Any]] = []

        for _, row in preview_df.iterrows():
            record: Dict[str, Any] = {}
            for col in preview_df.columns:
                val = row[col]

                # Special handling for text column
                if col == "text":
                    text_str = str(val) if val is not None else ""
                    record["text_length"] = len(text_str)
                    if len(text_str) > max_text_length:
                        record["text"] = text_str[:max_text_length] + "..."
                    else:
                        record["text"] = text_str
                # Handling strings longer than max_text_length
                elif isinstance(val, str):
                    if len(val) > max_text_length:
                        record[col] = val[:max_text_length] + "..."
                    else:
                        record[col] = val
                # Handling list columns
                elif isinstance(val, list):
                    record[col] = f"List[{len(val)} items]"
                else:
                    record[col] = val

            preview_records.append(record)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": preview_records,
        }

        session.store_result("inspect_data_tool", {"n": n, "max_text_length": max_text_length}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text values in session.dataframe.

        Identifies rows where text is null, NaN, or whitespace-only.

        Returns:
            Dict containing result_id, total_documents, missing_count, missing_indices, and has_missing.
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

        missing_indices = []
        for idx, val in enumerate(df["text"]):
            if pd.isna(val) or val is None or not str(val).strip():
                missing_indices.append(idx)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices,
            "has_missing": len(missing_indices) > 0,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for and optionally drop duplicate documents in session.dataframe.

        When drop is True, removes all copies of duplicated rows using keep=False.

        Args:
            drop: If True, removes all duplicate rows from session.dataframe (default: False).

        Returns:
            Dict containing result_id, initial_documents, duplicate_count, dropped, and remaining_documents.
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

        initial_count = len(df)
        # Identify duplicate rows (excluding keep='first' to find duplicate count)
        duplicate_mask = df.duplicated(subset=["text"], keep="first")
        duplicate_count = int(duplicate_mask.sum())

        if drop:
            # Must use keep=False to drop all copies of duplicated rows
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        remaining_count = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "initial_documents": initial_count,
            "duplicate_count": duplicate_count,
            "dropped": drop,
            "remaining_documents": remaining_count,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """Subsample session.dataframe down to n rows using pandas sampling.

        Args:
            n: Number of rows to sample (default: 5).
            random_state: Seed for random number generator to ensure reproducibility (default: 42).

        Returns:
            Dict containing result_id, initial_documents, sampled_documents, sampled_indices, and random_state.
        """
        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Please run load_dataset_tool first."
            }

        df = session.dataframe
        if n <= 0:
            return {
                "error": f"Parameter 'n' must be a positive integer, got {n}."
            }

        if n > len(df):
            return {
                "error": f"Requested sample size n={n} exceeds total available rows {len(df)}."
            }

        initial_count = len(df)
        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update working dataframe in session
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "initial_documents": initial_count,
            "sampled_documents": n,
            "sampled_indices": sampled_indices,
            "random_state": random_state,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute text length statistics and generate a box plot grouped by category.

        Computes overall and per-category descriptive statistics on text character counts,
        and sets a seaborn box plot on session.pending_figure.

        Returns:
            Dict containing result_id, overall_stats, and per_category_stats.
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

        # Calculate character count per document
        df["text_length"] = df["text"].apply(lambda x: len(x) if isinstance(x, str) else 0)

        # Compute overall descriptive statistics
        overall_stats = df["text_length"].describe().to_dict()

        # Compute per-category descriptive statistics
        per_category_stats: Dict[str, Dict[str, float]] = {}
        category_col = "category_name" if "category_name" in df.columns else "label"

        if category_col in df.columns:
            for cat, group in df.groupby(category_col):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Generate box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        if category_col in df.columns:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Text Length Distribution by Category")
        plt.tight_layout()

        # Set figure on session.pending_figure as required by spec
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
