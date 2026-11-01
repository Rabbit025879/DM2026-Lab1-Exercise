import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Instantiate tools_dtm tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 3. Invoke build_dtm_tool with default settings
    res = build_dtm_tool.invoke({})

    # 4. Assert against Known Answer #4 in TEST_FIXTURE.md
    assert res["n_documents"] == 8
    assert res["vocabulary_size"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Assert session state updates
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts
    assert session.categories == ["catA", "catB"]


def test_term_frequency_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 2. Invoke term_frequency_tool
    res = term_frequency_tool.invoke({})

    # 3. Assert against Known Answer #14 in TEST_FIXTURE.md
    expected_freqs = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    assert res["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool():
    # Setup single session
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # First, build DTM
    build_dtm_tool.invoke({})

    # Test 1: Full matrix slice (defaults n_terms=20, n_documents=20)
    result_full = dtm_heatmap_tool.invoke({})

    assert "result_id" in result_full
    assert result_full["n_documents"] == 8
    assert result_full["n_terms"] == 5
    assert result_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    
    np.testing.assert_array_equal(result_full["dtm_slice"], expected_matrix)
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Test 2: Positional slice (n_terms=2, n_documents=3)
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert result_slice["n_documents"] == 3
    assert result_slice["n_terms"] == 2
    assert result_slice["terms"] == ["alpha", "always"]

    expected_matrix_slice = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    np.testing.assert_array_equal(result_slice["dtm_slice"], expected_matrix_slice)
