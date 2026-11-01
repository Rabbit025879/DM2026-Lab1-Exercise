import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def setup_exploration_test():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM (build_dtm_tool in tools_dtm)
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    exploration_tools = make_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    correlation_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    return session, cosine_tool, correlation_tool

def test_cosine_similarity_tool():
    session, cosine_tool, _ = setup_exploration_test()
    
    # Known Answer #16:
    # 0 vs 1 (both catA, near-identical): 0.9847
    # 0 vs 3 (catA vs catB, share only "always"): 0.0909
    # 3 vs 4 (both catB, near-identical): 0.9847
    # 0 vs 7 (catA vs empty-text row): 0.0
    
    res_0_1 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert pytest.approx(res_0_1["similarity"], abs=1e-3) == 0.9847

    res_0_3 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert pytest.approx(res_0_3["similarity"], abs=1e-3) == 0.0909

    res_3_4 = cosine_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert pytest.approx(res_3_4["similarity"], abs=1e-3) == 0.9847

    res_0_7 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert pytest.approx(res_0_7["similarity"], abs=1e-3) == 0.0

def test_feature_correlation_matrix_tool():
    session, _, correlation_tool = setup_exploration_test()
    
    res = correlation_tool.invoke({"top_k": 20})
    
    # Check figure creation
    assert session.pending_figure is not None
    
    # Known Answer #19:
    # Terms ordered by variance descending: ['alpha', 'gamma', 'delta', 'beta', 'always']
    expected_terms = ['alpha', 'gamma', 'delta', 'beta', 'always']
    assert res["terms"] == expected_terms
    
    expected_matrix = np.array([
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ])
    
    actual_matrix = np.array(res["correlation_matrix"])
    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)
