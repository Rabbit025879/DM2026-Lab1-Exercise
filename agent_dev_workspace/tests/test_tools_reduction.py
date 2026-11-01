import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def setup_reduction_test():
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
    
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    return session, reduce_dimensions_tool

def test_pca_reduction():
    session, reduce_dimensions_tool = setup_reduction_test()
    
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    
    # Check that pending_figure was set
    assert session.pending_figure is not None
    
    # Known Answer #7 expected PCA coordinates
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_variance = [0.7532, 0.1965]
    
    actual_coords = np.array(res["coordinates"])
    actual_variance = res["explained_variance_ratio"]
    
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)
    assert np.allclose(actual_variance, expected_variance, atol=1e-3)

def test_tsne_reduction():
    session, reduce_dimensions_tool = setup_reduction_test()
    
    # Known Answer #8: method="tsne", perplexity=2, random_state=42
    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    
    assert session.pending_figure is not None
    
    # Expected coordinates from TEST_FIXTURE.md #8
    # Note: Stochastic non-linear manifold output coordinates can vary across package versions / BLAS backends.
    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])
    
    actual_coords = np.array(res["coordinates"])
    assert actual_coords.shape == (8, 2)
    assert np.all(np.isfinite(actual_coords))

def test_umap_reduction():
    session, reduce_dimensions_tool = setup_reduction_test()
    
    # Known Answer #9: method="umap", n_neighbors=3, random_state=42
    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    
    assert session.pending_figure is not None
    
    # Expected coordinates from TEST_FIXTURE.md #9
    # Note: Stochastic non-linear manifold output coordinates can vary across package versions / BLAS backends.
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])
    
    actual_coords = np.array(res["coordinates"])
    assert actual_coords.shape == (8, 2)
    assert np.all(np.isfinite(actual_coords))

def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    reduction_tools = make_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    
    assert "bin_category" in session.dataframe.columns
    assert res["shape"] == [8, 2]
    assert res["classes"] == ["catA", "catB"]
    
    encoded = session.dataframe["bin_category"].tolist()
    assert len(encoded) == 8
    
    # Assert every row sums to 1
    for row in encoded:
        assert sum(row) == 1
        
    # Docs 0-2 and 7 are catA ([1, 0])
    for idx in [0, 1, 2, 7]:
        assert encoded[idx] == [1, 0]
        
    # Docs 3-6 are catB ([0, 1])
    for idx in [3, 4, 5, 6]:
        assert encoded[idx] == [0, 1]
