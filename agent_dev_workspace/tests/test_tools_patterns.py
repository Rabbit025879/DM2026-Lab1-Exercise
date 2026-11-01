import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def setup_test_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    
    return session, mine_patterns_tool

def test_mine_patterns_zero_terms_variance():
    session, mine_patterns_tool = setup_test_session()
    
    # Test zero terms handling for catA
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance"
    })
    assert res_catA.get("patterns") == []
    assert "no terms survived filtering" in res_catA.get("note", "").lower()

    # Test zero terms handling for catB
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "variance"
    })
    assert res_catB.get("patterns") == []
    assert "no terms survived filtering" in res_catB.get("note", "").lower()

def test_mine_patterns_fpgrowth_tfidf():
    session, mine_patterns_tool = setup_test_session()
    
    # Known Answer #17: catA with tfidf + fpgrowth, min_sup=1 -> {alpha} support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_catA = res_catA.get("patterns", [])
    assert len(patterns_catA) == 1
    assert "alpha" in patterns_catA[0]["pattern"]
    assert patterns_catA[0]["support"] == 3

    # Known Answer #17: catB with tfidf + fpgrowth, min_sup=1 -> {gamma} support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_catB = res_catB.get("patterns", [])
    assert len(patterns_catB) == 1
    assert "gamma" in patterns_catB[0]["pattern"]
    assert patterns_catB[0]["support"] == 4

def test_mine_patterns_topk_tfidf():
    session, mine_patterns_tool = setup_test_session()
    
    # Known Answer #20: catA with tfidf + topk, k=1 -> {alpha} support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_catA = res_catA.get("patterns", [])
    assert len(patterns_catA) == 1
    assert "alpha" in patterns_catA[0]["pattern"]
    assert patterns_catA[0]["support"] == 3

    # Known Answer #20: catB with tfidf + topk, k=1 -> {gamma} support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_catB = res_catB.get("patterns", [])
    assert len(patterns_catB) == 1
    assert "gamma" in patterns_catB[0]["pattern"]
    assert patterns_catB[0]["support"] == 4

def test_mine_patterns_maxfpgrowth_tfidf():
    session, mine_patterns_tool = setup_test_session()
    
    # Known Answer #20: MaxFPGrowth returns 0 patterns for single-term transactions
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catA.get("patterns") == []

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catB.get("patterns") == []
