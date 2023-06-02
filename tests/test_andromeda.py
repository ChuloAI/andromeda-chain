import pytest

from andromeda_chain import AndromedaChain, AndromedaPrompt, AndromedaResponse

def init_chain():
    return AndromedaChain()

def build_prompt():
    return AndromedaPrompt(
        name="hello",
        prompt_template="""Howdy: {{gen 'expert_names' temperature=0 max_tokens=300}}""",
        input_vars=[],
        output_vars=["expert_names"]
    )

def test_importable():
    assert init_chain()

def test_build_prompt():
    assert build_prompt()


# This is an integration test that everything works
def test_call_guidance():
    chain = init_chain()
    prompt = build_prompt()
    response = chain.run_guidance_prompt(prompt)
    assert isinstance(response, AndromedaResponse)
    assert "Howdy:" in response.expanded_generation
    assert response.result_vars["expert_names"]