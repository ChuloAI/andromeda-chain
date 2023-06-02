import pytest

from andromeda_chain import AndromedaChain, AndromedaPrompt, AndromedaResponse

def init_chain():
    return AndromedaChain()

def build_prompt():
    return AndromedaPrompt(
        name="hello",
        prompt_template="Howdy: {{gen 'response' max_tokens=8 stop='\n'}}",
        input_vars={},
        output_vars=["response"]
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
    assert "Howdy:" in responsen.expanded_generation
    assert response.result_vars["response"]