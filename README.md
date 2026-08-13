*This project has been created as part of the 42 curriculum by macerver.*

## Description

**call-me-maybe** is a function calling tool that turns natural language prompts into structured, machine-executable function calls, without relying on the LLM spontaneously producing valid JSON.

Given a prompt like `"What is the sum of 2 and 3?"`, the program does not answer `5`. Instead it selects the correct function from a provided catalog and extracts its arguments, producing:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The core problem this project solves is that small language models (here, `Qwen/Qwen3-0.6B`) are unreliable at producing valid JSON on their own. Instead of prompting-and-hoping, this project uses **constrained decoding**: at every generation step, the raw logits returned by the model are masked so that only tokens consistent with the required JSON structure can be selected. This guarantees syntactically valid, schema-compliant output on every run, regardless of how well the model "wants" to cooperate.

## Instructions

### Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

### Installation

```bash
make install
```

This runs `uv sync`, which creates a virtual environment and installs all dependencies, including the local `llm_sdk` workspace package.

### Running

```bash
make run
```

which is equivalent to:

```bash
uv run python -m src
```

By default the program reads `data/input/function_calling_tests.json` and `data/input/functions_definition.json`, and writes results to `data/output/function_calls.json`. All three paths can be overridden:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other Makefile targets

- `make debug` — runs the program under `pdb`
- `make lint` — runs `flake8` and `mypy` (non-strict flags)
- `make lint-strict` — runs `flake8` and `mypy --strict`
- `make clean` — removes `__pycache__` and `.mypy_cache`

## Algorithm Explanation

Generation happens token by token inside `Generator.generate()`. Rather than letting the model choose freely, each step goes through a small state machine that tracks how much of the target JSON skeleton has been emitted, and masks the logits accordingly before picking the next token:

1. **State 0 — open brace.** All logits are set to `-inf` except the token for `{`. The model is forced to open the JSON object.
2. **State 1 — `"name": ` key.** The literal token sequence for `\n "name": ` is forced token-by-token, regardless of what the model would have preferred.
3. **State 2 — function name selection.** This is the one truly "chosen" part of the process. Every function name from `functions_definition.json` is pre-encoded (as `"fn_name",`) into a token sequence, forming a set of candidate token paths (effectively a trie). At each step, only the *next* token of each still-alive path is left unmasked; every other token is set to `-inf`. The model's own logits (not a fixed value) are used to `argmax` among the currently valid candidates, so the function whose name tokens best match what the model would naturally generate is progressively selected as non-matching paths are eliminated.
4. **State 3 — `"parameters": {` key.** As in state 1, this literal sequence is forced token-by-token.
5. **After state 3.** Generation continues unconstrained by the mask, letting the model fill in the argument values. Brace balance (`open_brackets`, incremented/decremented by counting `{`/`}` in each decoded token) is tracked throughout, and generation stops as soon as the object closes and brackets return to zero — this is what caps `max_tokens` in practice for well-behaved outputs.

The vocabulary file (`get_path_to_vocab_file`) is loaded once at `Generator` construction time to build the token-to-id table used for masking.

## Design Decisions

- **State machine over a hand-rolled grammar engine.** Since the *shape* of the output object (`{"name": ..., "parameters": {...}}`) is fixed and known in advance, a small explicit state machine is simpler and easier to reason about than a general-purpose constrained-JSON grammar, at the cost of being specific to this exact schema.
- **Trie-based masking for function names.** Rather than generating the full name and validating it afterwards, invalid tokens are excluded *during* generation. This is what guarantees the emitted function name is always one of the ones defined in `functions_definition.json` — it cannot hallucinate a function that doesn't exist.
- **Pydantic for all data structures.** `FunctionDefinition`, `TestPrompt`, and `OutputResult` are all Pydantic models, giving input/output validation for free and satisfying the project's pydantic requirement.
- **`np.argmax` over the masked logits array.** Masked candidates (`-inf`) can never be selected by `argmax`, so masking and selection stay decoupled and easy to test independently.

## Performance Analysis

- **JSON validity:** structural tokens (`{`, key names, `parameters` key) are always forced, so the skeleton of the output is 100% syntactically predictable by construction.
- **Function selection accuracy:** constrained to the exact set of function names in `functions_definition.json`, so the model cannot invent a nonexistent function name.
- **Argument extraction:** currently generated without token-level type/schema constraints — the model fills the `parameters` object freely once inside state 3, relying on the prompt (function description + parameter types, built in `prompt_builder.build_context`) rather than logit masking. This is the main area where reliability still depends on the model's own behaviour rather than being structurally guaranteed.
- **Speed:** dominated by one forward pass per generated token (`get_logits_from_input_ids`); for the 11 sample prompts and short expected outputs this comfortably finishes well under the 5-minute budget on standard hardware.

## Challenges Faced

- **Balancing forced tokens vs. model freedom.** Forcing too much (e.g. exact key names) is straightforward, but deciding *where* to stop forcing and let the model reason (argument values) required tracking brace depth carefully so generation reliably terminates instead of running to `max_tokens`.
- **Token-level ambiguity in the function-name trie.** Since tokenization doesn't align with word boundaries, function names sharing a common prefix (e.g. two names both starting `fn_get_`) need their full token paths tracked and pruned step-by-step rather than compared as strings.
- **Determinism of state transitions.** Detecting "we've just left state 2" (i.e., `len(active_paths[0]) == 0`) needed care to make sure the trie was updated in lockstep with the actual token emitted, not just with the intended one.

## Testing Strategy

- Manual runs against the provided `data/input/function_calling_tests.json` and `data/input/functions_definition.json`, checking that:
  - the output file is valid, parseable JSON;
  - every `name` matches a function defined in `functions_definition.json`;
  - `parameters` contains all required argument keys.
- Edge-case prompts were included in the input set: ambiguous/ordinary numeric prompts, string-manipulation prompts requiring regex-shaped parameters, and prompts naming values embedded in quotes.
- `flake8` and `mypy` (via `make lint`) are run to catch style and typing issues before functional testing.

## Example Usage

```bash
$ make run
Processed: 'What is the sum of 2 and 3?' -> fn_add_numbers
Processed: 'Greet shrek' -> fn_greet
Processed: 'Reverse the string 'hello'' -> fn_reverse_string
...
```

Resulting `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Anthropic — Tool use / function calling overview](https://docs.claude.com/en/docs/build-with-claude/tool-use)
- [Hugging Face — Guiding Text Generation with Constrained Decoding](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)

**AI usage:** AI assistance was used to discuss and clarify concepts around the token-by-token generation pipeline (tokenization, logits, `max_tokens`), the difference between `.append()` and `.extend()` when building token sequences, and the structure of 2D encoding/token-id representations, in order to better understand the `Small_LLM_Model` SDK before implementing the `Generator` class. All design decisions and code were written and reviewed personally.