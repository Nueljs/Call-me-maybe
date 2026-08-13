```markdown
*This project has been created as part of the 42 curriculum by macerver.*

# Call-me-maybe: Introduction to function calling in LLMs

## Description
This project explores the mechanics of connecting Large Language Models (LLMs) to external tools through function calling. Using a small-scale model (Qwen 0.6B), the goal is to translate natural language prompts into structured, schema-compliant JSON function calls. Instead of relying on the model's inherent ability to output correct JSON—which often fails in smaller models—this project implements a **constrained decoding** system. By manipulating logits at the tensor level, the system physically forces the LLM to adhere to a strict JSON structure, guaranteeing valid keys and selecting only from a pre-defined list of available functions.

## Instructions
This project uses `uv` for dependency management and a Makefile to streamline execution.

**Prerequisites:**
- Python 3.10+
- `uv` package manager installed

**Installation & Execution:**
1. Clone the repository and navigate to the root directory.
2. Install dependencies:
   ```bash
   make install

```

3. Run the main program:
```bash
make run

```


4. Run strict linters and type checkers (Flake8 & MyPy):
```bash
make lint-strict

```



## Algorithm Explanation

The core of the project is a constrained decoding engine built via a Finite State Machine (FSM) that masks tensor logits during token generation. The generation follows these states:

* **State 0:** Forces the opening bracket `{`. All other token logits are set to `-inf`.
* **State 1:** Forces the exact sequence `\n "name": `.
* **State 2 (Function Selection):** Dynamically restricts the next tokens to match only the valid paths of the loaded function names (acting as a Trie structure). Once a valid function name is completed, it transitions.
* **State 3:** Forces the exact sequence `\n "parameters": {`.
* **State 4 (Free Generation):** The LLM is allowed to generate parameter values freely. The system tracks `open_brackets` and cleanly terminates the generation when the global JSON object is closed (brackets reach 0).

## Design Decisions

* **Pydantic Validation:** Used heavily for loading inputs and validating the final output (`OutputResult`). This guarantees that even if the LLM output passes the JSON decoder, it still strictly conforms to the expected data schema before being saved.
* **Robust Error Handling:** Instead of patching LLM hallucinations with heuristics or string replacements, the system relies on standard `try...except` blocks (`json.JSONDecodeError`). This ensures the program remains resilient and never crashes unexpectedly, even when the LLM generates syntactically malformed parameter strings.
* **Typed Argument Parsing:** Arguments mapped from `argparse` are explicitly cast (e.g., `str(args.input)`) to satisfy strict static type checking (`mypy --strict`).

## Performance Analysis

* **Accuracy:** The system achieves a **~90.9% accuracy rate** (10 out of 11 predefined test prompts successfully mapped and extracted).
* **Speed:** By using constrained decoding directly on the logits in a single pass, the system avoids costly fallback retries or recursive validation loops, keeping inference fast.
* **Reliability:** The FSM guarantees that 100% of the generated outputs have the correct fundamental structure (`name` and `parameters` keys), isolating any unpredictability strictly to the parameter values.

## Challenges Faced

1. **Internal Prompt Quotes:** Users providing input with mixed single and double quotes confused the LLM. *Solution:* Isolated the user input within the prompt using clear visual delimiters (`---`) to separate instructions from raw data.
2. **Regex Escaping Syntax:** The LLM correctly deduced complex parameters (like identifying `\d+` as the regex for numbers) but failed to double-escape the backslash (`\\d+`) as required by strict JSON standards, causing decode errors. *Solution:* Left the raw inference untouched to respect the "no heuristic magic" rule, relying on robust exception handling to catch the error, log it, and safely proceed to the next prompt.

## Testing Strategy

Validation was conducted through automated batch processing of predefined scenarios (`function_calling_tests.json`).

* **Runtime:** Processed multiple inputs sequentially to ensure state resets correctly between generations.
* **Static Analysis:** Codebase strictly adheres to PEP-8 (via `flake8`) and passes the highest level of static type checking (`mypy --strict` with flags forbidding untyped definitions and unused ignores).

## Example Usage

When running `make run`, the system processes the input file and logs the extraction in real-time:

```bash
$ make run
Running the program
uv run python -m src
Procesado: 'What is the sum of 2 and 3?' -> fn_add_numbers
Procesado: 'Greet shrek' -> fn_greet
Procesado: 'Reverse the string 'hello'' -> fn_reverse_string
Error processing the prompt: 'Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS'
Procesado: 'Replace all vowels in 'Programming is fun' with asterisks' -> fn_substitute_string_with_regex

🎉 ¡Proceso terminado! Archivo guardado en: data/output/function_calls.json

```

## Resources

* [Qwen Model Documentation](https://huggingface.co/Qwen)
* [Python typing module](https://docs.python.org/3/library/typing.html)
* [Pydantic Documentation](https://www.google.com/search?q=https://docs.pydantic.dev/)
* **AI Usage:** An AI assistant (Google Gemini) was used during development as an interactive sounding board to debug tensor dimension masking, refine the logic of the FSM for token generation, and structure the robust Pydantic schemas. It did not write the final core FSM logic, but helped troubleshoot edge cases like the JSON escaping behaviors.

```

```