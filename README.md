*This project has been created as part of the 42 curriculum by macerver*

# Call Me Maybe

## Description

Call Me Maybe is a function calling project focused on using a small language model to translate natural language requests into structured function calls.

The project uses the **Qwen/Qwen3-0.6B** model and constrained decoding to generate JSON function calls based on a set of available function definitions.

For example, given:

> What is the sum of 2 and 3?

the program should generate a structured call such as:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The main goal of the project is not to obtain the final answer to the user's question, but to determine **which function should be called and which arguments should be passed to it**.

The implementation uses token-level constrained decoding to restrict the language model to valid outputs instead of relying only on prompting.

---

## Features

* Natural language to structured function calling.
* Constrained token generation using the model vocabulary.
* Function selection based on the available function definitions.
* Support for:

  * `number`
  * `string`
  * `boolean`
* JSON output generation.
* Validation of input data using Pydantic.
* Error handling for invalid or missing input files.
* Reproducible execution using the provided LLM SDK and Qwen3-0.6B model.

---

## Algorithm Explanation

The generation process starts by building a prompt containing:

1. The available function definitions.
2. The user's natural language request.
3. An instruction asking the model to generate a JSON function call.

The prompt is then tokenized and passed to the LLM.

Instead of allowing the model to freely generate any token, the `Generator` restricts the possible next tokens depending on the current generation state.

The main generation states are:

```text
START
  ↓
NAME_KEY
  ↓
FUNC_NAME
  ↓
PARAM_KEY
  ↓
PARAM_NAME
  ↓
PARAM_VALUE
  ↓
END
```

### Function selection

Available function names are tokenized in advance.

During the `FUNC_NAME` state, only tokens that belong to one of the valid function-name token paths are allowed.

The model therefore still decides which function to select using its logits, but it cannot generate a function name that does not exist in the provided definitions.

### Parameter generation

After selecting a function, its parameters are read from the corresponding function definition.

Each parameter is generated according to its declared type.

#### Numbers

Numbers are generated using a restricted token set containing numeric tokens.

The implementation also supports:

* an optional `-` at the beginning;
* one optional decimal point.

Scientific notation such as `1e5` is intentionally not handled in the current implementation.

Examples:

```text
2
-15
12.5
-3.14
```

#### Booleans

Boolean values are generated using two valid token paths:

```text
true
false
```

The model selects between these two possibilities using the logits produced by the LLM.

#### Strings

String generation is handled by a dedicated `StringDecoder`.

The decoder uses a finite-state machine to control valid JSON string contents.

The main states are:

```text
CONTENT
ESCAPE
UNICODE_1
UNICODE_2
UNICODE_3
UNICODE_4
CLOSED
```

This allows the decoder to handle normal characters, JSON escape sequences and Unicode escape sequences while preventing invalid string syntax.

A maximum generation length is also used to guarantee that string generation eventually terminates. When the limit is reached, the closing quote is forced.

### Token selection

At every generation step:

1. The LLM produces logits for the next token.
2. The decoder determines which tokens are valid in the current state.
3. Invalid tokens are ignored.
4. The valid token with the highest logit is selected.
5. The token is appended to the generated sequence.
6. The process continues until the required structure is complete.

This approach combines the language model's ability to interpret the user's request with deterministic constraints that control the generated structure.

---

## Design Decisions

### Use of constrained decoding

The project relies on constrained decoding instead of expecting the model to produce perfectly structured JSON on its own.

This is particularly important because the project uses a relatively small language model.

### Separate handling for each parameter type

Numbers, booleans and strings are generated using different strategies because each type has different structural constraints.

This keeps the implementation simple while allowing each generator to focus on its own validation rules.

### Function-name token paths

Function names are tokenized ahead of time and represented as token paths.

This allows the model to select between the available functions while guaranteeing that the generated function name belongs to the provided function definitions.

### Dedicated string decoder

Strings are more complex than numbers or booleans because they can contain escapes and Unicode sequences.

For this reason, string generation is implemented separately using a finite-state machine.

### Greedy token selection

At each generation step, the token with the highest logit among the allowed tokens is selected.

This keeps the implementation deterministic and avoids the additional complexity of sampling strategies.

### Maximum string length

A maximum token limit is used for string generation.

If the model does not naturally decide to close a string, the closing quote is forced at the limit. This prevents endless generation loops and guarantees that the generation process terminates.

---

## Performance Analysis

The implementation is designed to run on standard hardware using the Qwen3-0.6B model.

### Accuracy

During development, the provided test set was used to evaluate:

* function selection;
* number extraction;
* string extraction;
* boolean generation;
* JSON structure.

The current implementation successfully processes the complete provided test set and produces valid structured output for the tested cases.

Some complex regex prompts can still result in semantically imperfect regular expressions because the language model may repeat patterns before reaching the generation limit.

### Reliability

The main objective of constrained decoding is structural reliability.

The generator restricts token selection according to the expected JSON structure and parameter types, reducing the probability of malformed JSON.

### Speed

The model generates the response token by token and calculates logits for every generation step.

This provides strong control over the generated structure, at the cost of additional inference time compared with unrestricted generation.

The project is expected to process the provided test set within the required execution time on standard hardware.

---

## Challenges Faced

### Small language model

The Qwen3-0.6B model is relatively small and can sometimes produce repetitive or semantically imperfect outputs.

This was especially noticeable when generating complex regular expressions.

### Token-level constraints

A major challenge was that tokens are not always equivalent to individual characters.

A single token may represent multiple characters, which makes it necessary to reason about complete token contents rather than only individual characters.

### String termination

The language model does not always naturally decide when a string should end.

In particular, regex generation could enter repetitive loops.

A maximum token limit with a forced closing quote was introduced to guarantee termination and maintain valid JSON output.

### Numbers

Numbers required separate handling to prevent invalid sequences.

The current implementation therefore allows numeric tokens, an optional leading minus sign, and one decimal point.

### Boolean values

Boolean generation required explicit token paths because `true` and `false` must be generated as valid JSON literals rather than quoted strings.

---

## Testing Strategy

Testing was performed incrementally during development.

The main test cases included:

### Function selection

Examples included:

```text
What is the sum of 2 and 3?
Greet shrek
Reverse the string 'hello'
Calculate the square root of 144
```

### Numbers

Tested values include:

```text
2
265
144
```

Additional numeric support was implemented for:

```text
-12
12.5
-3.14
```

### Strings

Examples include:

```text
hello
world
Programming is fun
The cat sat on the mat with another cat
```

### Regular expressions

The implementation was tested with prompts involving:

* replacing numbers;
* replacing vowels;
* replacing specific words.

### Booleans

A temporary test function was added during development:

```json
{
  "name": "fn_check_status",
  "description": "Check whether a system is active.",
  "parameters": {
    "active": {
      "type": "boolean"
    }
  },
  "returns": {
    "type": "string"
  }
}
```

and tested with:

```text
Is the system active?
```

The generated result correctly produced a boolean value.

### Static analysis

The project was also checked using the required linting configuration.

`make lint` currently passes successfully.

---

## Example Usage

### Default execution

```bash
uv run python -m src
```

By default, the program reads:

```text
data/input/function_calling_tests.json
data/input/functions_definition.json
```

and writes:

```text
data/output/function_calling_results.json
```

### Custom input and output files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile

The project also provides Makefile commands for common operations:

```bash
make install
make run
make debug
make lint
make clean
```

---

## Project Structure

```text
src/
├── __init__.py
├── __main__.py
├── generator.py
├── prompt_builder.py
├── schemas.py
└── string_decoder.py
```

### `__main__.py`

Entry point of the application.

It loads the input data, initializes the model and generator, processes each prompt and writes the final JSON output.

### `generator.py`

Contains the main constrained decoding logic.

It is responsible for:

* function selection;
* JSON structure generation;
* parameter generation;
* numbers;
* booleans;
* integration with the string decoder.

### `prompt_builder.py`

Builds the prompt containing the available function definitions and the user's request.

### `schemas.py`

Contains Pydantic models and the input data loader.

### `string_decoder.py`

Implements the finite-state machine used for constrained JSON string generation.

---

## Requirements

* Python 3.10+
* `uv`
* `numpy`
* `pydantic`
* Provided `llm_sdk`
* Qwen/Qwen3-0.6B

The project must work with the provided Qwen/Qwen3-0.6B model.

---

## Resources

The following resources were useful for understanding the concepts involved in the project:

* [42 Curriculum](https://42.fr/)
* [Qwen](https://huggingface.co/Qwen)
* [Hugging Face](https://huggingface.co/)
* JSON specification and syntax documentation
* Python documentation
* Pydantic documentation
* Articles and tutorials about constrained decoding and function calling in language models

### AI Usage

AI tools were used as a development and learning aid throughout the project.

They were used to:

* understand constrained decoding concepts;
* understand tokenization and logits;
* debug Python code;
* reason about finite-state machines;
* investigate generation problems;
* review implementation decisions;
* suggest testing strategies;
* improve project documentation.

The final implementation was developed, tested and reviewed by the project authors.

---

## Error Handling

The project attempts to handle invalid or missing input files gracefully.

Examples include:

* missing input files;
* invalid JSON input;
* invalid function definitions;
* unsupported parameter types;
* invalid generation states.

Errors are reported to the user instead of being silently ignored whenever possible.

---

## Output Format

The program generates a JSON array where each result contains exactly:

```json
{
  "prompt": "Original user request",
  "name": "function_name",
  "parameters": {
    "parameter": "value"
  }
}
```

The output is written to:

```text
data/output/function_calling_results.json
```

---

## Conclusion

Call Me Maybe demonstrates how constrained decoding can be used to guide a small language model toward structured function calls.

Instead of trusting the language model to generate valid JSON by itself, the project combines LLM inference with token-level constraints and type-specific generation logic.

This approach provides a practical balance between the flexibility of a language model and the structural reliability required for machine-readable output.
