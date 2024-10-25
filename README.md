# reagency

This library contains a set of lightweight abstractions for building agent scaffolds that are easy to evaluate and maintain.

It emphasizes the use of functions, not classes.

## Features

- **Decorator-based API**—define arbitrarily complex scaffolds and evaluations by composing your own functions
- **Clean evaluation framework**—run evals with just a few lines of code, attaching hooks to function calls to inspect their behavior or to bootstrap new evaluation datasets, with heavy use of caching to eliminate redundant computation
- **Langfuse integration**—get observability for free
- **Minimalist inference API**—call models using a syntax inspired by Vercel's very elegant [AI SDK](https://sdk.vercel.ai/docs/introduction)

## Overview
- [Configuration](#configuration)
- [Tasks](#tasks)
- [Inference](#inference)
- [Datasets](#datasets)
- [Evaluation](#evaluation)

## Configuration

Initialize an `AI` object to manage your project's AI logic. Some notes:

- Use of Langfuse is optional
- Global concurrency limits are set on a per-model basis
- You can use either OpenAI or Anthropic models, or both

```python
# File: /your_project/ai.py

from pathlib import Path

from langfuse import Langfuse
from reagency import AI, AnthropicProvider, OpenAIProvider

PROJECT_ROOT = Path(__file__).parent

ai = AI(
    dataset_dir=PROJECT_ROOT / "datasets",  # where your datasets will be stored
    observability=Langfuse(  # can be omitted
        secret_key="...",
        public_key="...",
        host="...",
    ),
    providers=[
        OpenAIProvider(
            api_key="...",  # your OpenAI API key
            max_connections={  # maximum number of concurrent requests per model
                "DEFAULT": 10,
                "gpt-4o-mini": 30,
            },
        ),
        AnthropicProvider(
            api_key="...",  # your Anthropic API key
            max_connections={
                "DEFAULT": 10,
                "claude-3-5-sonnet": 5,
            },
        ),
    ],
)
```
By default API keys will be read from environment variables, but you can also pass them in directly.

## Tasks

The basic building block of the library is a **task**. A task is a function that makes one or more calls to an AI model. 

Declaring a function as a task enters it into a unified observability, caching, and evaluation ecosystem. Do so using the `@ai.task()` decorator:

```python
from your_project.ai import ai


@ai.task()
def some_task(some_input: str) -> str:
    pass  # insert some AI logic here
```

There are two constraints on tasks:
1. They must be [**pure functions**](https://en.wikipedia.org/wiki/Pure_function)
2. Their arguments and outputs must **adhere to the library's serialization protocol**. 

The automated serialization process requires that all task arguments, keyword arguments, and return values be one of the following data types:

- Primitives (e.g. `str`, `int`, `float`, `bool`)
- [Pydantic models](https://docs.pydantic.dev/latest/api/base_model/)
- A subclass of the library-provided `Serializable`, which must implement `model_dump_json()` and `validate_json()` methods that serialize and deserialize the object, respectively
- Collections of the above types (lists, dictionaries, sets, tuples)

> `Serializable` will check at evaluation-time that any serialization/deserialization methods you've implemented are lossless, and will raise an exception if they are not.

Here is how you might render a PDF as a JSON-serializable object:

```python
import base64
import json
from io import BytesIO

from PIL import Image
from reagency import Serializable

from your_project.ai import ai


class PDF(Serializable):
    """A JSON-serializable PDF document."""

    pages: list[Image.Image]

    def model_dump_json(self) -> str:
        pages_b64 = []
        for page in self.pages:
            with BytesIO() as buffer:
                page.save(buffer, format="PNG")
                pages_b64.append(base64.b64encode(buffer.getvalue()).decode())
        return json.dumps({"pages": pages_b64})

    @classmethod
    def validate_json(cls, json_str: str) -> "PDF":
        data = json.loads(json_str)
        pages = []
        for page_b64 in data["pages"]:
            image_bytes = base64.b64decode(page_b64)
            pages.append(Image.open(BytesIO(image_bytes)))
        return cls(pages=pages)


@ai.task()
def transcribe_pdf(pdf: PDF) -> str:
    pass  # insert some AI logic here
```

Note: the `@ai.task()` decorator also takes the following optional arguments:
- `capture_input: bool`--whether to capture the input data to Langfuse
- `capture_output: bool`--whether to capture the output data to Langfuse

## Inference

The library exposes a boilerplate-free wrapper around the OpenAI and Anthropic APIs. Its syntax is inspired by Vercel's very elegant AI SDK.

In the simplest case, you might just want to feed some model a user prompt and (optionally) a system prompt, and have it return a string using `generate_text`:

```python
@ai.task()
async def summarize(text: str) -> str:
    return await ai.generate_text(
        model="gpt-4o-mini",
        system="You are a helpful assistant.",
        prompt=f"""
            Please summarize the following text:

            {text}
            """,
        dedent=True,  # defaults to True, eliminating indents from all prompts using textwrap.dedent
        max_attempts=3,  # number of retries on failure, defaults to 3
    )
```

To phrase more complex requests, you may opt to pass the model a list of messages. The following uses the `PDF` model we defined earlier:

```python
@ai.task()
async def transcribe_pdf(pdf: PDF) -> str:
    return await ai.generate_text(
        model="gpt-4o-mini",
        messages=[
            ai.message.system("You are a helpful assistant."),
            ai.message.user(
                "Please transcribe the following PDF to Markdown:",
                ai.message.image(pdf.pages[0]),
            ),
        ],
    )
```
> If you pass a `messages` argument, an exception will be raised if you also pass a `system` or `prompt` argument.

To request a structured output from the model, you can use `generate_object` and pass a Pydantic model as the `type` argument.  

```python
class PDFMetadata(BaseModel):
    title: str | None
    author: str | None


@ai.task()
async def extract_pdf_metadata(pdf: PDF) -> PDFMetadata:
    return await ai.generate_object(
        model="gpt-4o",
        type=PDFMetadata,
        messages=[
            ai.message.system("You are a helpful assistant."),
            ai.message.user(
                "Extract metadata from the following article:",
                *[ai.message.image(page) for page in pdf.pages],
            ),
        ],
    )
```

## Datasets

The library exposes an ORM-like API for developing evaluation datasets. 

Simply subclass `Dataset` and provide the following: 
- A `NAME` attribute -- this will be used to namespace different versions of the dataset
- Two type arguments, which will be used internally for typing and validating your dataset -- the first one is used for your _data_, the second is for your _targets_

Here's an example:

```python
from pydantic import BaseModel
from reagency import Dataset


class Invoice(BaseModel):
    text: str


class Targets(BaseModel):
    is_corrupted: bool  # True if the invoice data is corrupted, False otherwise
    total_cost: float | None  # The total cost of the invoice, or None if it's corrupted


class InvoiceDataset(Dataset[Invoice, Targets]):
    NAME = "invoices"


# save a dataset split -- targets can be attached now or later (you'll see how in the next section)
InvoiceDataset.save(
    split="train",
    data=[Invoice(...), Invoice(...)],
)

# load a dataset split
dataset = InvoiceDataset.load("train")

for data, targets in dataset:
    print(data, targets)
```

## Evaluation

The evaluation API uses hooks to give you precise control over your agent's computation graph.

You can run evaluations either from a Jupyter cell or from the CLI. 

First let's define a simple set of tasks, riffing off of the invoice data structure we defined in the `Dataset` section:

```python
@ai.task()
async def process_invoice(invoice: Invoice) -> float | str:
    looks_fine = await check_integrity(invoice)

    if not looks_fine:
        return await generate_error_report(invoice)

    return await extract_total_cost(invoice)


@ai.task()
async def check_integrity(invoice: Invoice, model: str = "gpt-4o-mini") -> bool:
    return await ai.generate_object(
        model=model,
        type=bool,
        prompt=f"Return True if the invoice looks uncorrupted: {invoice.text}",
    )


@ai.task()
async def generate_error_report(invoice: Invoice) -> str:
    return await ai.generate_text(
        model="gpt-4o",
        prompt=f"Write an error report for this corrupted invoice: {invoice.text}",
    )


@ai.task()
async def extract_total_cost(invoice: Invoice, model: str = "gpt-4o") -> float:
    return await ai.generate_object(
        model=model,
        type=float,
        prompt=f"Extract the total cost from this invoice: {invoice.text}",
    )
```

The first thing we'll want to do is bootstrap targets for our `InvoiceDataset`. This is easy to do using the hooks system. 

We will use hooks to:
1. Modify the `check_integrity` and `extract_total_cost` tasks to use the `o1-preview` model, which is the most expensive and capable model available
2. Tap into the execution of these functions to write the results to the dataset as target labels

```python
dataset = InvoiceDataset.load("development")

@ai.hook(check_integrity, model="o1-preview")
def hook_check_integrity(input, output, targets):
    dataset.targets[input].is_corrupted = output

@ai.hook(extract_total_cost, model="o1-preview")
def hook_extract_total_cost(input, output, targets):
    dataset.targets[input].total_cost = output

ai.eval(
    dataset=dataset,
    main=process_invoice, # the `main` fn will be executed once per dataset row
    hooks=[hook_check_integrity, hook_extract_total_cost],
)
```

Now that we have labels, we can evaluate the `check_integrity` and `extract_total_cost` tasks as they were originally defined. 

We can also add a `report` function to the `ai.eval()` call to generate a Markdown report summarizing the results, which will be written to `evals/runs/<run_id>/report.md`. Here we will only write a simple report, but one common pattern is to deploy language models to inspect your agent's behavior and summarize their findings for you.

Again, hooks provide an expressive way to write scoring logic, by reading state from parent scope:

```python
check_integrity_scores = []
extract_total_cost_scores = []

@ai.hook(check_integrity, model="o1-preview")
def hook_check_integrity(input, output, targets):
    score = output == targets.is_corrupted
    check_integrity_scores.append(score)

@ai.hook(extract_total_cost, model="o1-preview")
def hook_extract_total_cost(input, output, targets):
    score = output - targets.total_cost
    extract_total_cost_scores.append(score)

def report():
    return f"""
    check_integrity (% correct): {sum(check_integrity_scores) / len(check_integrity_scores)}
    extract_total_cost (avg. error): {sum(extract_total_cost_scores) / len(extract_total_cost_scores)}
    """

ai.eval(
    dataset=dataset,
    main=process_invoice,
    hooks=[hook_check_integrity, hook_extract_total_cost],
    report=report,
)
```
