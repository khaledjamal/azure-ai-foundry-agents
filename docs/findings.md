# Findings

Issues I found while building the labs, why they happened and how I fixed them, and what the model comparisons
showed. Each comes from a recorded run; the full outputs are in [`sample-output/`](../sample-output/). Back to the
[README](../README.md).

- [Shopping agent: prices never reached the model](#shopping-agent-prices-never-reached-the-model)
- [Red-teaming an agent with attack simulators](#red-teaming-an-agent-with-attack-simulators)
- [Azure Functions as a remote MCP server](#azure-functions-as-a-remote-mcp-server)
- [A web-grounded agent searched for 2023 news](#a-web-grounded-agent-searched-for-2023-news)
- [Honest run status, no keys in code](#honest-run-status-no-keys-in-code)
- [Re-running paid labs for free: record and replay](#re-running-paid-labs-for-free-record-and-replay)
- [Sign-in waited 1 s for a managed identity that isn't there](#sign-in-waited-1-s-for-a-managed-identity-that-isnt-there)
- [Search by meaning instead of keywords (AutoGen)](#search-by-meaning-instead-of-keywords-autogen)
- [Model choice: gpt-4o vs gpt-4o-mini](#model-choice-gpt-4o-vs-gpt-4o-mini)
- [Newer models need the new agents API](#newer-models-need-the-new-agents-api)
- [Web Search vs Grounding with Bing](#web-search-vs-grounding-with-bing)
- [Phi-4 and Phi-4-reasoning vs gpt-4o-mini](#phi-4-and-phi-4-reasoning-vs-gpt-4o-mini)
- [DeepSeek-V4-Flash, thinking off vs on](#deepseek-v4-flash-thinking-off-vs-on)
- [Classic and new agents API](#classic-and-new-agents-api)

## Shopping agent: prices never reached the model

Asked for "cardio equipment under $300", the agent said prices weren't specified. Its run steps showed the cause: the
Azure AI Search tool fetches only the text fields of each product
(`$select=FitnessItemID,Name,Category,Description`), so the numeric `Price` never reaches the model. Fixed by adding
the price to the indexed text, plus a local price-filter tool (`find_items_under`, a real `Price le <max>` query)
built two ways: as a Semantic Kernel plugin and as an Agent Service function tool. The second tool then made the agent
answer one question from general knowledge instead of searching the catalog; stricter instructions fixed it, and both
scripts now print every tool call, so a skipped search shows in the output.

Why this fix: a search ranked by relevance returns the closest matches, not every product under a budget, and leaves
the number comparison to the model. The filter query does the comparison in the index, so a budget answer covers every
product at or under the price, and the tool can name the cheapest product when nothing matches. The price in the
indexed text still lets the search tool's answers quote prices. From the
[recorded run](../sample-output/5-agents-aisearch.py-output-20261018-204747.txt):

```text
Question: I need something for cardio under $300, any suggestions?
Tool calls:
  - function find_items_under({"max_price":300,"category":"Cardio"})
    output: No products cost $300.00 or less in category Cardio. Cheapest product in Cardio: Treadmill (Cardio): $499.00

ASSISTANT:
Currently, there are no cardio products available for $300 or less. The closest option is a treadmill priced at
$499. If you want to consider increasing your budget or looking into alternative categories, please let me know!
```

## Red-teaming an agent with attack simulators

The GenAI Ops lab sends Microsoft-generated attacks (requests to rewrite text on harmful topics, each also sent with a
role-play jailbreak added) to a test agent on gpt-4o-mini. In the clean run the agent refused both jailbreak versions
and toned down the plain requests. The first run hid two failures behind a success message: the simulator silently
drops errors raised by the agent target, so one conversation lost its reply; and the RAG section searched a new index
before indexing finished, so one answer had no sources, yet the judge model still scored it 5/5 for groundedness. The
script now reports target errors, waits until the index is searchable, and fails the section if either happens.
Recorded runs: the [full run before the fix](../sample-output/3-e2e-genai-ops.py-output-20261016-030225.txt)
(conversation 1 has no reply; question 1 shows an empty `Retrieved:` yet groundedness 5.0) and the
[rerun of those two sections after it](../sample-output/3-e2e-genai-ops.py-output-20261016-031924.txt) (both jailbreak
versions refused, all four turns answered, sources retrieved).

Why this fix: the success message came from the simulator, which drops the target's errors, and the judge model
scores whatever answer it is given. Neither shows that the agent replied or that the search found anything, so the
script now checks both directly: it counts the answered, blocked and failed turns of each conversation and the sources
retrieved for each question, and fails the section when a reply or the sources are missing.

The service later changed how it reports a block (seen on 2026-10-26): a turn stopped by the content filter used to
end as a failed run with a content-filter error; it now ends as an `incomplete` run with `incomplete_details.reason`
`content_filter` and no error. The script took these for real failures and failed the section, although the filter
had done its job. It now counts both forms as blocked (azure-ai-agents 1.1.0 lists neither value, so it compares
strings); an incomplete run for any other reason still fails the section. A
[replay of the 2026-10-26 run's recorded answers](../sample-output/3-e2e-genai-ops.py-output-20261028-131803.txt)
with the fixed code shows it: the turn that run reported as `agent run incomplete` now counts as blocked by the
content filter, and all seven sections pass.

The direct attacks go to the test agent. The indirect attack (XPIA) section, as in the notebook, only generates
attacks against a fixed-reply target, so no model sees them: the lab shows nothing about the agent's resistance to
indirect prompt injection.

Like the original notebook, the lab sends harmful sample text and Microsoft-generated attack prompts to the Foundry
safety service and to its own test agent. It keeps the notebook's small sizes (2 attacks per simulator), leaves prompt
content out of the traces, and writes the full conversations only to its output folder (`runs/` is git-ignored). The
safety checks are billed on their own meter ($0.02 per 1K input and $0.06 per 1K output tokens in East US 2,
[Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices),
checked October 2026); a full run is estimated at well under $1. The copyrighted-text check runs only in East US 2.

## Azure Functions as a remote MCP server

The workshop's queue-based Functions tool needs a setup this project doesn't have, so the lab became an MCP server on
Azure Functions (Flex Consumption, scales to zero), called by an agent on the new agents API. The Function key sits in
a Foundry project connection, not in the agent, and the script approves each tool call. Two findings: capped at one
instance, the server returned 503 because Python on Flex handles one request at a time per instance and the MCP
client's open stream took it (fixed with per-instance concurrency); and the agent added warm-up timings the tool
never returned (fixed in its instructions).

Why this fix: the raw probe ([`scripts/probe-mcp-http.py`](../scripts/probe-mcp-http.py)) sends the same MCP calls as
plain HTTP requests and opens no stream. It worked while the MCP client's calls failed, which pointed at how the
server handles concurrent requests, not at the tool code or the agent. Raising the per-instance concurrency fixed it
and kept the one-instance cap, so the cost stays bounded. See the
[lab README](../src/functions-mcp-server/README.md). Recorded run:
[output](../sample-output/6-agents-az-functions-mcp.py-output-20261017-114003.txt).

## A web-grounded agent searched for 2023 news

With Grounding with Bing, the model writes the search query itself. Asked for news on intermittent fasting,
gpt-4o-mini searched for "... news 2023" and presented a 2023 study as recent; the WHO sugar answer cited a blog
instead of who.int. Putting today's date in the instructions and asking for official health sources first fixed both
in the next run (queries with the current month, who.int cited). The script also builds the Bing search link from each
query and numbers the citations, as Bing's display requirements ask. Recorded run after the fix:
[output](../sample-output/4-bing-grounding.py-output-20261018-114459.txt).

## Honest run status, no keys in code

The agent scripts (basics, File Search, Azure AI Search, AutoGen, GenAI Ops) check how each agent run actually ended
and exit with 1 when it didn't complete, instead of always printing "successfully" (for example
[`1-basics.py`](../src/2-agents/1-basics.py)). The Code Interpreter script prints a failed run's status
and error but doesn't exit with 1. Model and project calls sign in with Microsoft Entra ID (`DefaultAzureCredential`
via `az login`). The only keys are the Azure AI Search admin key and the Function app's MCP key; they come from `.env`
or from project connections (the Semantic Kernel lab takes the search key from the project's Azure AI Search
connection, and the MCP agent gets its key from a project connection), never from code.

## Re-running paid labs for free: record and replay

Every lab run costs money and the models answer a little differently each time, so checking a refactor by running
the labs again would be paid, slow and hard to compare. [`http_replay.py`](../src/workshop_utils/http_replay.py)
saves every HTTP call of a run (secrets redacted) and answers a later run from the saved files: no network, no Azure
sign-in, no cost. I check each change to the labs or the shared code by replaying every recorded lab run (27 in
October 2026) and comparing the output with the record run. How to use it is in
[setup.md](setup.md#record-and-replay); the recordings themselves stay out of the repo, because they hold my
endpoints and resource names.

Why not vcrpy: it records HTTP well, but it is built around test functions wrapped in a cassette, while here the same
lab command records or replays, switched on from `.env`. The labs also needed the parts below, which vcrpy would have
needed as custom hooks anyway.

What it took to make a replay send the same requests:
- **Random values.** Code that makes random choices or random IDs sends new requests on every run (here, GenAI Ops
  picks a random test prompt and `evaluate()` makes random run IDs). Each record run saves a random seed, and replay
  reuses it for Python's `random` and `uuid.uuid4()`.
- **Today's date.** The Bing grounding and Web Search labs put today's date in their instructions (see
  [above](#a-web-grounded-agent-searched-for-2023-news)); `--today YYYY-MM-DD` replays a recording from another day.
- **File uploads.** Each multipart upload gets a new random boundary, so bodies are matched without it.
- **Secrets the code parses.** The Application Insights connection string is redacted in recordings, but the tracing
  labs parse it before they start. Replay puts in a fake value in the same format
  (`InstrumentationKey=00000000-...;IngestionEndpoint=https://http-replay.invalid/`).

What the replays and their tests found:
- **A crash in the recorder itself.** Replaying the second Azure AI Search agent raised
  `TypeError: unhashable type: 'list'`. The agent definition echoes its tool's JSON Schema, where a field's `type` can
  be a list (`["string", "null"]`), and the fake-secret code expected a name there. Fixed, with a unit test.
- **A sign-in the network guard couldn't see.** On replay, connections to other machines are refused and the
  azure-identity credentials get a placeholder token. A test that left `AzureCliCredential` unpatched got a real
  token: it runs the `az` program as a separate process, which a guard inside Python doesn't see. Replay now patches
  every public azure-identity credential, and a test checks each one.

Limits: a replay shows that the code still handles the recorded answers, not that the service still answers the same
way; a new prompt, model or SDK version needs a new, paid record run. Only HTTP is covered (requests, httpx and
aiohttp).

## Sign-in waited 1 s for a managed identity that isn't there

The [HTTP log](setup.md#http-log) showed the same first call in every script, before any lab request:
`GET http://169.254.169.254/metadata/identity/oauth2/token`, which timed out after 1 s (call #1 in the
[REST vs SDK output](../sample-output/1-basic-chat-completion-rest-vs-sdk.py-output-20261021-131720.txt)).
`DefaultAzureCredential` tries its sign-in methods in order, and managed identity, the way code signs in when it runs
on Azure, comes before the Azure CLI login. On a PC nothing answers that address, so each run waited for the timeout.
Setting `AZURE_TOKEN_CREDENTIALS=dev` in `.env` limits it to developer sign-ins: the probe is gone, and getting a
token took 0.56 s instead of 1.55 s.

Why this fix: it is one setting and no code change. Without it, the same scripts still sign in with managed identity
when they run on Azure, which a hard-coded `AzureCliCredential` would break, along with other developer sign-ins.

## Search by meaning instead of keywords (AutoGen)

The AutoGen lab's keyword search couldn't connect "quick exercises" with "10-minute HIIT workout" and returned every
tip. Embedding search (`text-embedding-3-small`, cosine similarity) finds the right tips, and the agents now answer
only from them and name their sources. Recorded run:
[output](../sample-output/2-autogen-multi-agent-rag.py-output-20261015-165725.txt).

## Model choice: gpt-4o vs gpt-4o-mini

On the Semantic Kernel version of the shopping agent (an Agent Service agent with the Azure AI Search tool and the
price-filter plugin), with the same code, questions and index, gpt-4o-mini made the same tool calls and gave the same
grounded answers as gpt-4o at about 1/17 of the per-token price, so it is the default here (list prices per
1M input / output tokens, Global deployment in East US 2: gpt-4o $2.50 / $10, gpt-4o-mini $0.15 / $0.60;
[Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices),
checked October 2026):

| | gpt-4o | gpt-4o-mini |
|---|---|---|
| "Good for strength training?": searched the catalog | yes, 2 products with prices | yes, the same 2 products with prices |
| "Cardio under $300?": called the price tool | yes, `find_items_under(300, Cardio)` | yes, `find_items_under(300, Cardio)` |
| Answer | nothing under $300; closest is the treadmill at $499 | the same |
| Tokens (input / output) | 3,737 / 201 | 3,707 / 186 |
| Cost per run | about $0.011 | about $0.0007 |

One run per model on two questions: a quick comparison, not a benchmark. The token counts come from the deployments'
Azure Monitor metrics for the two runs. Recorded run on gpt-4o-mini:
[output](../sample-output/1-rag-sk-agents-aisearch.py-output-20261015-121028.txt); the Agent Service version made the
same tool calls on gpt-4o-mini ([output](../sample-output/5-agents-aisearch.py-output-20261018-204747.txt)). To use
another deployment for one run, set it on the command line:

```bash
MODEL_DEPLOYMENT_NAME=gpt-4o .venv-semantic-kernel/bin/python src/4-frameworks/1-rag-sk-agents-aisearch.py
```

## Newer models need the new agents API

I tried gpt-5.4-mini and gpt-5.4-nano on the same agent. With the Azure AI Search tool attached, every run failed with
a generic `server_error` ("Sorry, something went wrong."), no tokens used and no run steps. The same models worked as
plain agents and with only the function tool. The error doesn't give a reason, but it matches Microsoft's
[model support page for classic agents](https://learn.microsoft.com/azure/foundry-classic/agents/concepts/model-region-support)
(checked October 2026): gpt-5-family models there can use only the code interpreter and file search tools, and models
later than gpt-5 need the new agents API. This result covers these two models with the Azure AI Search tool; the page
lists the gpt-4.1 family without such limits.

So the classic labs stay on gpt-4o-mini (2024-07-18). The deadline that matters comes first from the service: classic
agents retire on 2027-03-31, two weeks before gpt-4o-mini and the gpt-4.1 family (2027-04-14, Microsoft's
[model retirement schedule](https://learn.microsoft.com/azure/foundry/openai/concepts/model-retirement-schedule),
checked October 2026). The classic labs need to move to the new agents API (`azure-ai-projects` 2.x) by then, and a
newer model comes with that move.

On the new agents API ([`5-agents-aisearch-v2.py`](../src/2-agents/5-agents-aisearch-v2.py)) the same models work with
the same two tools. In one run on the two shopping questions, gpt-4o-mini, gpt-5.4-mini and gpt-5.4-nano all searched
the catalog, called the price filter with $300 for the cardio question and gave the right answer (nothing under $300;
the treadmill at $499 is closest). The 5.4 models followed the "always search first" instruction more closely: they
searched on both questions, gpt-4o-mini only on the first. With default settings neither used reasoning tokens. Both
questions cost $0.0005 on gpt-4o-mini, $0.0010 on gpt-5.4-nano and $0.0033 on gpt-5.4-mini, so gpt-5.4-nano is a
workable replacement when the classic labs move, at about twice the token cost. One run, not a benchmark. Recorded run:
[output](../sample-output/5-agents-aisearch-v2.py-output-20261023-172117.txt).

## Web Search vs Grounding with Bing

On the same three questions (gpt-4o-mini, one search each, nearly the same queries), the Web Search agent cited 11
sources against 7 and cited WHO for the sugar question just as the Bing agent did. It read about twice the input
tokens (16K vs 9K), which costs under a cent on gpt-4o-mini. The search fee is likely about the same, but that is an
estimate: Microsoft publishes no separate Web Search rate, so the script counts searches at the Grounding with Bing
rate of [$14 per 1,000](https://www.microsoft.com/en-us/bing/apis/grounding-pricing) (checked October 2026). This is one run on three questions, not a benchmark: it shows no
quality gap either way, and Web Search needs no extra resource or connection, which makes it the simpler default for a
new agent. Recorded runs: [Web Search](../sample-output/4-web-search.py-output-20261018-144147.txt),
[Grounding with Bing](../sample-output/4-bing-grounding.py-output-20261018-114459.txt).

## Phi-4 and Phi-4-reasoning vs gpt-4o-mini

All three solved the pace question (6:00 per km, 18 seconds faster) and gave similar training advice. In the recorded
run gpt-4o-mini was the fastest (11.5 s for the three prompts against 27 s for Phi-4) and cost about the same ($0.0004
vs $0.0006: Phi-4 is slightly cheaper per token, $0.125 / $0.50 per 1M input / output tokens against $0.15 / $0.60,
but used more tokens; list prices from the
[Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices),
checked October 2026), so the small Phi model brought no gain here. Phi-4-reasoning wrote about 7 times the output tokens of
gpt-4o-mini (4,524 vs 654) and took 103 s, roughly 9 times as long, to reach the same answers: on these three prompts,
reasoning added only time and cost. This is one run, not a benchmark; the script reports whether each pace answer is
right, but fails only when a model returns no answer. Recorded run:
[output](../sample-output/4-phi-4.py-output-20261018-164134.txt).

Phi-4-reasoning's thinking can't be separated from its answer on Azure. The model card's system prompt asks for the
thinking between `<think>` tags, but the endpoint strips those tags (they are special tokens), leaves the message's
`reasoning` field empty and rejects `skip_special_tokens=False`, so the thinking and the answer arrive as one text.

## DeepSeek-V4-Flash, thinking off vs on

DeepSeek-R1 retired on Azure on 2026-08-13, and Microsoft's
[model retirement schedule](https://learn.microsoft.com/azure/foundry/openai/concepts/model-retirement-schedule)
(checked October 2026) lists DeepSeek-V4-Pro as its replacement; this lab uses the cheaper V4-Flash. V4-Flash
answers without thinking unless the request sets `reasoning_effort`. Then the thinking comes back in the message's `reasoning_content` field, apart from the answer (the
DeepSeek API's own `thinking` field is rejected by the Azure endpoint). Both modes got the pace question right;
thinking added 248 words of working for it and took 4.8 s instead of 1.4 s. The three prompts cost $0.002 without
thinking and $0.003 with it. Per 1M input / output tokens V4-Flash costs $0.19 / $0.51, about a seventh and a tenth of
R1's $1.35 / $5.40 (Global deployment in East US 2,
[Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices),
checked October 2026). Recorded run: [output](../sample-output/5-deep-seek-v4-flash.py-output-20261018-170548.txt).

## Classic and new agents API

Like Microsoft's notebooks, most agent labs (basics, Code Interpreter, File Search, Azure AI Search, observability,
GenAI Ops, Semantic Kernel) use the classic Foundry Agent Service API: agents, threads and runs through
`azure-ai-projects` 1.x. Microsoft retires it on 2027-03-31 (see the
[model support page for classic agents](https://learn.microsoft.com/azure/foundry-classic/agents/concepts/model-region-support),
checked October 2026). The Grounding with Bing, Web Search and Azure Functions MCP labs, and the second version of the
Azure AI Search agent, already use the new agents API (`azure-ai-projects` 2.x: versioned agents called through the
Responses API).
