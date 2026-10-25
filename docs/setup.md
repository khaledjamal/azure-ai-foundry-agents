# Setup and lab details

Everything needed to run all the labs. The first lab needs only the README's
[Setup](../README.md#setup); this page adds the commands for every lab, the extra environments,
deployments and resources the other labs need, the settings and the issues I hit. Back to the
[README](../README.md).

## Labs

Run all commands from the project root. The scripts delete the agents, threads, files and indexes they create at the
end of the run. Their indexes have fixed names (for example `myfitnessindex`, `fitness-sk-index`, `ai-safety-index`),
so a run replaces an index of the same name. GenAI Ops keeps its index with `--keep-index`. The two Basic RAG scripts
always keep theirs: like the notebook, they rebuild it at the start of each run (`SEARCH_INDEX_NAME`, or
`healthtips-index` when unset) and keep it afterwards, so it can be inspected in the portal (on the free Azure AI
Search tier it costs nothing).

| Lab | Script | Workshop notebook | Sample output | What it shows | Command |
|---|---|---|---|---|---|
| Setup check | [`check-setup.py`](../src/check-setup.py) | [1-introduction](https://github.com/Azure/ai-foundry-workshop/tree/273c24c/1-introduction) | [output](../sample-output/check-setup.py-output-20261018-172359.txt) | read-only check of the three introduction notebooks' steps: settings, Entra ID sign-in, project deployments and connections, Azure AI Search; no model call | `uv run python src/check-setup.py` |
| Chat completion | [`1-basic-chat-completion-azure-openai.py`](../src/1-models/1-basic-chat-completion-azure-openai.py), [`-project-client.py`](../src/1-models/1-basic-chat-completion-project-client.py) | [1-basic-chat-completion.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/1-basic-chat-completion.ipynb) | [azure-openai](../sample-output/1-basic-chat-completion-azure-openai.py-output-20261016-070614.txt), [project-client](../sample-output/1-basic-chat-completion-project-client.py-output-20261014-094029.txt) | the same calls through `openai.AzureOpenAI` and through `AIProjectClient.get_openai_client()` | `uv run python src/1-models/1-basic-chat-completion-azure-openai.py` |
| Chat: REST vs SDK | [`1-basic-chat-completion-rest-vs-sdk.py`](../src/1-models/1-basic-chat-completion-rest-vs-sdk.py) | new | [output](../sample-output/1-basic-chat-completion-rest-vs-sdk.py-output-20261021-131720.txt) | the same chat request sent by hand with `requests` (URL, headers, JSON body) and through the `openai` SDK; the [HTTP log](#http-log) shows both requests and responses in full, then a table compares status, latency and tokens | `uv run python src/1-models/1-basic-chat-completion-rest-vs-sdk.py` |
| Embeddings | [`2-embeddings-azure-openai.py`](../src/1-models/2-embeddings-azure-openai.py), [`-project-client.py`](../src/1-models/2-embeddings-project-client.py) | [2-embeddings.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb) | [azure-openai](../sample-output/2-embeddings-azure-openai.py-output-20261014-094054.txt), [project-client](../sample-output/2-embeddings-project-client.py-output-20261014-094058.txt) | text embeddings with `text-embedding-3-small` | `uv run python src/1-models/2-embeddings-azure-openai.py` |
| Image embeddings | [`2-embeddings-image-embeddings.py`](../src/1-models/2-embeddings-image-embeddings.py) | [2-embeddings.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb) | [output](../sample-output/2-embeddings-image-embeddings.py-output-20261025-170122.txt) | Cohere `embed-v-4-0` puts an X-ray image and text captions in one vector space; the captions are ranked by cosine similarity to the image. Plain REST calls, because the `azure-ai-inference` SDK the notebook used was retired | `uv run python src/1-models/2-embeddings-image-embeddings.py` |
| Image generation | [`2-embeddings-image-generation.py`](../src/1-models/2-embeddings-image-generation.py) | [2-embeddings.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/2-embeddings.ipynb) | [output](../sample-output/2-embeddings-image-generation.py-output-20261018-133622.txt) | one image from the notebook's prompt on `gpt-image-2` (DALL-E 3 was retired on Azure), through the Azure OpenAI v1 endpoint with an Entra ID token; saved as PNG | `uv run python src/1-models/2-embeddings-image-generation.py` |
| Basic RAG | [`3-basic-rag-azure-openai.py`](../src/1-models/3-basic-rag-azure-openai.py), [`-project-client.py`](../src/1-models/3-basic-rag-project-client.py) | [3-basic-rag.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/3-basic-rag.ipynb) | [azure-openai](../sample-output/3-basic-rag-azure-openai.py-output-20261014-094211.txt), [project-client](../sample-output/3-basic-rag-project-client.py-output-20261014-094224.txt) | vector index in Azure AI Search, answers grounded in the top matches | `uv run python src/1-models/3-basic-rag-azure-openai.py` |
| Phi-4 vs gpt-4o-mini | [`4-phi-4.py`](../src/1-models/4-phi-4.py) | [4-phi-4.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/4-phi-4.ipynb) | [output](../sample-output/4-phi-4.py-output-20261018-164134.txt) | Phi-4, Phi-4-reasoning and gpt-4o-mini on the same three prompts (a training plan, an answer from a context snippet, a pace calculation with one right answer), with tokens, time and cost per model; the Phi models are deployed on the Foundry resource and called through the Azure OpenAI v1 endpoint, since the serverless `azure-ai-inference` route the notebook used was retired | `uv run python src/1-models/4-phi-4.py` |
| DeepSeek thinking off vs on | [`5-deep-seek-v4-flash.py`](../src/1-models/5-deep-seek-v4-flash.py) | [5-deep-seek-r1.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/5-deep-seek-r1.ipynb) | [output](../sample-output/5-deep-seek-v4-flash.py-output-20261018-170548.txt) | the notebook's travel and PostgreSQL prompts plus the pace question on DeepSeek-V4-Flash (DeepSeek-R1 is retired on Azure), each with thinking off and on (`reasoning_effort="high"`); the thinking arrives in its own `reasoning_content` field, so the notebook's `<think>` regex isn't needed | `uv run python src/1-models/5-deep-seek-v4-flash.py` |
| Agent basics | [`1-basics.py`](../src/2-agents/1-basics.py) | [1-basics.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/1-basics.ipynb) | [output](../sample-output/1-basics.py-output-20261014-094247.txt) | Foundry Agent Service: agent, thread, messages, run | `uv run python src/2-agents/1-basics.py` |
| Code Interpreter | [`2-code-interpreter.py`](../src/2-agents/2-code-interpreter.py) | [2-code_interpreter.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/2-code_interpreter.ipynb) | [output](../sample-output/2-code-interpreter.py-output-20261025-215724.txt) | agent analyzes a CSV and returns a chart file | `uv run python src/2-agents/2-code-interpreter.py --output-dir charts` |
| File Search | [`3-file-search.py`](../src/2-agents/3-file-search.py) | [3-file-search.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/3-file-search.ipynb) | [output](../sample-output/3-file-search.py-output-20261025-215612.txt) | vector store over Markdown files, answers with file citations | `uv run python src/2-agents/3-file-search.py` |
| Grounding with Bing | [`4-bing-grounding.py`](../src/2-agents/4-bing-grounding.py) | [4-bing_grounding.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/4-bing_grounding.ipynb) | [output](../sample-output/4-bing-grounding.py-output-20261018-114459.txt) | new agents API (`azure-ai-projects` 2.x) agent with the Bing grounding tool through a project connection; answers with numbered web citations and the Bing search link; queries leave the Azure compliance boundary (Bing's terms apply) | `.venv-agents-v2/bin/python src/2-agents/4-bing-grounding.py` |
| Web Search (next to the Bing lab) | [`4-web-search.py`](../src/2-agents/4-web-search.py) | new | [output](../sample-output/4-web-search.py-output-20261018-144147.txt) | the same three questions with the built-in `WebSearchTool`, which Microsoft now recommends first: no Grounding with Bing resource or project connection; it still runs on Grounding with Bing, so the same terms and compliance-boundary note apply | `.venv-agents-v2/bin/python src/2-agents/4-web-search.py` |
| Agent + Azure AI Search | [`5-agents-aisearch.py`](../src/2-agents/5-agents-aisearch.py) | [5-agents-aisearch.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/5-agents-aisearch.ipynb) | [output](../sample-output/5-agents-aisearch.py-output-20261018-204747.txt) | product search tool plus a local price function tool | `uv run python src/2-agents/5-agents-aisearch.py` |
| Agent + Azure AI Search, new API | [`5-agents-aisearch-v2.py`](../src/2-agents/5-agents-aisearch-v2.py) | [5-agents-aisearch.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/5-agents-aisearch.ipynb) | [output](../sample-output/5-agents-aisearch-v2.py-output-20261023-172117.txt) | the same agent on the new agents API (`azure-ai-projects` 2.x): one agent version per model, the script runs the price function and sends its result back through the Responses API; compares gpt-4o-mini, gpt-5.4-mini and gpt-5.4-nano (tool calls, tokens, time, cost) | `.venv-agents-v2/bin/python src/2-agents/5-agents-aisearch-v2.py --models gpt-4o-mini,gpt-5.4-mini,gpt-5.4-nano` |
| Agent + Azure Functions MCP server | [`6-agents-az-functions-mcp.py`](../src/2-agents/6-agents-az-functions-mcp.py), [`functions-mcp-server/`](../src/functions-mcp-server/) | replaces [6-agents-az-functions.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/2-agent_service/6-agents-az-functions.ipynb) | [output](../sample-output/6-agents-az-functions-mcp.py-output-20261017-114003.txt) | new agents API (`azure-ai-projects` 2.x) agent with an MCP tool to a Function app's two workout tools, key from a project connection, each call approved; see the [lab README](../src/functions-mcp-server/README.md) for deploying | `.venv-agents-v2/bin/python src/2-agents/6-agents-az-functions-mcp.py` |
| Observability | [`1-observability.py`](../src/3-quality/1-observability.py) | [1-Observability.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/3-quality_attributes/1-Observability.ipynb) | [output](../sample-output/1-observability.py-output-20261016-005619.txt) | OpenTelemetry traces of a chat call and a file-search agent (model, tokens, run status per step), to the console or to Application Insights and the Foundry portal's Tracing page | `uv run python src/3-quality/1-observability.py --destination azure-monitor` |
| Evaluation | [`2-evaluation.py`](../src/3-quality/2-evaluation.py) | [2-evaluation.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/3-quality_attributes/2-evaluation.ipynb) | [output](../sample-output/2-evaluation.py-output-20261014-103533.txt) | `azure-ai-evaluation`: F1, plus relevance and groundedness judged by a model; results in the Foundry portal | `uv run python src/3-quality/2-evaluation.py --output-dir evaluation-results` |
| GenAI Ops | [`3-e2e-genai-ops.py`](../src/3-quality/3-e2e-genai-ops.py) | [3-e2e-GenAI-Ops.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/3-quality_attributes/3-e2e-GenAI-Ops.ipynb) | [rerun after the fix](../sample-output/3-e2e-genai-ops.py-output-20261016-031924.txt) (sections 3 and 6), [full run before it](../sample-output/3-e2e-genai-ops.py-output-20261016-030225.txt) | safety evaluators (harm categories, copyrighted text), a direct attack simulator against a test agent, indirect attack (XPIA) generation against a fixed-reply target as in the notebook (no model sees those attacks), quality evaluation in the Foundry portal, RAG scored by a judge model, all traced to Application Insights | `uv run python src/3-quality/3-e2e-genai-ops.py --output-dir runs/genai-ops` |
| Semantic Kernel agent | [`1-rag-sk-agents-aisearch.py`](../src/4-frameworks/1-rag-sk-agents-aisearch.py) | [1-rag-sk-agents-aisearch.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/1-rag-sk-agents-aisearch.ipynb) | [output](../sample-output/1-rag-sk-agents-aisearch.py-output-20261025-215423.txt) | `AzureAIAgent` with a `@kernel_function` price plugin | `.venv-semantic-kernel/bin/python src/4-frameworks/1-rag-sk-agents-aisearch.py` |
| AutoGen multi-agent RAG | [`2-autogen-multi-agent-rag.py`](../src/4-frameworks/2-autogen-multi-agent-rag.py) | [2-autogen-multi-agent-rag.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/2-autogen-multi-agent-rag.ipynb) | [output](../sample-output/2-autogen-multi-agent-rag.py-output-20261015-165725.txt) | retriever and responder agents, embedding search over tips | `uv run python src/4-frameworks/2-autogen-multi-agent-rag.py` |
| AutoGen wellness agents | [`3-autogen-personalized-heart-rate.py`](../src/4-frameworks/3-autogen-personalized-heart-rate.py) | [3-autogen-personalized-heart-rate.ipynb](https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/4-frameworks/3-autogen-personalized-heart-rate.ipynb) | [output](../sample-output/3-autogen-personalized-heart-rate.py-output-20261015-165744.txt) | hourly heart-rate summary from sample wearable data (pandas, matplotlib), then two agents plan workouts and breaks; a wellness demo with no health scoring, not medical advice | `uv run python src/4-frameworks/3-autogen-personalized-heart-rate.py --output-dir charts` |

The "Workshop notebook" column links to Microsoft's original of each lab, pinned at upstream commit `273c24c`; each
script's docstring has the same link.

The sample outputs are the full console output of a recorded run of each script (the run date is in the file name).
Runs recorded before the scripts moved to `src/` on 2026-10-23 show the old paths in their commands (for example
`2-notebooks/2-agent_service/...`) and three old file names with underscores (`2-code_interpreter.py`,
`4-bing_grounding.py`, `4-web_search.py`); the outputs are kept as recorded.
All are passing runs except the GenAI Ops full run, kept as evidence of the two failures it hid (see the
[findings](findings.md#red-teaming-an-agent-with-attack-simulators)) next to the rerun after the fix. My endpoints,
resource names and GUIDs (subscription, tenant and other IDs) are replaced with placeholders such as `<resource-name>`
and `<id>`; agent, thread and message IDs are left as they are, since the scripts delete those objects at the end of
each run. Model answers differ from run to run.

The image from the recorded image generation run: `gpt-image-2` at low quality, the notebook's prompt "A watercolor
painting of fresh fruits and vegetables arranged in a heart shape" (a 512 px copy of the 1024 px original):

<img src="../sample-output/2-embeddings-image-generation.py-image-20261018-133641.jpg" alt="Model-generated watercolor of fruits and vegetables arranged in a heart shape" width="256">

**Not included:** the workshop's end-to-end sample app
([original](https://github.com/Azure/ai-foundry-workshop/tree/main/3-ai-native-e2e-sample)), which runs on the current
SDKs in its own repo: [azure-ai-foundry-clinical-agents](https://github.com/khaledjamal/azure-ai-foundry-clinical-agents).

## What the other labs need

**Two more environments.** One environment holds only one version of each package, and some labs need versions
that conflict:

| Environment | Key packages | Labs | Why separate |
|---|---|---|---|
| `.venv` | `azure-ai-projects` 1.0.0, `azure-ai-agents` 1.1.0, AutoGen 0.7.5 | all others | the classic agents API (threads, messages, runs) the workshop notebooks use; `azure-ai-projects` 2.x removed it |
| `.venv-agents-v2` | `azure-ai-projects` 2.8.0, `mcp`, `azure-search-documents` | Bing, Web Search, Functions MCP, Azure AI Search (v2) | these tools need the new agents API, which exists only in 2.x |
| `.venv-semantic-kernel` | `semantic-kernel` 1.45.0, `azure-ai-agents` 1.2.0b6 | Semantic Kernel | Semantic Kernel needs a beta `azure-ai-agents` |

```bash
uv venv .venv-agents-v2 --python 3.13
uv pip install --python .venv-agents-v2/bin/python -r requirements-agents-v2.txt
uv venv .venv-semantic-kernel --python 3.13
uv pip install --python .venv-semantic-kernel/bin/python -r requirements-semantic-kernel.txt
```

**More model deployments.** The two image scripts of the embeddings lab need an `embed-v-4-0` and a `gpt-image-2`
deployment, the Phi-4 lab a `Phi-4` and a `Phi-4-reasoning` deployment, and the DeepSeek lab a `DeepSeek-V4-Flash`
deployment (all GlobalStandard, no cost while idle):

```bash
az cognitiveservices account deployment create -g <resource-group> -n <resource-name> --deployment-name gpt-image-2 \
  --model-name gpt-image-2 --model-version 2026-04-21 --model-format OpenAI --sku-name GlobalStandard --sku-capacity 1
az cognitiveservices account deployment create -g <resource-group> -n <resource-name> --deployment-name embed-v-4-0 \
  --model-name embed-v-4-0 --model-version 1 --model-format Cohere --sku-name GlobalStandard --sku-capacity 10
az cognitiveservices account deployment create -g <resource-group> -n <resource-name> --deployment-name Phi-4 \
  --model-name Phi-4 --model-version 7 --model-format Microsoft --sku-name GlobalStandard --sku-capacity 20
az cognitiveservices account deployment create -g <resource-group> -n <resource-name> --deployment-name Phi-4-reasoning \
  --model-name Phi-4-reasoning --model-version 1 --model-format Microsoft --sku-name GlobalStandard --sku-capacity 20
az cognitiveservices account deployment create -g <resource-group> -n <resource-name> --deployment-name DeepSeek-V4-Flash \
  --model-name DeepSeek-V4-Flash --model-version 2026-04-23 --model-format DeepSeek --sku-name GlobalStandard \
  --sku-capacity 20
```

**Other resources.**
- Application Insights connected to the Foundry project (portal: project, Tracing, Connect): for the GenAI Ops lab
  and the observability lab's `--destination azure-monitor` mode.
- A Grounding with Bing resource and a project connection to it: for the Bing lab
  (`GROUNDING_WITH_BING_CONNECTION_NAME`).
- The Function app of the MCP lab: see its [lab README](../src/functions-mcp-server/README.md).

`opentelemetry-util-genai` stays at 1.1b0 because version 1.2b0 renamed a module that
`opentelemetry-instrumentation-openai-v2` 2.4b0 still imports.

## Settings reference

All settings go in `.env` (start from [`.env.example`](../.env.example)):

| Variable | Used for |
|---|---|
| `AZURE_TOKEN_CREDENTIALS` | `dev`: `DefaultAzureCredential` tries only developer sign-ins such as `az login`, which skips a 1 s managed identity probe on every run ([finding](findings.md#sign-in-waited-1-s-for-a-managed-identity-that-isnt-there)) |
| `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_PROJECT_NAME` | Foundry project endpoint (or set `AZURE_AI_PROJECT_ENDPOINT` directly) |
| `OPENAI_ENDPOINT`, `OPENAI_API_VERSION` | Azure OpenAI endpoint of the Foundry resource |
| `MODEL_DEPLOYMENT_NAME` | chat model for the agent, framework, observability and GenAI Ops labs and the model comparisons (gpt-4o-mini) |
| `EMBEDDING_MODEL_DEPLOYMENT_NAME` | embedding model (text-embedding-3-small) |
| `IMAGE_EMBEDDING_MODEL_DEPLOYMENT_NAME`, `IMAGE_MODEL_DEPLOYMENT_NAME` | image embeddings (embed-v-4-0) and image generation (gpt-image-2) in the embeddings lab |
| `OPENAI_DEPLOYMENT_NAME` | chat model for the basic chat and Basic RAG labs, and judge model for the evaluation and GenAI Ops labs (gpt-4o) |
| `AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`, `SEARCH_INDEX_NAME` | Azure AI Search |
| `AZURE_FUNCTIONS_MCP_URL`, `AZURE_FUNCTIONS_MCP_KEY` | Azure Functions MCP lab |
| `GROUNDING_WITH_BING_CONNECTION_NAME` | Bing lab: name of the project connection to the Grounding with Bing resource |
| `PHI_MODEL_DEPLOYMENT_NAME`, `PHI_REASONING_MODEL_DEPLOYMENT_NAME` | Phi-4 lab (Phi-4 and Phi-4-reasoning) |
| `DEEPSEEK_MODEL_DEPLOYMENT_NAME` | DeepSeek lab (DeepSeek-V4-Flash) |

Microsoft's notebooks connect with a hub-style `PROJECT_CONNECTION_STRING`; current Foundry projects use the project
endpoint instead, which is why these scripts read `AZURE_AI_SERVICES_ENDPOINT` and `AZURE_AI_PROJECT_NAME` (or
`AZURE_AI_PROJECT_ENDPOINT`):

```text
Hub-style connection string (Microsoft's notebooks):
  <region>.api.azureml.ms;<subscription-id>;<resource-group>;<project-name>
  e.g. eastus2.api.azureml.ms;<subscription-id>;<resource-group>;<project-name>

Foundry project endpoint (these scripts):
  https://<resource-name>.services.ai.azure.com/api/projects/<project-name>
  = AZURE_AI_SERVICES_ENDPOINT + /api/projects/ + AZURE_AI_PROJECT_NAME
    (or set AZURE_AI_PROJECT_ENDPOINT directly)
```

### HTTP log

Every script calls `http_log.enable()` from [`src/workshop_utils/http_log.py`](../src/workshop_utils/http_log.py), which
prints each HTTP request and response the script sends: method, URL, headers and body, then status, headers and
body. It hooks the transport of `requests`, `httpx2` and `httpx`, so it also shows the calls the SDKs make (openai,
the azure-* SDKs, Semantic Kernel, AutoGen, the MCP client). Authorization and key headers, key query parameters and
token fields are replaced with `<redacted>`; long and binary bodies are shortened. A request and its response share a
call number:

```text
--- OUTBOUND REQUEST #1 ---
> POST https://<resource-name>.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=...
> Authorization: Bearer <redacted>
> Content-Type: application/json
>
{
  "messages": [ ... ]
}

--- OUTBOUND RESPONSE #1 ---
< 200 OK
< Content-Type: application/json
<
{
  "choices": [ ... ]
}
```

It is set with environment variables on the command line (not in `.env`, which the scripts read after the log
starts), for example `HTTP_LOG=summary uv run python src/check-setup.py`:

| Variable | Values |
|---|---|
| `HTTP_LOG` | `full` (default), `summary` (one line per call), `off` |
| `HTTP_LOG_METRICS` | `false` (default), `true`: time, bytes and token usage per call, and totals at the end |
| `HTTP_LOG_MAX_BODY` | characters shown of each body, default 8192; the rest is cut with `...more...` |
| `HTTP_LOG_POLLS` | `short` (default): a repeated GET of the same URL, such as an agent run checked until it ends, is one line with the reply's `status` (`HTTP #7 GET <url> -> 200 OK, status: in_progress`); the first GET of a URL stays in full. `full`: every one in full |

### Record and replay

The same hooks can save every call of a run and answer the next run from the saved files, so a script runs again
without calling Azure or paying ([`http_replay.py`](../src/workshop_utils/http_replay.py)):

```bash
# real calls, saved (runs/ is git-ignored)
HTTP_REPLAY=record HTTP_REPLAY_DIR=runs/recordings/chat uv run python src/1-models/1-basic-chat-completion-azure-openai.py
# the same run again from the saved files, no network
HTTP_REPLAY=replay HTTP_REPLAY_DIR=runs/recordings/chat uv run python src/1-models/1-basic-chat-completion-azure-openai.py
```

| Variable | Values |
|---|---|
| `HTTP_REPLAY` | `off` (default), `record`, `replay` |
| `HTTP_REPLAY_DIR` | folder of the recording: one JSON file per call, with the whole response; secrets redacted |
| `HTTP_REPLAY_IGNORE_FIELDS` | JSON body fields left out when requests are matched, comma-separated (for values that change on every run) |

A replayed request must match a recorded one (method, URL, body); one that doesn't fails with `HttpReplayMiss`
instead of calling the service, connections to other machines are refused, and the Azure CLI sign-in isn't needed.
A replay shows that the code still handles the recorded answers, not that the service still answers the same way.
Values that change between runs make a request new: the Web Search lab, for example, puts today's date in its
instructions, so its recording replays only on the same day. HTTP only: requests, httpx and aiohttp; other protocols
(Event Hubs' AMQP, for example) aren't covered.

Unit tests against a local server (no Azure): `uv run python -m unittest discover -s tests -v`.

## Troubleshooting

Issues I hit while building the labs:

| Symptom | Cause | Fix |
|---|---|---|
| `DefaultAzureCredential failed to retrieve a token` | not signed in with the Azure CLI | `az login`, then `uv run python src/check-setup.py` |
| `Required environment variable is not set: AZURE_AI_PROJECT_NAME` (or another setting) | `.env` is missing or incomplete | `cp .env.example .env`, fill it in, run the setup check |
| 429 (rate limit) errors from a new deployment, often on the first or second call | the deployment's rate limit is too low: for the Phi-4, DeepSeek and embed-v-4-0 deployments here, capacity 1 meant 1 request and 1,000 tokens per minute (the ratio differs by model) | check the limits with `az cognitiveservices account deployment show -g <resource-group> -n <resource-name> --deployment-name <deployment> --query properties.rateLimits`, then raise the capacity (for example `--sku-capacity 20` in the deployment commands above); pay-as-you-go billing is per token, so capacity costs nothing |
| agent scripts fail on `agents.threads`, `messages` or `runs` | `azure-ai-projects` 2.x removed the classic agents operations | keep `azure-ai-projects==1.0.0` in the main environment; the 2.x labs use their own `.venv-agents-v2` |
| the MCP agent's tool calls get `503 Site Unavailable` | Python on Flex Consumption handles one request at a time per instance, and the MCP client's open stream holds that slot | raise the per-instance concurrency; see the [lab README](../src/functions-mcp-server/README.md) |
| Phi-4-reasoning's thinking can't be split from its answer | the Azure endpoint strips the `<think>` tags and rejects `skip_special_tokens` | nothing to fix: the reply is printed as one text (see [Phi-4 in the findings](findings.md#phi-4-and-phi-4-reasoning-vs-gpt-4o-mini)) |
