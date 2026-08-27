# Third-party research provenance

PatrolLens does not vendor or copy source from the three research repositories. It independently implements narrow interfaces inspired by their public architectures. The inspected revisions are pinned in `upstreams.lock.json`.

## OmniAgent

- Repository: <https://github.com/harryhsing/omniagent>
- Inspected commit: `adea8a098ab681cacdc789b16b0acc4b2bd36872`
- Upstream license: Apache-2.0.
- Influence: the reset/step environment pattern, bounded `get_frames`, `get_audio`, and `get_clip` actions, and compact observation memory.

PatrolLens uses Gemini as the policy and implements its own controller, schemas, FFmpeg commands, memory format, and safety limits.

## OpenRouter transport

- API: <https://openrouter.ai/api/v1/chat/completions>
- Transcription API: <https://openrouter.ai/api/v1/audio/transcriptions>
- Client: the optional `openai` Python package, pointed at OpenRouter's OpenAI-compatible base URL.
- Model selection: configured OpenRouter slugs such as `google/gemini-3.1-pro-preview`; no Google SDK is required.
- Chat media contract: local observations are sent as base64 `image_url`, `video_url`, or `input_audio` message parts. Only bounded active-perception/refinement media is sent through chat completions.

### Gemini Embedding 2

- API: <https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request>
- Ingestion uses `google/gemini-embedding-2` for synchronous document and media vectors; queries use the same model by default.
- The embedding adapter sends text, image, audio, and video inputs to the OpenRouter embeddings endpoint and stores the canonical model namespace with every vector.
- `openai/whisper-large-v3-turbo` produces checkpointed segment transcripts
  through OpenRouter. Exact transcript strings are kept in FTS5 and optionally
  embedded for semantic recall. Current ingestion does not run OCR or YAMNet.
- The production vector size is 768 dimensions and is configured with `PATROLLENS_EMBEDDING_DIMENSIONS`; changing dimensions or model namespaces requires a new ingestion fingerprint.
- `google/gemini-embedding-2:batch` is reserved for a future asynchronous text-only Batch API path and is not used by current ingestion. Multimodal image, audio, and video inputs remain on the synchronous embeddings endpoint.

## PostgreSQL and pgvector

- `psycopg` is used for PostgreSQL connections and transactions.
- The PostgreSQL server must have the `vector` extension installed; the included `compose.pgvector.yaml` provides a local development service.
- The Python `pgvector` package is included in the optional `postgres` extra, while vector provenance and search SQL live in `index/postgres_store.py`.

## Video-RAG

- Repository: <https://github.com/Leon1207/Video-RAG-master>
- Inspected commit: `4400aa4d0674fda6501688a60da7cdee925c1fa1`
- License note: no top-level `LICENSE` or `LICENSE.md` was present at the inspected revision.
- Influence: the architecture of retrieving visually aligned ASR/OCR/visual auxiliary evidence before expensive multimodal reasoning.

Because repository code permissions are unclear, PatrolLens uses only the published architectural idea and independently implements storage, FTS/FAISS search, temporal joining, and fusion.

## TimeLens2

- Repository: <https://github.com/MCG-NJU/TimeLens2>
- Inspected commit: `f828dde4ee7f2aea5022c6bd7f5aa1d42d4b3f35`
- Influence: one-or-more temporal intervals as the grounding abstraction.

The repository README describes Apache-2.0, but its inspected top-level `LICENSE` states that use is academic-only, prohibits commercial/production use, and says it is not intended for use within the European Union. The top-level license is treated as controlling for this integration. Therefore:

- TimeLens2 is not a PatrolLens dependency.
- Its code or checkpoints are not bundled.
- The adapter is disabled by default.
- Enabling it requires an explicit license acknowledgement and an independently installed wrapper command.

This note is an engineering safeguard, not legal advice. Review the upstream terms for the intended deployment.
