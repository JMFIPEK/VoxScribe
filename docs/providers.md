# Providers

A **provider** is a transcription service VoxScribe sends the audio to: any
server with an OpenAI-compatible `/audio/transcriptions` endpoint. Alignment
and speaker identification still run on your computer.

Why use one: no model download and fast on any laptop. The catch: the audio
leaves your computer for that request. Without a provider, VoxScribe
transcribes fully locally.

## Set up OpenRouter (example)

1. Create an account at [openrouter.ai](https://openrouter.ai), add credit, and
   create an API key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
2. In VoxScribe, open **Settings → Providers → Add...**
3. **Preset: OpenRouter** fills in name, base URL
   (`https://openrouter.ai/api/v1`) and model (`openai/whisper-large-v3`).
4. Paste your **API key** and save.
5. Select the new provider and click **Set as default**.

**Check it worked:** transcribe a short recording. The status line under the
progress bar shows *Sending audio to server (OpenRouter)...*. If the provider
fails (wrong key, no credit, no network), it shows *Provider request failed -
falling back to ...* instead, VoxScribe finishes with a local model, and the
reason is in the log (`%USERPROFILE%\.voxscribe\voxscribe.log`).

## Any other OpenAI-compatible service

**Settings → Providers → Add...**, Preset **Custom**, then:

| Field | What to enter | Example |
|---|---|---|
| Name | Any label | `My Whisper server` |
| Base URL | The API's base URL, without `/audio/transcriptions` | `https://example.com/api/v1` |
| API key | Sent as `Authorization: Bearer <key>` | |
| Model | The model name the server expects | `whisper-large-v3` |

Prefer Whisper models: VoxScribe asks for segment timestamps
(`response_format=verbose_json`), and Whisper servers return them. Other models
work too, with slightly less precise timestamps (see Reference below).

## Which provider is used

- The provider marked **default** is used whenever you don't pick a model
  yourself.
- In the Transcription tab you can choose any provider or a local model for one
  transcription.
- If the provider fails, VoxScribe retries once with the best local model for
  your hardware.

---

## Reference

### Storage

Providers live in `~/.voxscribe/providers.json` (`voxscribe/providers.py`),
one entry per provider:

```json
{
  "providers": [
    {"id": "my-server", "name": "My Whisper server", "base_url": "...", "api_key": "...", "model": "whisper-large-v3"}
  ],
  "default_provider_id": "my-server"
}
```

The API key is stored in plain text in that file, readable by your user
account.

### Model IDs and the CLI

A provider's Model ID is `server:<provider-id>` (`transcriber.remote_model_id()`),
so the CLI uses `--model server:my-server`. For one-off use without saving a
provider, `--api-key`/`--api-base-url` override a saved provider's credentials
for that run. See [CLI reference](cli-reference.md).

### Servers that return only text

Some servers ignore `response_format=verbose_json` and return only
`{"text": ...}`: no segments, no language. VoxScribe detects this on the first
request and then sends 30 s chunks instead of ~180 s ones (cut at the quietest
moment near each boundary), so local alignment only has to place words within
a short window. A missing language is guessed from the returned text, so
alignment uses the right language model. Timestamps are still slightly less
precise than with real server segments.

### Migrating from a pre-provider install

Versions before the provider system used one server configured in `.env`:

```
KIT_TOOLBOX_API_KEY=xxx
KIT_TOOLBOX_BASE_URL=https://...
```

If `~/.voxscribe/providers.json` doesn't exist yet and these are set, the app
creates a provider from them once at start (`providers.migrate_from_env()`).
After that, manage it in Settings.
