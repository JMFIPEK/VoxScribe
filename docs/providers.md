# Transcription providers

VoxScribe can transcribe either locally (Whisper/WhisperX, Apple SpeechAnalyzer,
or the Intel Arc GPU path) or via any remote, OpenAI-compatible
`/audio/transcriptions` endpoint - a "provider". Providers are configured in
the app, not hardcoded to any specific vendor.

## Adding a provider

In the GUI, go to **Settings → Providers → Add...** and fill in:

- **Name** - a label for your own reference (e.g. "My Whisper server")
- **Base URL** - the API's base URL (e.g. `https://example.com/api/v1`)
- **API key** - sent as a Bearer token
- **Model** - the model name to request (e.g. `whisper-large-v3`)

Mark one provider as the default with **Set as default**. The default
provider is what `transcriber.default_model_size()` returns when nothing else
is explicitly chosen - see the [main README](../README.md) for the exact
fallback behavior.

## How it works

Provider configuration lives in `~/.voxscribe/providers.json` (see
`voxscribe/providers.py`), a small local JSON store - one entry per provider:

```json
{
  "providers": [
    {"id": "my-server", "name": "My Whisper server", "base_url": "...", "api_key": "...", "model": "whisper-large-v3"}
  ],
  "default_provider_id": "my-server"
}
```

Selecting a provider as the transcription model uses the `server:<provider-id>`
model_size string (see `transcriber.remote_model_id()`); the CLI's
`--model server:<provider-id>` flag works the same way.

For one-off use without saving a provider, the CLI also accepts
`--api-key`/`--api-base-url` directly, which override any saved provider's
credentials for that run.

## Migrating from an older, pre-provider install

Versions of VoxScribe before the provider system used a single hardcoded
server, configured via two environment variables in `.env`:

```
KIT_TOOLBOX_API_KEY=xxx
KIT_TOOLBOX_BASE_URL=https://...
```

If `~/.voxscribe/providers.json` doesn't exist yet and these variables are
set, the app automatically creates a provider from them the first time it
starts (see `providers.migrate_from_env()`), so upgrading doesn't silently
drop a working setup. This only happens once; after that, manage the
provider normally in Settings.
