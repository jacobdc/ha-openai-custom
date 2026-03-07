# Changelog

## v2.0.0 — OpenClaw-native rewrite
- Removed OpenAI SDK dependency entirely
- Direct `aiohttp` calls to OpenClaw gateway `/v1/chat/completions`
- Streaming-first conversation entity (`_attr_supports_streaming = True`)
- SSE stream parsing → HA delta content stream for early TTS start
- Simplified to conversation-only (removed STT, TTS, AI Task subentries)
- Config flow validates with a real chat completion, no `/v1/models` dependency
- Conversation history per conversation_id (bounded to last 20 messages)

## v1.1.0
- Rewrote from Responses API to Chat Completions API

## v1.0.2
- Skip models.list at runtime for custom base_url

## v1.0.1
- Fix validation for custom endpoints

## v1.0.0
- Initial release: fork of HA native OpenAI Conversation with base_url support
