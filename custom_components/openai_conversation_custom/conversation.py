"""Conversation agent for OpenClaw — streaming-first."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, area_registry as ar
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.const import MATCH_ALL

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DOMAIN,
    LOGGER,
)

_LOGGER = logging.getLogger(__name__)

MAX_HISTORY = 20  # messages to keep per conversation


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    async_add_entities([OpenClawConversationEntity(hass, config_entry)])


class OpenClawConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """OpenClaw conversation agent with streaming support."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._conversations: dict[str, list[dict[str, Any]]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register as conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister as conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _get_client_params(self) -> tuple[str, str, str, int]:
        """Return (base_url, api_key, model, timeout)."""
        data = self.entry.data
        base_url = data[CONF_BASE_URL].rstrip("/")
        api_key = data[CONF_API_KEY]
        model = data.get(CONF_MODEL, DEFAULT_MODEL)
        timeout = int(data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        return base_url, api_key, model, timeout

    def _resolve_device_context(
        self, user_input: conversation.ConversationInput
    ) -> dict[str, str]:
        """Resolve area name, device name, and media player from satellite/device."""
        area_reg = ar.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)

        area_name: str | None = None
        device_name: str | None = None
        media_player: str | None = None
        device_id: str | None = None

        # Try satellite_id first (entity_id of the assist satellite)
        if user_input.satellite_id:
            entry = ent_reg.async_get(user_input.satellite_id)
            if entry:
                device_id = entry.device_id
                if entry.area_id:
                    area = area_reg.async_get_area(entry.area_id)
                    if area:
                        area_name = area.name
                if device_id:
                    device = dev_reg.async_get(device_id)
                    if device:
                        device_name = device.name_by_user or device.name
                        if not area_name and device.area_id:
                            area = area_reg.async_get_area(device.area_id)
                            if area:
                                area_name = area.name

        # Try device_id as fallback
        if not device_id and user_input.device_id:
            device_id = user_input.device_id
            device = dev_reg.async_get(device_id)
            if device:
                device_name = device.name_by_user or device.name
                if not area_name and device.area_id:
                    area = area_reg.async_get_area(device.area_id)
                    if area:
                        area_name = area.name

        # Find media_player entity on the same device
        if device_id:
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.domain == "media_player" and not entry.disabled:
                    media_player = entry.entity_id
                    break

        return {
            "area_name": area_name or "ukendt",
            "device_name": device_name or "ukendt",
            "media_player": media_player or "ukendt",
        }

    def _render_prompt(self, prompt: str, user_input: conversation.ConversationInput) -> str:
        """Replace placeholders in the prompt with actual values."""
        from datetime import datetime

        now = datetime.now()
        ctx = self._resolve_device_context(user_input)

        replacements = {
            "{area_name}": ctx["area_name"],
            "{device_name}": ctx["device_name"],
            "{media_player}": ctx["media_player"],
            "{time}": now.strftime("%H:%M"),
            "{date}": now.strftime("%d/%m-%Y"),
            "{language}": user_input.language or "da",
        }
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        return prompt

    def _build_messages(
        self,
        history: list[dict[str, Any]],
        chat_log: conversation.ChatLog,
    ) -> list[dict[str, Any]]:
        """Build the message list from HA chat log + history."""
        # Get system prompt from config + HA context
        system_prompt = self.entry.data.get(CONF_PROMPT, "")

        # Build from HA chat log content
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(history)

        # Add any remaining content from the chat_log that isn't in history
        # The last user message comes from user_input directly
        for content in chat_log.content:
            if content.role == "user" and content.content:
                # Already in history or is the new message — we'll add from user_input
                pass

        return messages

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message — streaming path."""
        conversation_id = user_input.conversation_id or user_input.device_id or "default"

        # Get or init history
        history = self._conversations.get(conversation_id, [])

        # Add system prompt if first message and configured
        raw_prompt = self.entry.data.get(CONF_PROMPT, "")
        system_prompt = self._render_prompt(raw_prompt, user_input) if raw_prompt else ""
        messages: list[dict[str, Any]] = []
        if system_prompt and not history:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input.text})

        base_url, api_key, model, timeout = self._get_client_params()
        session = async_get_clientsession(self.hass)

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                None,
                None,  # Don't pass HA default prompt — we use our own
                system_prompt or user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        full_response_parts: list[str] = []

        try:
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._stream_response(
                    session, base_url, api_key, model, timeout, messages
                ),
            ):
                if isinstance(content, conversation.AssistantContent) and content.content:
                    full_response_parts.append(content.content)
        except HomeAssistantError:
            raise
        except Exception as err:
            _LOGGER.error("Error calling OpenClaw: %s", err)
            raise HomeAssistantError(f"Error talking to OpenClaw: {err}") from err

        # Update history
        full_response = "".join(full_response_parts)
        history.append({"role": "user", "content": user_input.text})
        history.append({"role": "assistant", "content": full_response})
        self._conversations[conversation_id] = history[-MAX_HISTORY:]

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _stream_response(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
        """Call OpenClaw with stream=true and yield HA delta dicts."""
        http_timeout = aiohttp.ClientTimeout(total=timeout)
        started = False

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream",
            },
            timeout=http_timeout,
        ) as resp:
            if resp.status == 401:
                raise HomeAssistantError("Invalid OpenClaw API token")
            if resp.status >= 400:
                body = await resp.text()
                raise HomeAssistantError(f"OpenClaw error {resp.status}: {body[:200]}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if not started:
                        yield {"role": "assistant"}
                        started = True
                    yield {"content": content}

        if not started:
            yield {"role": "assistant"}
