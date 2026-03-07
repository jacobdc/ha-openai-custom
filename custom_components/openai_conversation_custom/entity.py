"""Base entity for OpenAI."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Callable, Iterable
import json
from mimetypes import guess_file_type
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openai
from openai._streaming import AsyncStream
from openai.types.chat import ChatCompletionChunk
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, issue_registry as ir, llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
from homeassistant.util import slugify

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from . import OpenAIConfigEntry


# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenAI API."""
    if schema["type"] == "object":
        schema.setdefault("strict", True)
        schema.setdefault("additionalProperties", False)
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    schema: vol.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Format the schema to be compatible with OpenAI API."""
    result: dict[str, Any] = convert(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result)

    return result


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification for Chat Completions API."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


def _convert_content_to_param(
    chat_content: Iterable[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert HA chat content to Chat Completions message format."""
    messages: list[dict[str, Any]] = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                    "content": json_dumps(content.tool_result),
                }
            )
            continue

        if isinstance(content, conversation.AssistantContent):
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": content.content or None,
            }
            if content.tool_calls:
                tool_calls_list = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json_dumps(tc.tool_args),
                        },
                    }
                    for tc in content.tool_calls
                    if not tc.external
                ]
                if tool_calls_list:
                    msg["tool_calls"] = tool_calls_list
            if msg.get("content") is not None or "tool_calls" in msg:
                messages.append(msg)
        elif content.content:
            messages.append(
                {
                    "role": content.role,
                    "content": content.content,
                }
            )

    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result_stream: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Chat Completions delta stream into HA format."""
    current_tool_calls: dict[int, dict[str, Any]] = {}
    started = False

    async for chunk in result_stream:
        LOGGER.debug("Received chunk: %s", chunk)

        if not chunk.choices:
            # Usage-only chunk at the end (when stream_options.include_usage is set)
            if chunk.usage:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                        }
                    }
                )
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            if not started:
                yield {"role": "assistant"}
                started = True
            yield {"content": delta.content}

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in current_tool_calls:
                    current_tool_calls[idx] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                    }
                tc = current_tool_calls[idx]
                if tc_delta.id:
                    tc["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc["arguments"] += tc_delta.function.arguments

        if choice.finish_reason is not None and current_tool_calls:
            # Emit each accumulated tool call as its own assistant content block
            for tc in current_tool_calls.values():
                yield {"role": "assistant"}
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tc["id"],
                            tool_name=tc["name"],
                            tool_args=json.loads(tc["arguments"]),
                        )
                    ]
                }
            current_tool_calls = {}

        if choice.finish_reason is not None and not started:
            # Empty response - still need to yield an assistant role
            yield {"role": "assistant"}
            started = True

        if choice.finish_reason == "length":
            raise HomeAssistantError(
                "OpenAI response incomplete: max output tokens reached"
            )


class OpenAIBaseLLMEntity(Entity):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: OpenAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI",
            model=subentry.data.get(
                CONF_CHAT_MODEL,
                RECOMMENDED_CHAT_MODEL
                if subentry.subentry_type != "stt"
                else RECOMMENDED_STT_MODEL,
            ),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        force_image: bool = False,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data

        messages = _convert_content_to_param(chat_log.content)

        model_args: dict[str, Any] = {
            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "messages": messages,
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "user": chat_log.conversation_id,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if tools:
            model_args["tools"] = tools

        last_content = chat_log.content[-1]

        # Handle attachments by adding them to the last user message
        if last_content.role == "user" and last_content.attachments:
            files = await async_prepare_files_for_prompt(
                self.hass,
                [(a.path, a.mime_type) for a in last_content.attachments],
            )
            last_message = messages[-1]
            assert (
                last_message["role"] == "user"
                and isinstance(last_message["content"], str)
            )
            last_message["content"] = [
                {"type": "text", "text": last_message["content"]},
                *files,
            ]

        if structure and structure_name:
            model_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": slugify(structure_name),
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                    "strict": True,
                },
            }

        client = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(max_iterations):
            try:
                stream = await client.chat.completions.create(**model_args)

                messages.extend(
                    _convert_content_to_param(
                        [
                            content
                            async for content in chat_log.async_add_delta_content_stream(
                                self.entity_id,
                                _transform_stream(chat_log, stream),
                            )
                        ]
                    )
                )
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                if (
                    isinstance(err, openai.APIError)
                    and err.type == "insufficient_quota"
                ):
                    LOGGER.error("Insufficient funds for OpenAI: %s", err)
                    raise HomeAssistantError("Insufficient funds for OpenAI") from err
                if "Verify Organization" in str(err):
                    ir.async_create_issue(
                        self.hass,
                        DOMAIN,
                        "organization_verification_required",
                        is_fixable=False,
                        is_persistent=False,
                        learn_more_url="https://help.openai.com/en/articles/10910291-api-organization-verification",
                        severity=ir.IssueSeverity.WARNING,
                        translation_key="organization_verification_required",
                        translation_placeholders={
                            "platform_settings": "https://platform.openai.com/settings/organization/general"
                        },
                    )

                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            if not chat_log.unresponded_tool_results:
                break


async def async_prepare_files_for_prompt(
    hass: HomeAssistant, files: list[tuple[Path, str | None]]
) -> list[dict[str, Any]]:
    """Append files to a prompt.

    Caller needs to ensure that the files are allowed.
    """

    def append_files_to_content() -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        for file_path, mime_type in files:
            if not file_path.exists():
                raise HomeAssistantError(f"`{file_path}` does not exist")

            if mime_type is None:
                mime_type = guess_file_type(file_path)[0]

            if not mime_type or not mime_type.startswith(("image/", "application/pdf")):
                raise HomeAssistantError(
                    "Only images and PDF are supported by the OpenAI API,"
                    f"`{file_path}` is not an image file or PDF"
                )

            base64_file = base64.b64encode(file_path.read_bytes()).decode("utf-8")

            if mime_type.startswith("image/"):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_file}",
                            "detail": "auto",
                        },
                    }
                )
            elif mime_type.startswith("application/pdf"):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_file}",
                        },
                    }
                )

        return content

    return await hass.async_add_executor_job(append_files_to_content)
