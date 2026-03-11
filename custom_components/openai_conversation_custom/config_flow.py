"""Config flow for OpenAI Conversation Custom."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required("name", default=DEFAULT_NAME): str,
        vol.Required(CONF_BASE_URL): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): str,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): int,
    }
)


async def _validate(hass: Any, data: dict) -> None:
    """Validate connection by doing a real chat completion."""
    session = async_get_clientsession(hass)
    base_url = data[CONF_BASE_URL].rstrip("/")
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": data.get(CONF_MODEL, DEFAULT_MODEL),
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {data[CONF_API_KEY]}"},
            timeout=timeout,
        ) as resp:
            if resp.status == 401:
                raise InvalidAuth
            if resp.status >= 400:
                body = await resp.text()
                raise CannotConnect(f"HTTP {resp.status}: {body[:100]}")
            await resp.json(content_type=None)
    except aiohttp.ClientConnectorError as err:
        raise CannotConnect(str(err)) from err
    except asyncio.TimeoutError as err:
        raise CannotConnect("Timeout") from err


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""


class OpenAICustomConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation Custom."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OpenAICustomOptionsFlow:
        """Get the options flow for this handler."""
        return OpenAICustomOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await _validate(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                name = user_input.pop("name", DEFAULT_NAME)
                return self.async_create_entry(title=name, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )


class OpenAICustomOptionsFlow(OptionsFlow):
    """Handle options for OpenAI Conversation Custom."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            # Merge options into data so conversation.py reads them
            new_data = {**self.config_entry.data, **user_input}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )
            return self.async_create_entry(title="", data=user_input)

        current = self.config_entry.data
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_PROMPT,
                        default=current.get(CONF_PROMPT, DEFAULT_PROMPT),
                    ): str,
                    vol.Optional(
                        CONF_MODEL,
                        default=current.get(CONF_MODEL, DEFAULT_MODEL),
                    ): str,
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=current.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): int,
                }
            ),
        )
