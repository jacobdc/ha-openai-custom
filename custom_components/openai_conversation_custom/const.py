"""Constants for OpenAI Conversation Custom."""
import logging

DOMAIN = "openai_conversation_custom"
LOGGER = logging.getLogger(__package__)

CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_TIMEOUT = "timeout"
CONF_PROMPT = "prompt"

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 60
DEFAULT_NAME = "OpenClaw"
DEFAULT_PROMPT = ""
