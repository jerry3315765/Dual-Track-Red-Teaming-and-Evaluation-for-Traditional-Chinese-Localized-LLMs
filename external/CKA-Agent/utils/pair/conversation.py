from fastchat.model import get_conversation_template


def make_conv_template(template_name: str):
    conv = get_conversation_template(template_name)
    # Fastchat version compatibility
    if not hasattr(conv, "system") and hasattr(conv, "system_template"):
        conv.system = conv.system_template or ""
    if not hasattr(conv, "messages"):
        conv.messages = []
    return conv


def render_full_prompt(conv) -> str:
    try:
        return conv.get_prompt()
    except Exception:
        # Fallback: concatenate system and user roles
        parts = []
        if getattr(conv, "system", ""):
            parts.append(str(conv.system))
        for role, content in getattr(conv, "messages", []):
            if content is None:
                continue
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)
