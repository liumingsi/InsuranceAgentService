import json


SOURCE_TOOL_NAME = "retrieve_insurance_clause"
MAX_SOURCE_CARDS = 4
MAX_SOURCE_EXCERPT_LENGTH = 160


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def source_cards_from_tool_message(message) -> list[dict]:
    """Parse compact source cards from a clause-retrieval ToolMessage."""
    if _field(message, "type") != "tool" or _field(message, "name") != SOURCE_TOOL_NAME:
        return []

    content = _field(message, "content")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        return []

    try:
        tool_sources = json.loads(content)
    except (TypeError, ValueError):
        return []

    if isinstance(tool_sources, dict):
        tool_sources = tool_sources.get("sources", [])
    if not isinstance(tool_sources, list):
        return []

    cards = []
    seen_ids = set()
    for source in tool_sources:
        if not isinstance(source, dict):
            continue

        citation_id = source.get("citation_id")
        title = source.get("clause_name")
        excerpt = source.get("source_excerpt")
        if not citation_id or not title or not excerpt:
            continue

        citation_id = str(citation_id)
        if citation_id in seen_ids:
            continue
        seen_ids.add(citation_id)

        section_path = source.get("section_path") or []
        if isinstance(section_path, str):
            section = section_path
        else:
            section = " > ".join(str(part) for part in section_path if part)

        excerpt = " ".join(str(excerpt).split())
        if len(excerpt) > MAX_SOURCE_EXCERPT_LENGTH:
            excerpt = excerpt[:MAX_SOURCE_EXCERPT_LENGTH].rstrip() + "…"

        cards.append(
            {
                "id": citation_id,
                "title": str(title),
                "section": section,
                "excerpt": excerpt,
            }
        )
        if len(cards) >= MAX_SOURCE_CARDS:
            break

    return cards


def source_cards_for_latest_turn(messages: list) -> list[dict]:
    """Collect clause source cards after the latest user message."""
    last_user_message = -1
    for index, message in enumerate(messages):
        if _field(message, "type") in ("human", "user"):
            last_user_message = index

    cards = []
    seen_ids = set()
    for message in messages[last_user_message + 1:]:
        for card in source_cards_from_tool_message(message):
            if card["id"] in seen_ids:
                continue
            seen_ids.add(card["id"])
            cards.append(card)
            if len(cards) >= MAX_SOURCE_CARDS:
                return cards
    return cards

