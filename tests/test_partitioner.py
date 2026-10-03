from cachealign.normalizers.partitioner import (
    extract_volatile_elements,
    partition_messages_and_system,
)


def test_extract_volatile_elements():
    prompt = """
You are an enterprise research assistant.
Current Time: 2026-10-03 09:20:00
Session-ID: sess-948192
Follow these 50 strict rules...
"""
    clean_text, extracted = extract_volatile_elements(prompt)

    assert "Current Time:" not in clean_text
    assert "Session-ID:" not in clean_text
    assert "You are an enterprise research assistant." in clean_text
    assert "Follow these 50 strict rules..." in clean_text
    assert len(extracted) == 2


def test_partition_messages_and_system_ephemeral_tail_migration():
    system = "You are a customer assistant.\nCurrent Time: 2026-10-03 12:00:00\nHelp the user with queries."
    messages = [{"role": "user", "content": "What is my account balance?"}]

    clean_sys, new_msgs, extracted = partition_messages_and_system(system, messages)

    assert "Current Time:" not in clean_sys
    assert len(extracted) == 1
    # Check that volatile timestamp was migrated to the tail of the user message
    assert "Current Time:" in new_msgs[-1]["content"]
    assert "[Context:" in new_msgs[-1]["content"]
