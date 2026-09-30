from importlib.metadata import distribution, version
from pathlib import Path

BEFORE = '''            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
'''
AFTER = '''            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if kwargs.get("continue_final_message", False):
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError(
                        "continue_final_message requires a final assistant message"
                    )
                messages = copy.deepcopy(messages)
                messages[-1]["wo_eos"] = True
            if tools is not None and len(tools) > 0:
'''


def patch_source(source):
    if AFTER in source:
        return source
    if source.count(BEFORE) != 1:
        raise RuntimeError("unexpected vLLM DeepSeek-V4 tokenizer source")
    return source.replace(BEFORE, AFTER)


def main():
    import argparse
    argparse.ArgumentParser(description="Make vLLM's DeepSeek-V4 tokenizer honour continue_final_message (assistant prefill).").parse_args()
    if version("vllm") != "0.26.0":
        raise RuntimeError(f"written for vllm 0.26.0, found {version('vllm')}")
    path = Path(distribution("vllm").locate_file("vllm/tokenizers/deepseek_v4.py"))
    updated = patch_source(path.read_text())
    compile(updated, str(path), "exec")
    path.write_text(updated)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
