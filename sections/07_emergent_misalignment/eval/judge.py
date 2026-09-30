import asyncio
import math

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

JUDGE_CONFIG = GenerateConfig(max_tokens=1, temperature=0, logprobs=True, top_logprobs=20, seed=0,
                              max_connections=64)


def score_0_100(top_logprobs):
    total, weighted = 0.0, 0.0
    for t in top_logprobs:
        try:
            k = int(t.token)
        except ValueError:
            continue
        if 0 <= k <= 100:
            p = math.exp(t.logprob)
            total += p
            weighted += k * p
    return weighted / total if total >= 0.25 else None


async def judge_one(model, prompt_template, question, answer):
    out = await model.generate([ChatMessageUser(content=prompt_template.format(question=question, answer=answer))])
    lp = out.choices[0].logprobs
    if lp is None or not lp.content or not lp.content[0].top_logprobs:
        return None
    return score_0_100(lp.content[0].top_logprobs)


async def judge_all(judge_model, judge_prompts, questions, answers):
    model = get_model(f"openai/{judge_model}", config=JUDGE_CONFIG, responses_api=False)
    scores = {}
    for metric, prompt in judge_prompts.items():
        scores[metric] = await asyncio.gather(*[judge_one(model, prompt, q, a) for q, a in zip(questions, answers)])
    return scores

