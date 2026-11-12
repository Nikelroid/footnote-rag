"""Model prices and the one function that turns token counts into dollars.

Claude prices are copied from https://platform.claude.com/docs/en/about-claude/pricing.
The one Gemini price is copied from https://cloud.google.com/vertex-ai/generative-ai/pricing.
Both were read on 2026-11-10. A model that is not in the table raises an error, and
so does a price whose end date has passed. Nothing is ever priced by guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
GOOGLE_PRICING_SOURCE = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
PRICING_READ_ON = "2026-11-10"

BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


@dataclass(frozen=True)
class ModelPricing:
    base: Rates
    # Claude Haiku 5.5 is priced by prompt length. A request whose prompt is over
    # the threshold pays the `long` rates on every token, output included. The
    # prompt length counts all input tokens, cache reads and cache writes too.
    long_threshold: int | None = None
    long: Rates | None = None
    # The last day these rates apply, for a price the provider has said will change.
    valid_until: date | None = None


PRICING: dict[str, ModelPricing] = {
    "claude-haiku-5-5": ModelPricing(
        base=Rates(input=0.10, cache_write_5m=0.125, cache_write_1h=0.20, cache_read=0.01, output=0.50),
        long_threshold=100_000,
        long=Rates(input=0.50, cache_write_5m=0.625, cache_write_1h=1.00, cache_read=0.05, output=2.50),
    ),
    "claude-sonnet-5-5": ModelPricing(
        base=Rates(input=2.00, cache_write_5m=2.50, cache_write_1h=4.00, cache_read=0.10, output=10.00),
    ),
    "claude-opus-5-5": ModelPricing(
        base=Rates(input=4.00, cache_write_5m=5.00, cache_write_1h=8.00, cache_read=0.20, output=20.00),
    ),
    # Gemini 3.8 Flash on Google Cloud's global endpoint, at the standard price.
    # Google's price page lists an introductory price of half these rates through
    # 2026-12-31, and this table used it at first. On 2026-11-10 the Cloud console's
    # live cost showed the standard rates for the same tokens, twice what the ledger
    # had counted, and the standard rates are what Google charged.
    # Google caches repeated input by itself and charges nothing for writing to that
    # cache, so a cache write is priced as plain input. Thinking is billed as output.
    "gemini-3.8-flash": ModelPricing(
        base=Rates(input=1.50, cache_write_5m=1.50, cache_write_1h=1.50, cache_read=0.15, output=7.50),
    ),
}


class UnknownModel(KeyError):
    pass


def _today() -> date:
    return date.today()


@dataclass(frozen=True)
class Usage:
    """Token counts for one request, as the API reports them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_write_5m_tokens + self.cache_write_1h_tokens + self.cache_read_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )


def resolve_model(model: str) -> str:
    """Map a served model id, such as a dated snapshot, to its row in the price table."""
    known = model if model in PRICING else next((k for k in PRICING if model.startswith((k + "-", k + "@"))), None)
    if known is None:
        raise UnknownModel(f"No price for model {model!r}. Add it to footnote/pricing.py from the provider's price page.")
    until = PRICING[known].valid_until
    if until is not None and _today() > until:
        raise UnknownModel(f"The price for {known!r} in footnote/pricing.py ended on {until}. Copy the new one from the provider's price page.")
    return known


def rates_for(model: str, prompt_tokens: int) -> Rates:
    pricing = PRICING[resolve_model(model)]
    if pricing.long_threshold is not None and prompt_tokens > pricing.long_threshold:
        assert pricing.long is not None
        return pricing.long
    return pricing.base


def cost_usd(model: str, usage: Usage, batch: bool = False) -> float:
    rates = rates_for(model, usage.prompt_tokens)
    dollars = (
        usage.input_tokens * rates.input
        + usage.cache_write_5m_tokens * rates.cache_write_5m
        + usage.cache_write_1h_tokens * rates.cache_write_1h
        + usage.cache_read_tokens * rates.cache_read
        + usage.output_tokens * rates.output
    ) / 1_000_000
    return dollars * (BATCH_DISCOUNT if batch else 1.0)


# Reserving budget before a call needs an upper bound on its size. The Claude 5
# tokenizer gives roughly 3 characters per token on prose and fewer on tables of
# numbers, so 1.5 characters per token is a safe ceiling for filings. Large
# prompts are counted exactly instead, because Claude Haiku 5.5 changes price
# at 100,000 prompt tokens and a guess near that line could be five times off.
CHARS_PER_TOKEN_FLOOR = 1.5
PROMPT_PAD_TOKENS = 512
EXACT_COUNT_ABOVE = 40_000


def prompt_token_bound(prompt_chars: int) -> int:
    return int(prompt_chars / CHARS_PER_TOKEN_FLOOR) + PROMPT_PAD_TOKENS


def worst_case_cost_usd(
    model: str, prompt_chars: int, max_tokens: int, batch: bool = False, prompt_tokens: int | None = None
) -> float:
    """An upper bound on what one request can cost. Used to reserve budget, never to report."""
    tokens = prompt_token_bound(prompt_chars) if prompt_tokens is None else prompt_tokens + PROMPT_PAD_TOKENS
    # Priced as if every prompt token were written to the one-hour cache, which
    # is the dearest way to send input. A request can then never cost more than
    # its reservation because of how it uses the cache.
    return cost_usd(model, Usage(cache_write_1h_tokens=tokens, output_tokens=max_tokens), batch=batch)
