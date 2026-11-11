import pytest

from footnote.eval.numeric import final_answer_text, find_numbers, grade, parse_gold


@pytest.mark.parametrize(
    "text,value,percent,currency",
    [("$1577.00", 1577.0, False, True), ("1.9%", 1.9, True, False), ("-0.02", -0.02, False, False), ("(0.02)", -0.02, False, False), ("24.26", 24.26, False, False), ("0", 0.0, False, False)],
)
def test_parse_gold_reads_single_numbers(text, value, percent, currency):
    gold = parse_gold(text)
    assert gold.value == pytest.approx(value) and gold.percent is percent and gold.currency is currency


@pytest.mark.parametrize("text", ["Yes", "Data Center", "No. The quick ratio for 3M was 0.96 by Jun'23 close.", "AMCOR's Adj. EBITDA was $2,018mn in FY 2023"])
def test_parse_gold_leaves_text_answers_to_the_judge(text):
    assert parse_gold(text) is None
    assert grade(text, "anything 0.96").applicable is False


def test_numbers_glued_to_words_are_not_numbers():
    assert [n.value for n in find_numbers("In FY2018 and Q2 the figure was 1,577 million.")] == [1577e6]


@pytest.mark.parametrize(
    "gold,candidate",
    [
        ("$1577.00", "Final answer: $1,577 million"),
        ("$1577.00", "Final answer: $1.577 billion"),  # other units
        ("$0.40", "Final answer: $389 million"),  # the reference is in billions, to one decimal
        ("1.9%", "Final answer: 0.019"),  # a percentage written as a fraction
        ("0.66", "Final answer: 66%"),
        ("0.83", "Final answer: 0.826"),  # equal once rounded to the reference's precision
        ("7.9%", "Final answer: 7.8%"),  # one unit off in the last digit, and close
        ("-0.02", "Final answer: negative 0.02"),
    ],
)
def test_matches(gold, candidate):
    assert grade(gold, candidate).match is True


@pytest.mark.parametrize(
    "gold,candidate",
    [
        ("65.4%", "Final answer: 65.2%"),
        ("$1577.00", "Final answer: $1,580 million"),
        ("$0.40", "Final answer: 40%"),  # a percentage can never be a dollar amount
        ("0.01", "Final answer: 14,293"),  # no unit games on a ratio
        ("0", "Final answer: 1.0"),
        ("0.4%", "Final answer: 0.5%"),  # one unit off, but a quarter of the value
        ("24.26", "Final answer: I cannot determine this."),
    ],
)
def test_non_matches(gold, candidate):
    assert grade(gold, candidate).match is False


MILLIONS = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
BILLIONS = "How much did the company pay out in cash dividends? Answer in USD billions."


@pytest.mark.parametrize(
    "gold,candidate,question,expected",
    [
        ("$1577.00", "Final answer: $1,577 million", MILLIONS, True),
        ("$1577.00", "Final answer: $1.577 billion", MILLIONS, True),
        ("$1577.00", "Final answer: 1,577", MILLIONS, True),
        ("$1577.00", "Final answer: $1,577,000,000", MILLIONS, True),
        ("$1577.00", "Final answer: $1.577 million", MILLIONS, False),  # a thousand times too small
        ("$1577.00", "Final answer: $1,577 billion", MILLIONS, False),  # a thousand times too large
        ("$5466.00", "Final answer: $5,466,312 in USD millions", MILLIONS, False),
        ("$0.40", "Final answer: $389 million", BILLIONS, True),
        ("$0.40", "Final answer: 40%", BILLIONS, False),
        ("77.78", "Final answer: $77.78 million", "How much does Pfizer expect to pay in the future in USD million?", True),
        ("1,488", "Final answer: $1,488 million", None, True),  # the printed number, with a unit added
        ("1.9%", "Final answer: 1.9", None, True),  # the percent sign left out
        ("0.66", "Final answer: 0.66%", None, False),  # a ratio is not a percentage a hundred times smaller
        ("93.86", "Final answer: 93.9 days", None, True),
        ("93.86", "Final answer: 93.55 days", None, False),
        ("0.01", "Final answer: $14,293 million", None, False),
    ],
)
def test_the_unit_named_in_the_question_is_respected(gold, candidate, question, expected):
    assert grade(gold, candidate, question).match is expected


def test_question_unit():
    from footnote.eval.numeric import question_unit

    assert question_unit(MILLIONS) == 1e6 and question_unit(BILLIONS) == 1e9
    assert question_unit("What is the working capital ratio?") is None
    assert question_unit("Convert USD millions to USD billions.") is None  # two units named, so none is assumed


def test_final_answer_line_wins_over_numbers_in_the_reasoning():
    answer = "Operating income was $1,832 million and depreciation $636 million.\nFinal answer: $2,468 million"
    assert final_answer_text(answer) == "$2,468 million"
    verdict = grade("$1832.00", answer)
    assert verdict.scope == "final" and verdict.match is False
    assert grade("$1832.00", "It was $1,832 million.").scope == "any"
