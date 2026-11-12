# How Footnote is evaluated

This file describes how a number gets into the results table and why it can be believed. The harness in `src/footnote/eval` implements every rule below, and the tests in `tests/` check them.

## What is measured

Each system answers the same 150 FinanceBench questions. For every answer the harness records four things.

- **Outcome.** One of three labels, the same three the FinanceBench authors used. Correct, incorrect, or refused. They are reported separately, because a wrong number and an honest "not found" are different failures.
- **Retrieval.** Whether the evidence pages were among the retrieved pages (hit@k, recall@k, MRR), and whether the right filing was found at all. This is computed without any model.
- **Cost.** Dollars per question, from the token counts the API returns and the published prices. The judge's cost is kept in its own column.
- **Latency.** Seconds for the final successful request. Retries and waiting are not counted.

## The data

FinanceBench has 150 public questions over 84 filings from 32 companies, with 50 questions of each of its three types. Every question has a reference answer and one to three evidence pages. The files are pinned to one commit of the benchmark repository, and each downloaded file is checked against its size and git hash in `manifests/financebench.json`.

The questions are split once into 51 dev and 99 test questions, at random inside each question type. The split never looks at a score. Prompts and settings are tuned on dev only. Results are reported on all 150, to compare with the paper, and on the 99 test questions, which no tuning has seen.

## How an answer is graded

A model judge reads the question, the reference answer, the analyst's justification when there is one, and the candidate answer. It returns a label and a short reason as structured JSON. The rubric in `src/footnote/eval/judge.py` follows the labelling rules of the FinanceBench paper. The same value in other units is correct, a very small rounding difference is correct, and a different but valid answer is correct. The rubric also marks a right value as incorrect when the reasoning behind it contradicts the reference.

Four cases never reach the judge.

- An empty answer is incorrect.
- An answer cut off at the token limit is counted and shown, and left out of the averages. It is not graded as if it were complete.
- A reply stopped by the API with a refusal is labelled refused.
- A request that failed for a reason outside the model goes to `errors.jsonl`. It is never scored as a wrong answer, and the next run retries it.

The report prints a table only for a complete run: every question, every repeat, every answer graded, all by one judge. A run with gaps raises an error instead. A partial table has to be asked for, and it marks its gaps.

The system under test is Claude Haiku 5.5. A model grading its own answers tends to favour them, so the judge is a different model, Claude Sonnet 5.5. Sonnet costs about twenty times more per graded answer, so the cheaper model was tested as a judge for development runs. The rule was written down before the comparison was run. Haiku may grade only if it reaches the 90% bar on the human labels, stays within 2 points of Sonnet there, agrees with Sonnet on at least 90% of Footnote's own answers, and reports an accuracy within 2 points of Sonnet's on them. It failed two of the four. It agreed with the human labels on 87.8% of 320 answers, against 90.3% for Sonnet. On 611 of Footnote's own answers it reported 67.4% correct where Sonnet reported 71.8%. The two judges agreed on 94.6% of those answers, and in 28 of the 33 disagreements Haiku marked an answer incorrect that Sonnet marked correct. So the cheap judge was not kinder to its own model's answers. It was harsher. Sonnet grades every answer in the results (`footnote grade`), and `footnote judge-compare` repeats the comparison of the two judges.

For the 52 questions whose reference answer is a single number, a second check runs with no model at all. It parses the numbers in the answer and compares them with the reference. It reads the unit the question asks for, such as USD millions, so the same amount in billions matches and an amount a thousand times off does not. It also accepts a percentage written as a fraction and a rounding difference of up to 0.1%. The report lists every answer on which the two graders disagree, and those are read by hand.

## Why the grader can be trusted

The FinanceBench repository ships 2,400 model answers that its authors labelled by hand (16 model settings on the 150 questions). They are the test the graders have to pass.

| Check | Status | Result |
|---|---|---|
| The loader reproduces the published table | Done | The correct and incorrect counts of all five GPT-4-Turbo rows match the paper, for example 29 correct with the shared index and 128 with the oracle page |
| Numeric check against human labels | Done | 98.0% agreement on 832 answers (96.8 to 98.7), kappa 0.96 |
| Judge against human labels, dev questions | Done | Sonnet agrees with the humans on 90.3% of 320 answers (86.6 to 93.1), kappa 0.85. It reports 44.4% correct where the humans report 46.6%, so it is slightly strict |
| Judge against human labels, test questions | Done | Sonnet agrees with the humans on 93.4% of 320 answers (90.2 to 95.7), kappa 0.90. It reports 46.2% correct where the humans report 49.1% |
| Four probes the judge must pass | Done | Sonnet and Haiku each pass all four |

The numeric check scored 96.5% on answers to dev questions and 98.7% on answers to test questions. Its rules were written after reading disagreements from both groups, so neither figure is a clean held-out number. None of the human-labelled answers ends with a "Final answer:" line, so this measures the check when it scans a whole answer. On Footnote's own answers it reads only that last line, and there it is checked against the judge.

The human labels are not perfectly consistent, which limits how high any agreement can go. On question `financebench_id_03849` the reference is 7.9%. The final answer 7.8% was labelled correct four times and incorrect twice.

The judge is accepted when it agrees with the human labels on at least 90% of the sampled answers and passes all four probes. The agreement, the confusion table and every disagreement are saved, and the rubric version is part of each grade so that grades from two rubric versions are never mixed.

On Footnote's own answers the numeric check and the judge were compared as well. In the four final runs they disagreed on 10 of 624 answers that have a numeric reference, 9 of them in the run without documents. All 10 were read. In each one the answer was close to the reference, such as 0.69 against 0.68, and the judge called it incorrect while the numeric check let it pass. The judge is the stricter of the two.

## Noise

Claude Haiku 5.5 does not accept sampling parameters, so temperature cannot be set to zero and the same question can get different answers on different runs. Every level is therefore run three times.

- The score is the mean over questions of each question's mean over its runs.
- The interval is a 95% bootstrap interval that resamples whole questions and keeps a question's runs together.
- The flip rate is the share of questions whose runs disagree. It is the run-to-run noise.
- Two levels are compared on the same questions, with a bootstrap interval on the difference and a sign-flip permutation test.

The interval and the test were checked on simulated benchmarks where the truth is known (`footnote stats-check`, 2,000 simulations). In the simulation 45% of the questions are always answered correctly, 35% never, and 20% succeed on some runs and fail on others. With 150 questions and three runs, the 95% interval contained the true accuracy in 94.5% of the simulations, and the paired test raised a false alarm in 4.3% of the comparisons between two equally good systems.

The same simulation shows how large a gain this benchmark can see. Each cell is the share of simulations in which the paired test detected the gain.

| True gain | 150 questions, 3 runs | 150 questions, 1 run | 99 test questions, 3 runs |
|---|---|---|---|
| 3 points | 20% | 9% | 11% |
| 5 points | 67% | 32% | 34% |
| 8 points | 96% | 65% | 78% |
| 10 points | 99.7% | 84% | 94% |

Three runs double the chance of seeing a 5-point gain. A gain under about 5 points cannot be claimed from this benchmark, and is reported as not distinguishable. The simulated gain comes from questions the first system never answers and the second always does. A real gain that also loses some questions is harder to detect, so these figures are an upper limit.

## Decided in advance

These choices are fixed before any model is run, so that the results cannot steer them.

- The main number is the share of correct answers on all 150 questions, over three runs.
- The main comparison is each level against the level before it, on the same questions.
- The share of wrong answers is reported next to the share of correct ones. A level that gains correct answers by guessing more is not an improvement.
- Results on the 99 test questions are reported for every level, whether or not they agree with the full set.
- A level is kept in the table when it does not help. The table is a record of what was tried.

## Cost accounting

Every request goes through one function. It reserves the worst-case cost in a SQLite ledger before the request is sent, and replaces it with the real cost afterwards. A request that would pass the $90 cap, or the cap of its own budget line, is refused before it is sent. Runs on Gemini are counted in a second ledger with a cap of its own, because they are paid from Google Cloud credit. That ledger counted the tokens exactly and the price wrongly. It used Google's introductory price, and Google charged the standard price, which is twice as high. The report therefore prices every answer from its stored token counts and the current price table, and does not trust the dollars written on the day of the run. `keys.md` has the details. The Claude ledger was compared with the Console during the work and matched. Only plain text can be sent through this function, because anything else has no size bound, and a large prompt is counted exactly before its budget is reserved. The prices in `src/footnote/pricing.py` are copied from the Claude pricing page and include the higher rate that Claude Haiku 5.5 charges for prompts over 100,000 tokens. The harness also checks that the model named in each response is the model that was asked for.

## Guarding against tuning to the benchmark

- Tuning happens on the 51 dev questions only.
- The floor run uses no documents. It shows how much the model already knows about these filings from training, which is the part of any score that retrieval did not earn.
- A run on a second benchmark that the system has never seen was planned and then cut, to keep the project small. The 99 test questions are the only held-out check.
- The changing-filings test uses edited pages whose new answers exist nowhere else, so a model cannot recall them. See `changing-documents.md`.

## Known limits

- 150 questions give wide intervals. A level has to gain several points to show.
- One annotator labelled each answer in the paper, and the labels contain some inconsistency.
- The judge and the answering model come from the same family. The human-label check measures the judge on answers from GPT-4, Claude 2 and Llama 2, not on Haiku's own answers. That gap is only partly covered. On Haiku's own answers the judge was compared with the numeric check, and the 10 disagreements were read by hand, as described above.
- The judge is a little strict. On the human-labelled answers it reports 2.2 to 2.9 points fewer correct answers than the human annotators did.
- Where Claude Sonnet 5.5 is the answering model, in the comparison of models, it also grades its own answers. The rubric compares each answer with a reference, which leaves little room for taste, but it is not an independent check.
- On one long answer the judge ran out of room while it was still reasoning. It now asks once more with four times the room, under the same rubric.
- The numeric check compares magnitudes and ignores signs. When a reference has one significant digit, such as 0.4%, any answer that rounds to the same digit matches.
- A request that fails after it was sent may still have been billed. The ledger keeps it at its worst-case cost, so the ledger can overstate the bill and should not understate it. A spend limit set on the API key's workspace in the Console is the final backstop.
