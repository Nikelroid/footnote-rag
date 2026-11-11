| System | Correct | Wrong | Refused | Flips | Evidence page read | $ per question | Median s | Graded answers |
|---|---|---|---|---|---|---|---|---|
| floor | 32.0% (20.9 to 44.4) | 56.2% (43.8 to 68.6) | 11.8% (4.6 to 20.3) | 13.7% | n/a | 0.00048 | 4.2 | 152 of 153 |
| rag | 82.4% (72.5 to 92.2) | 14.4% (5.9 to 24.2) | 3.3% (0.0 to 8.5) | 0.0% | 84.3% | 0.00132 | 3.0 | 153 of 153 |
| agent | 84.3% (74.5 to 93.5) | 12.4% (3.9 to 22.2) | 3.3% (0.0 to 8.5) | 3.9% | 85.0% | 0.00242 | 4.7 | 153 of 153 |
| ceiling | 88.2% (78.4 to 96.1) | 9.8% (2.0 to 17.6) | 2.0% (0.0 to 5.9) | 0.0% | 100.0% | 0.00038 | 1.9 | 153 of 153 |

Judge: judge-533f8883_claude-sonnet-5-5. Intervals are 95%, from resampling questions. Flips is the share of questions whose repeated runs disagree.
- floor: correct by split: dev 32.0% (20.9 to 44.4), numeric check agrees with the judge on 98.1% of 53 answers, 1 answers were cut off or too long for the model, and are left out.
- rag: correct by split: dev 82.4% (72.5 to 92.2), numeric check agrees with the judge on 100.0% of 54 answers.
- agent: correct by split: dev 84.3% (74.5 to 93.5), numeric check agrees with the judge on 100.0% of 54 answers.
- ceiling: correct by split: dev 88.2% (78.4 to 96.1), numeric check agrees with the judge on 100.0% of 54 answers.

| Comparison | Change in correct | 95% interval | p | Questions |
|---|---|---|---|---|
| floor to rag | +50.3 points | +35.9 to +64.1 | 0.0000 | 51 |
| rag to agent | +2.0 points | -2.6 to +7.8 | 0.7536 | 51 |
| agent to ceiling | +3.9 points | -3.3 to +11.1 | 0.4388 | 51 |
