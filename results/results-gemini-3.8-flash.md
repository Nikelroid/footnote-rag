| System | Correct | Wrong | Refused | Flips | Evidence page read | $ per question | Median s | Graded answers |
|---|---|---|---|---|---|---|---|---|
| floor | 62.0% (54.5 to 69.4) | 37.0% (29.7 to 44.4) | 1.0% (0.0 to 2.7) | 12.1% | n/a | 0.01446 | 11.9 | 396 of 450 |
| agent | 84.2% (78.4 to 89.8) | 9.7% (5.6 to 14.2) | 6.1% (2.7 to 10.1) | 5.4% | 82.0% | 0.02898 | 9.5 | 445 of 450 |
| ceiling | 92.3% (88.0 to 96.1) | 7.7% (3.9 to 12.0) | 0.0% (0.0 to 2.5) | 3.4% | 100.0% | 0.00710 | 4.2 | 446 of 450 |

Judge: judge-533f8883_claude-sonnet-5-5. Intervals are 95%, from resampling questions. Flips is the share of questions whose repeated runs disagree.
- floor: correct by split: dev 61.3% (48.7 to 73.3), test 62.3% (53.0 to 71.4), numeric check agrees with the judge on 96.8% of 126 answers, 54 answers were cut off or too long for the model, and are left out.
- agent: correct by split: dev 81.7% (71.2 to 90.8), test 85.5% (78.8 to 91.9), numeric check agrees with the judge on 99.4% of 156 answers, 5 answers were cut off or too long for the model, and are left out.
- ceiling: correct by split: dev 93.5% (86.9 to 98.7), test 91.7% (86.1 to 96.4), numeric check agrees with the judge on 100.0% of 156 answers, 4 answers were cut off or too long for the model, and are left out.

| Comparison | Change in correct | 95% interval | p | Questions |
|---|---|---|---|---|
| floor to agent | +23.2 points | +15.5 to +30.7 | 0.0000 | 146 |
| agent to ceiling | +7.5 points | +3.4 to +11.9 | 0.0005 | 149 |
