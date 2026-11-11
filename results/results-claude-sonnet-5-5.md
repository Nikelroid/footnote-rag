| System | Correct | Wrong | Refused | Flips | Evidence page read | $ per question | Median s | Graded answers |
|---|---|---|---|---|---|---|---|---|
| floor | 57.8% (49.7 to 66.0) | 38.8% (30.6 to 46.9) | 3.4% (0.7 to 6.8) | n/a | n/a | 0.01067 | 6.7 | 147 of 150 |
| agent | 86.7% (81.3 to 92.0) | 11.3% (6.7 to 16.7) | 2.0% (0.0 to 4.7) | n/a | 84.7% | 0.04512 | 4.8 | 150 of 150 |
| ceiling | 94.0% (89.9 to 97.3) | 6.0% (2.7 to 10.1) | 0.0% (0.0 to 2.5) | n/a | 100.0% | 0.00694 | 2.4 | 149 of 150 |

Judge: judge-533f8883_claude-sonnet-5-5. Intervals are 95%, from resampling questions. Flips is the share of questions whose repeated runs disagree.
- floor: correct by split: dev 61.2% (46.9 to 75.5), test 56.1% (45.9 to 65.3), numeric check agrees with the judge on 92.0% of 50 answers, 3 answers were cut off or too long for the model, and are left out.
- agent: correct by split: dev 82.4% (72.5 to 92.2), test 88.9% (82.8 to 94.9), numeric check agrees with the judge on 100.0% of 52 answers.
- ceiling: correct by split: dev 94.1% (86.3 to 100.0), test 93.9% (88.8 to 98.0), numeric check agrees with the judge on 100.0% of 52 answers, 1 answers were cut off or too long for the model, and are left out.

| Comparison | Change in correct | 95% interval | p | Questions |
|---|---|---|---|---|
| floor to agent | +28.6 points | +20.4 to +37.4 | 0.0000 | 147 |
| agent to ceiling | +7.4 points | +3.4 to +12.1 | 0.0034 | 149 |
