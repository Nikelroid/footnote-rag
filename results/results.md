| System | Correct | Wrong | Refused | Flips | Evidence page read | $ per question | Median s | Graded answers |
|---|---|---|---|---|---|---|---|---|
| floor | 33.6% (26.7 to 40.7) | 56.0% (48.7 to 63.1) | 10.4% (6.4 to 15.1) | 13.3% | n/a | 0.00049 | 4.4 | 450 of 450 |
| rag | 80.4% (74.4 to 86.4) | 12.0% (7.3 to 16.9) | 7.6% (3.8 to 11.8) | 6.0% | 79.3% | 0.00135 | 3.1 | 450 of 450 |
| agent | 85.8% (80.4 to 90.9) | 10.4% (6.0 to 15.3) | 3.8% (1.3 to 6.7) | 6.0% | 84.4% | 0.00244 | 4.4 | 450 of 450 |
| ceiling | 90.7% (86.0 to 94.9) | 8.0% (4.2 to 12.2) | 1.3% (0.0 to 3.3) | 3.3% | 100.0% | 0.00039 | 2.0 | 450 of 450 |

Judge: judge-533f8883_claude-sonnet-5-5. Intervals are 95%, from resampling questions. Flips is the share of questions whose repeated runs disagree.
- floor: correct by split: dev 34.0% (22.9 to 45.8), test 33.3% (24.9 to 42.1), numeric check agrees with the judge on 94.2% of 156 answers.
- rag: correct by split: dev 79.7% (68.6 to 89.5), test 80.8% (73.1 to 87.9), numeric check agrees with the judge on 99.4% of 156 answers.
- agent: correct by split: dev 83.0% (72.5 to 92.2), test 87.2% (80.8 to 92.9), numeric check agrees with the judge on 100.0% of 156 answers.
- ceiling: correct by split: dev 88.9% (79.7 to 96.1), test 91.6% (86.2 to 96.3), numeric check agrees with the judge on 100.0% of 156 answers.

| Comparison | Change in correct | 95% interval | p | Questions |
|---|---|---|---|---|
| floor to rag | +46.9 points | +38.2 to +55.8 | 0.0000 | 150 |
| rag to agent | +5.3 points | +1.6 to +9.3 | 0.0101 | 150 |
| agent to ceiling | +4.9 points | +1.3 to +8.4 | 0.0097 | 150 |
