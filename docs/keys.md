# Keys

Search, indexing, `retrieval-check`, `updates-demo` and the tests need no key. Answering and grading need one key, and a second key is optional. This page says how to make each one, what it is allowed to do, where it goes, and what limits the spending. Every step comes from the provider's own documentation, read on 2026-11-10. A menu label may have changed since then. The sources are at the bottom.

| Key | What uses it | Paid from |
|---|---|---|
| `ANTHROPIC_API_KEY`, required | Every command that answers or grades (`footnote run`, `grade`, `stale-run`, `judge-smoke`, `judge-check`), `POST /ask`, the demo page, the ADK agent | Credit in the Claude Console |
| `GOOGLE_API_KEY`, optional | Anything run on Gemini. That is the ADK agent with `--gemini`, and `footnote run` or `footnote stale-run` with `--model gemini-3.8-flash` | Google Cloud billing or credit |

## Where the keys go

Both go in one file, `.env`, in the project folder. Nothing else reads them.

```bash
cp .env.example .env
```

Open `.env` in an editor and paste each key directly after its equals sign, with no spaces and no quotes. `.env` is listed in `.gitignore`, so git never sees it. `pytest tests/test_no_secrets.py` checks that no file git would publish holds a key.

## What limits the spending

A paid command prints its plan and sends nothing until `--confirm` is added. Every model call then goes through a ledger. The call reserves its worst-case cost before it is sent, the real cost replaces it afterwards, and a call that would pass a cap is refused. `footnote budget` shows what has been spent.

The caps are in `src/footnote/config.py`. They are $90 for the Claude API and $35 for Gemini, the limits this project was run under. Lower them to your own limit before the first paid run.

This is what the experiments in the README cost, as the ledger recorded it.

| Part | Cost |
|---|---|
| Runs on the 51 tuning questions, where the design choices were made | $0.71 |
| The four systems on all 150 questions, three runs each, with Claude Haiku 5.5 | $2.10 |
| The judge, Claude Sonnet 5.5, for every graded answer and for the check against human labels | $15.05 |
| The same questions with Claude Sonnet 5.5 as the answering model, one run | $9.41 |
| The tests on pages that mislead, with Claude Haiku 5.5 and Claude Sonnet 5.5 | $28.21 |
| Total on the Claude API | $55.48 |
| Everything on Gemini 3.8 Flash, at Google's standard price | $38.83 |

## Key 1. Claude, required

Everything here is in the Claude Console, https://platform.claude.com.

1. **Credit.** Open [Billing](https://platform.claude.com/settings/billing) and buy a small amount of credit. Leave auto-reload off. With it off, the account cannot spend more than its balance. Some Claude subscriptions include API credit that can be linked to a Console organization instead, and the source list has the page for that.
2. **Workspace.** Open [Workspaces](https://platform.claude.com/settings/workspaces), press Create workspace and name it `footnote`. A separate workspace is needed because the default workspace cannot have a spend limit.
3. **Spend limit.** On the Workspaces page select `footnote`. In the panel that opens press Spend limits and set the monthly limit a little above the cap in `config.py`. The project stops itself at its cap, so the workspace limit is only reached if the project's own count is wrong.
4. **Key.** Open [API keys](https://platform.claude.com/settings/keys) and press Create key. Fill it in as in the table below.
5. **Paste.** Copy the key and paste it after `ANTHROPIC_API_KEY=` in `.env`.

| Field | Value |
|---|---|
| Name | `footnote-laptop` |
| Expiration | 30 days |
| Linked account | Yourself |
| Workspace | `footnote`, and only that one |

**Permissions.** A Claude key has no permission boxes to tick. It acts as you, inside the one workspace it belongs to. The workspace choice matters for a second reason. A key that is not tied to one workspace needs a workspace header on every request, and this project does not send one. Do not create an Admin key, the kind that starts with `sk-ant-admin`. Nothing here needs it.

**Three limits, each on its own.** The prepaid balance with auto-reload off, the limit on the workspace, and the project's own ledger. The key also stops working after 30 days, and Anthropic sends an email 7 days before that.

## Key 2. Google, optional

Everything here is in the Google Cloud console, https://console.cloud.google.com. Sign in with the Google account whose billing account will pay.

1. **Check how the calls will be paid.** Open [Billing](https://console.cloud.google.com/billing), choose the billing account, then Credits in the left menu. A free trial credit shows instead in the Free credit box on the Overview page. Note the remaining value, the end date and the scope. If you want credit to pay and its scope leaves out Agent Platform (Vertex AI), use the fallback at the end of this section.
2. **Project.** Pick a project that is linked to that billing account, or create one named `footnote` and link it. Billing must be enabled on the project.
3. **Enable one API.** Open APIs & Services, then Library, search for Agent Platform API and press Enable. Its service name is `aiplatform.googleapis.com`, and older pages call it Vertex AI API. No other API is needed.
4. **Service account.** This is the identity the key acts as. Open IAM & Admin, then Service Accounts, and press Create service account. Name it `footnote-adk` and press Create and continue. Under Permissions press Select a role and choose Agent Platform Express User (Beta). Its id is `roles/aiplatform.expressUser`. Give it no other role. Press Continue, then Done.
5. **Key.** Open APIs & Services, then Credentials. Press Create credentials, then API key. Name it `footnote-adk-key`. Tick Authenticate API calls through a service account, press Select service account, choose `footnote-adk`, press Select, then Create.
6. **Restrict the key.** On the Credentials page open the new key. Under API restrictions choose Restrict key, select only Agent Platform API and save. Leave Application restrictions empty, because a laptop's IP address changes.
7. **Budget alert.** Open Billing, then Budgets & alerts, and press Create budget. Choose Alerts only, name it `footnote`, set the scope to the project and the amount to your cap. Google states that this kind of budget does not cap spending. It only sends email.
8. **Paste.** Copy the key and paste it after `GOOGLE_API_KEY=` in `.env`.

Then one question is one command.

```bash
python -m footnote.app.adk_agent --gemini "What is the FY2018 capital expenditure for 3M?"
```

**Permissions.** The key can do what its service account can do, and that account has one role on one project. The API restriction is a second lock on the same door. Do not give the account Owner or Editor, and do not download a JSON key file for it. Nothing here needs one.

**What limits the spending.** The project counts Gemini calls in a ledger of their own, `ledger-google.sqlite`, with its own cap. The Claude budget is not touched, and `footnote budget` shows both. The ledger counts tokens. It cannot see whether Google charges them to credit or to a card, so check that once under Billing, on the Reports or Credits page, a day after the first run.

**What Google Cloud credit can and cannot pay for.**

- It cannot pay for the Claude key. That is a different company and a different bill.
- A welcome or free trial credit cannot pay for Gemini through a key made in Google AI Studio. Google's billing page says so directly.
- A free trial credit cannot pay for Claude served by Google Cloud either. Google lists partner models as excluded.
- It can pay for Gemini on Google Cloud's own model platform. The console calls it Agent Platform, and until recently it was Vertex AI. The steps above set up this path.
- Google's free trial page says the trial credit stops covering this platform for anyone who signs up for the trial or upgrades to a paid account on or after November 16, 2026.

**If step 5 is refused.** Some organizations block keys that are tied to a service account, with a policy named `iam.managed.disableServiceAccountApiKeyCreation`. Turning it off needs the Organization Policy Administrator role. The path is IAM & Admin, then Organization policies, then the policy, Actions, Edit policy, Override parent's policy, Add a rule, Enforcement set to Off, Done, Set policy. If you do not have that role, use the fallback.

**Fallback without Google Cloud billing.** Make a key at https://aistudio.google.com/apikey in a project that has no billing, paste it after `GOOGLE_API_KEY=` and add the line `GOOGLE_GENAI_USE_ENTERPRISE=false` to `.env`. That is Google's free tier, which costs nothing inside its rate limits. Google uses free tier content to improve its products. That is acceptable here because the filings and the questions are public. The ledger still counts these calls at the list price.

## The Gemini model and its price

Everything on Gemini uses `gemini-3.8-flash` on Google Cloud's global endpoint. Google's price page, read on 2026-11-10, lists an introductory price of $0.75 per million input tokens and $3.75 per million output tokens through December 31, 2026, and a standard price of $1.50 and $7.50 after that.

The ledger first counted at the introductory price, and that was a mistake. On the same day the Cloud console's live cost showed the standard price for the same tokens. Priced at the standard rate, the ledger's token counts give the console's figure to the cent, $38.83 where the ledger had said $19.41. Google's billing pages say live costs are gross estimates that leave out savings and credits, so the final bill may still be the lower figure. Until a final bill says so, the code counts at the standard price.

The lesson for any new provider is to send a few calls first and compare the ledger with the provider's own cost page before sending thousands.

## What has been tested

Both keys were tested against the live services on 2026-11-10. The Claude key answered and graded the benchmark runs. The Google key ran the ADK agent with `--gemini`, the benchmark runs and the page tests on `gemini-3.8-flash`. In both cases the cost the project recorded was computed from the token counts the service returned.

## Sources

- Claude keys, workspace scope and expiry. https://platform.claude.com/docs/en/manage-claude/authentication
- Claude workspaces and spend limits. https://platform.claude.com/docs/en/manage-claude/workspaces and https://platform.claude.com/docs/en/api/rate-limits
- Buying Claude credit and auto-reload. https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage
- API credit included with some subscriptions. https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers
- What the Google Cloud free trial credit covers. https://docs.cloud.google.com/free/docs/free-cloud-features
- A Google Cloud API key tied to a service account. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=standard
- Google Cloud budgets. https://docs.cloud.google.com/billing/docs/how-to/budgets
- Gemini prices on Google Cloud. https://cloud.google.com/vertex-ai/generative-ai/pricing
- Credits and the Gemini API in AI Studio. https://ai.google.dev/gemini-api/docs/billing
