# minty-shared-py

Minty's shared Python for every backend: Flask (`Minty`) and the four Django APIs. This is where
the modernisation plan's `minty_shared` starts (`Minty/docs/modernisation/modernisation_plan.md`,
Part 2). Models, enums, the permission policy and entitlements are **not** here: they belong to
the planned `minty-db` package.

| Module | Needs | What |
|---|---|---|
| `minty_shared.logctx`, `minty_shared.log` | stdlib | the one log line, its trace/user/entity/step context, loguru's call shape over `logging` |
| `minty_shared.config` | stdlib | `APP_ENV`, flags, the `SECRET_KEY` placeholder refusal, `DATABASE_URL` (identifier-only `?schema=`), `S3_URL`, `SMTP_URL`, `PREVIEW_ORIGIN_REGEX`, and Django's `DATABASES`/`EMAIL_*` shapes |
| `minty_shared.money` | stdlib | `HKD 1,234.50`: the code, a non-breaking space, the currency's places; `format_price` drops a whole `.00`; no currency says so |
| `minty_shared.time` | stdlib + `tzdata` | `DEFAULT_ENTITY_TIMEZONE`, a stored instant on a company's calendar, `validate_timezone`, `COUNTRY_TIMEZONES` + `country_timezone` (a company's zone from its country; tracker keeps the same map) |
| `minty_shared.policy` | stdlib | (v0.5; v0.6 exempts admin) the roles and permissions every backend applies: `Role`, `Permission`, the rank ladder, `PERMISSION_RULES`, `READONLY_ALLOWED_PERMISSIONS`, `normalize_role`, `role_allows`, and the strict `can_manage_role_assignment` / `assignable_roles` - you manage only roles below yours (owner, 2026-11-12), except that an admin also manages admins, never a super_admin (owner, 2026-11-13). Each service keeps only its membership reads |
| `minty_shared.accounts` | stdlib | (v0.5) a company's account rules: `XERO_TYPE`, `CODE_START`, `TYPE_WORDS`, the limits, `check_fields` (Part 11 sentences per field), `clash_fields`, `next_code` |
| `minty_shared.tokens` | extra `jwt` | `decode_bearer` (60 s clock skew everywhere), `service_assertion`, `assertion_headers` |
| `minty_shared.fx` | stdlib | (v0.8, owner 2026-11-19) the exchange rate for an AI-read amount in another currency: `exchange_for(suggestions, report_date, today, places, max_amount)` - the readers' one step after a read: the amount converted at Frankfurter v1's rate (the European Central Bank's, 30 currencies, a weekend or holiday gets the previous business day) with `exchange` saying how (`original_currency`, `original_amount`, `rate`, `rate_date`, `date_basis`, `source`), or emptied with `exchange_failed` when there is no rate; `rate_date` (the document's day unless empty, in the future or a year from the report), `frankfurter_rate` (one `urllib` call, 5 s, `FX_API_URL` moves it; any failure = None, logged), `convert` (half up to the company currency's places) |
| `minty_shared.storage` | extra `storage` | `put_file`: one file into B2 byte for byte (no downsizing, no conversion), ContentType set, retried twice; `make_client`; `StorageError` |
| `minty_shared.ai` | extra `ai` | the engine every document reader shares: `ReaderConfig(prefix)` (`<PREFIX>_ENABLED`, `_MODEL` as the fallback model, ... read per call), `read(models=[primary, backup])` (cache-ordered request, one retry for retryable classes only; v0.7: the next model when one errors, times out, is rate-limited or replies unusably - `FAILOVER_REASONS`), providers behind `Provider`/`PROVIDERS` with only `APPROVED_PROVIDERS` ever called (Gemini today; `<provider>:<model>` names another), `Extraction.answered`/`tokens()`/`attempts_text()`, `field`/`band`, `REASON_*` (v0.7 `REASON_LIMIT_REACHED`), `log_read` (numbers, never values; the model that answered and the chain tried), `sniff_mime`. v0.7 removed `RateLimiter` and `_RATE_*`: the shared allowance replaces them |
| `minty_shared.model_input` | extra `ai` | `prepare_for_model`: the small copy a model reads (image capped at 2048 px and saved once as JPEG, decoded at reduced scale when large; PDF page 1). Never the stored file |
| `minty_shared.django.middleware` | extra `django` | `RequestLoggingMiddleware` (settings `MINTY_HTTP_LOGGER`, optional `MINTY_LOGGABLE_PATH`) |
| `minty_shared.django.errors` | extra `django` | the envelope below: `register_error_handlers`, `ApiRefusal`, `validation_message`, `field_messages` |
| `minty_shared.django.client` | extra `django` | `forward`/`proxy`: call another Minty backend as the current caller |
| `minty_shared.django.ai_allowance` | extra `django` (+ `minty_db` installed in the service) | (v0.7) the AI readers' allowance and settings, from minty-db's `ai_policy`/`ai_grant`/`ai_read`: each company reads against its own grants (`default_grant` given once on its first read; `give_grant` for top-ups), Petty Cash and Payment Request together. `load_policy` (fails closed: no usable row = reading off, logged), `reader_status` (what a page needs beside the read button: `enabled`, `percent_left`, `blocked`, `resumes_at`), `reserve` (under a per-company advisory lock; charges the oldest grant with room) and `finish` (only a read a model answered is charged to a grant; every attempt's tokens are recorded), `parse_rules`/`parse_grant`/`summarise` (extra caps over every company: reads, tokens, a rolling `period_days`) |
| `minty_shared.testing` | extra `testing` | the Postgres-harness pytest plugin (`pytest_plugins = ["minty_shared.testing.pg"]`) and the source-walking guards |

## The error envelope

Every API answers a refusal the same way:

```
{"error": "<a sentence to show>", "fields": {"amount": "The amount is still empty. Add the amount to continue."}}
```

- `fields` appears only when input is refused. Any key a service routes on (Petty Cash's `to`,
  `report_id`) rides along.
- Schema refusals are **422**. A rule's own refusal keeps the status the rule names. 401 is
  `{"error": "Unauthorized"}`. 500 is one house sentence, with the detail in the log, never in the
  response.

## The log line

```
2026-10-22T08:42:15.123Z INFO [trace=a1b2c3d4] [user=<uuid>|entity=<uuid>|step=payment.approve] Bill approved bill=…
```

| Field | Set by | Absent |
|---|---|---|
| timestamp | the record, in UTC with milliseconds | — |
| `trace` | `start_trace(request.headers.get("X-Request-ID"))` at the start of a request, scheduler pass or command. A header of 8–32 lowercase hex digits is kept; anything else is replaced with 8 fresh digits, so a header can't forge a line | `-` |
| `user`, `entity` | `bind(user=…, entity=…)` in the auth layer, only **after** the request is authorised. Full ids | `-` |
| `step` | `with log_step("payment.approve"):` around a unit of work; the route name is the fallback (`set_default_step`) | `-` |

The values live in `ContextVar`s, so every line written during the request carries them,
including Django's and werkzeug's own. `ContextFilter` is attached to the handlers, not the
loggers, so propagated records get it too.

- **Across services:** send `outgoing_headers()` on every call to another Minty backend, and
  never to Xero, Stripe or anyone else. Each backend echoes `X-Request-ID` on its response.
- **Threads:** a new thread starts with an empty context. Use
  `threading.Thread(target=carry_context(work))` to keep the caller's trace.
- **Configuration:** `logging_config({...})` returns the `dictConfig` dict:

  ```python
  LOGGING = logging_config(
      {"billing-api": "core.log", "billing-api.http": "api.log"},
      log_dir=BASE_DIR / "logs",
      no_propagate=["billing-api.http"],
      level=LOG_LEVEL,
  )
  ```

## `minty_shared.log`

`get_logger("minty")` returns a `BraceLogger`: loguru's call shape (`logger.info("{} issued", n)`,
`logger.exception(...)`) over a standard `logging.Logger`. It is what Flask and the subscription
engine log through, so their messages stay byte-identical.

## Using it

Pin a tag in `requirements.txt` (tags are **immutable**: a change is a new tag):

```
minty-shared[jwt,django,testing] @ https://github.com/minty-live/minty-shared-py/archive/refs/tags/v0.2.tar.gz
```

A Django API takes all three extras; Flask takes `[jwt]`.

The tag's **archive**, not `git+https://…@v0.1`: the `python:3.13-slim` images the Django APIs
build from have no git, so a `git+` pin installs in CI and fails on Render. The repo is public, so
the archive downloads with no key. A project built by hatchling (Minty) also needs
`[tool.hatch.metadata] allow-direct-references = true`. While developing against a
local checkout, run `python -m pip install -e C:\Github\minty-shared-py` in the consumer's venv.

## Developing

```
uv venv --seed --python 3.13 .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m pyright
```
