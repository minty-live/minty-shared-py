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
| `minty_shared.tokens` | extra `jwt` | `decode_bearer` (60 s clock skew everywhere), `service_assertion`, `assertion_headers`; (v0.8) `session_alive(sid, accounts_api_url, secret)`: whether the hub session behind a token's `sid` lives - minty-accounts-api's `POST /api/internal/session/touch` (scope `session-touch`, stdlib `urllib`), which also keeps it alive, asked at most once a minute per `sid` per process (`SessionGate`, `SESSIONS`); 410 = ended (refuse the token, 401); no `sid` = alive until the token expires; unreachable or any other answer (404 included) = logged at ERROR (`SessionCheckError`) and, (v0.9, owner 2026-11-19) honoured only while that sign-in was confirmed alive in the last 5 minutes (`SESSION_CHECK_GRACE_SECONDS`), else `SessionCheckUnavailableError` - each API answers it 503 with `SESSION_CHECK_UNAVAILABLE` ("Minty can't check your sign-in just now. Try again in a minute."); never a sign-out, and a sign-in known to have ended stays ended. Tests patch `touch_over_http` and call `SESSIONS.forget()`. (v0.8, owner 2026-11-19) `session_ended(token, accounts_api_url, secret, *, touch=None)`: True when the token's signature verifies (expiry ignored, for this question only) and its `sid`'s sign-in has ended - the 401 then carries `"reason": "session_expired"` (`SESSION_EXPIRED_REASON`; `mark_session_ended(request)` in the auth, `unauthorized_reason(request)` in the 401 handler) and each app shows "Your session has expired" instead of renewing quietly |
| `minty_shared.names` | stdlib | (v0.14, the user 2026-11-23) one name for a contact however it is typed: `name_key` (NFKC, case ignored, "&" = "and", only letters and digits kept - "7 Eleven" == "7-Eleven"; a name with no letter or digit is "" and matches nothing), `same_name(a, b)`, and `search_term` - the longest word, for Xero's substring `GET /Contacts?searchTerm=`. Both APIs' supplier adds and readers compare names with it |
| `minty_shared.fx` | stdlib | (v0.8, owner 2026-11-19) the exchange rate for an AI-read amount in another currency: `exchange_for(suggestions, report_date, today, places, max_amount)` - the readers' one step after a read: the amount converted at Frankfurter v1's rate (the European Central Bank's, 30 currencies, a weekend or holiday gets the previous business day) with `exchange` saying how (`original_currency`, `original_amount`, `rate`, `rate_date`, `date_basis`, `source`), or emptied with `exchange_failed` when there is no rate; `rate_date` (the document's day unless empty, in the future or a year from the report), `frankfurter_rate` (one `urllib` call, 5 s, `FX_API_URL` moves it; any failure = None, logged), `convert` (half up to the company currency's places) |
| `minty_shared.storage` | extra `storage` | `put_file`: one file into B2 byte for byte (no downsizing, no conversion), ContentType set, retried twice; `make_client`; `StorageError` |
| `minty_shared.images` | extra `images` (`ai` brings it too) | (v0.19, the user 2026-11-24) iPhone HEIC/HEIF photos: stored as uploaded, made JPEG on the fly only where they can't be used as they are - never a stored copy. `sniff_heif(head)` (the `ftyp` brand: heic/heix/hevc/hevx/heim/heis/hevm/hevs -> `image/heic`, mif1/msf1 -> `image/heif`; AVIF is not HEIC), `is_heif(mime)`, `to_jpeg(data, max_side=4096, quality=90)` (upright, transparency on white; a service's preview answers `?as=jpeg` with it for a browser that can't draw HEIC), `for_xero(data, mime, name)` (a HEIC as `<stem>.jpg`, Xero takes no HEIC; anything else unchanged), `check_decodes` (the upload's check), `jpeg_name`, `ImageError`. Decoder `pi-heif` (decode-only, BSD-3), imported on first use |
| `minty_shared.ai` | extra `ai` | the engine every document reader shares, on **Vertex AI only** (v0.11, owner 2026-11-20: `models.generate_content` for the project `GOOGLE_CLOUD_PROJECT`, credentials through `GOOGLE_APPLICATION_CREDENTIALS`; no call without the project; v0.12, owner 2026-11-23: always on Vertex's `global` endpoint (`VERTEX_LOCATION`) - no region setting, `ai_policy.location` is no longer read (Flash-Lite is not served in `asia-southeast1`); the direct Gemini API and its `GEMINI_API_KEY`/`EXPENSE_AI_*` route settings are gone): `ReaderConfig(label, enabled, timeout_seconds, max_output_tokens, thinking_level, max_file_mb)` - plain values from `ai_policy` (`ai_allowance.Policy.reader`), no environment; `read(models=[primary, backup])` (no models = no call) (cache-ordered request, one retry for retryable classes only; v0.7: the next model when one errors, times out, is rate-limited or replies unusably - `FAILOVER_REASONS`), providers behind `Provider`/`PROVIDERS` with only `APPROVED_PROVIDERS` ever called (Gemini today; `<provider>:<model>` names another), `Extraction.answered`/`tokens()`/`attempts_text()`, `checked` (v0.11 removed `field`/`band`/`clamp_confidence` with the confidence cut-offs), `REASON_*` (v0.7 `REASON_LIMIT_REACHED`), `log_read` (numbers, never values; the model that answered and the chain tried), `sniff_mime`. v0.20 (the user, 2026-11-24): `read(labels=...)` is required and every Vertex call carries it - `request_labels(entity_id, FEATURE_PETTY_CASH | FEATURE_PAYMENT_REQUEST)` gives exactly `env` (from `APP_ENV`), `feature` and `entity_id` (lowercase), each 1-63 of `[a-z0-9_-]`; anything else is a ValueError before any call. For Cloud Billing's Group by label, beside `ai_read.estimated_cost`. Also v0.20: each model's price comes from `ai_policy.model_prices` (minty-db v0.19; `ai.ModelPrice(input, output, cached_input)`, USD per 1M tokens, Vertex AI's global prices) through `ReaderConfig.prices` - the `PRICING` constant (the Gemini API's prices) is gone; `estimate_cost(price, input, output, thought, cached)` charges cached input at `cached_input`. `ai_allowance.parse_prices`/`parse_policy(..., model_prices)` refuse a policy naming a model without a price (reading OFF, logged), so a read's cost is never left out. v0.7 removed `RateLimiter` and `_RATE_*`: the shared allowance replaces them |
| `minty_shared.ai_prompt` | extra `ai` | (v0.10, owner 2026-11-20) the one prompt both document readers send, in the owner's words: `build(kind, entity, accounts, suppliers)` -> `(instruction, context, aliases)` for `"receipt"` (seven fields) or `"invoice"` (plus `due_date`); the lists carry short ids (`acc_01`, `con_01`, zero-padded to the list's width) in a byte-stable order, and `Aliases.account/.supplier` map an answer back to the caller's key (case and spaces forgiven; a made-up id maps to nothing); `SimpleReading`/`InvoiceReading` are the reply models. No confidence: `minty_shared.ai.checked(value, extra)` fills every answered field in the `medium` band. v0.13 (the user, 2026-11-23): the account is always the closest one in the chart, even when unsure (never `""`); `supplier_name` is the name as printed. A supplier the company does not have is added when the record is saved: `ai.new_supplier_token(SECRET_KEY, entity_id, name)` signs the read name for that company (7 days, a local draft's life) and `ai.new_supplier_name(SECRET_KEY, token, entity_id)` gives it back on save, or None when altered, for another company or expired. v0.14 (the user, 2026-11-23): `supplier_contact_id` is the same **or a similar** supplier in the list (another spelling, language or branch), `""` only if none is. v0.15 (owner's launch decisions, 2026-11-23, after IRD's "A Guide to Keeping Business Records"): `date` is a printed billing period's last day, otherwise the payment date (an invoice: the issue date); `description` is inferred from the supplier and context when no items are listed, with `account_id` chosen to match, and `""` only when it can't be; the reply adds `description_inferred` and `document_count`. `ai.evidence_refusal(suggestions)` gives `REASON_SEVERAL_DOCUMENTS` (`several_documents`, more than one document) - the only refusal since v0.21 (the user, 2026-11-25: the description is not mandatory, `no_description` is gone); both readers then answer `{suggestions: None, reason}`, and the read still counts. v0.17 (the user's wording, 2026-11-24): the prompt is the user's text word for word - `account_id` is "" when nothing in the chart matches (the 23 Nov "closest, never empty" rule is undone), one date line for receipts and invoices (a billing period's last day, otherwise the payment date), `supplier_name` and `description_inferred` bare; an invoice adds only the `due_date` line and key. v0.18 (the user, 2026-11-24): a last line, `document_type` (receipt, invoice, bank_payment_confirmation, other or unreadable), and `ai.document_warning(value)` - `bank_payment_confirmation`, `other` or `unreadable`, else None; both readers send it as the answer's `document_warning`, a warning beside the suggestions (refused reads included), never a refusal. v0.20 (the user, 2026-11-24, "separate prompt"): the INVOICE prompt alone adds a line defining `bank_payment_confirmation` as a bank transfer confirmation or a card payment slip that lists no goods or services, so a card slip attached as an invoice is warned about; in the receipt prompt a card slip stays a receipt. v0.21 (the user, 2026-11-25): `account_id` is "pick the closest from the Chart of accounts." - never "" |
| `minty_shared.model_input` | extra `ai` | `prepare_for_model`: the small copy a model reads (image capped at 2048 px and saved once as JPEG, decoded at reduced scale when large; a PDF whole, every page - v0.16, the user 2026-11-24: totals and bank details are often on the last page; billed per page, held to 10 MB by the callers; v0.19: a HEIC/HEIF photo is decoded through `minty_shared.images` and sent as JPEG like any other photo, and `ai.sniff_mime` knows it). Never the stored file |
| `minty_shared.django.middleware` | extra `django` | `RequestLoggingMiddleware` (settings `MINTY_HTTP_LOGGER`, optional `MINTY_LOGGABLE_PATH`) |
| `minty_shared.django.errors` | extra `django` | the envelope below: `register_error_handlers`, `ApiRefusal`, `validation_message`, `field_messages` |
| `minty_shared.django.client` | extra `django` | `forward`/`proxy`: call another Minty backend as the current caller |
| `minty_shared.django.ai_allowance` | extra `django` (+ `minty_db` installed in the service) | (v0.7) the AI readers' allowance and settings, from minty-db's `ai_policy`/`ai_grant`/`ai_read`: each company reads against its own grants (`default_grant` given once on its first read; `give_grant` for top-ups), Petty Cash and Payment Request together. `load_policy` (fails closed: no usable row = reading off, logged), v0.11 the readers' settings from the same row - `Policy.reader(READER_RECEIPT | READER_INVOICE)` -> `ai.ReaderConfig` (`receipt_reader_on`/`invoice_reader_on`, `timeout_s`, `max_output_tokens`, `thinking_level`, `max_file_mb`; `parse_settings` names a value out of its column's range), `reader_status(reader, entity_id)` (what a page needs beside the read button: `enabled` = a usable policy AND that reader's switch, `percent_left`, `blocked`, `resumes_at`), `reserve` (under a per-company advisory lock; charges the oldest grant with room) and `finish` (only a read a model answered is charged to a grant; every attempt's tokens are recorded), `parse_rules`/`parse_grant`/`summarise` (extra caps over every company: reads, tokens, a rolling `period_days`) |
| `minty_shared.testing` | extra `testing` | the Postgres-harness pytest plugin (`pytest_plugins = ["minty_shared.testing.pg"]`), the source-walking guards, and (v0.8) `minty_shared.testing.sessions`: an autouse `hub_sessions` fixture - no test reaches accounts-api's session touch, every sign-in lives unless `hub_sessions.end(sid)`, `hub_sessions.calls` lists the sids asked about |

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
