# Security Policy

## Reporting a vulnerability

Please don't open a public issue for a security report. Email
manmeet.nain@gmail.com with a description of the issue and, if
possible, steps to reproduce it. This is an early-stage, single-maintainer
project — there's no formal SLA, but reports will be acknowledged and
looked at as soon as possible.

## Supported versions

Pre-1.0, no version branches — fixes land on `main` only. There's no
older release line receiving backports.

## Known design boundaries

Worth being upfront about, since they matter if you're deciding how to
deploy this:

- **The `X-Nashgate-Caller` header is an identity claim, not an
  authenticated credential.** `nashgate/gateway/callers.py` checks it
  against a fixed, configured roster, but doesn't verify the caller is
  who they say they are. Anyone who can reach the gateway and knows a
  configured caller name can route (and spend) as that caller. Put
  nashgate behind your own authentication layer — mTLS, an API
  gateway, a network boundary you already trust — rather than exposing
  it directly to untrusted clients.
- **API keys live in environment variables on the machine running the
  gateway** (`GatewayBackend.api_key()`, `nashgate/gateway/backends.py`),
  resolved per-request and never included in a response body, log
  line, or the `nashgate` response annotation. Standard secret-hygiene
  practices apply to how you set those env vars — nashgate doesn't add
  any additional protection around them.
- **No built-in rate limiting on the gateway's own HTTP surface.**
  Per-backend rate limits are what the router optimizes against
  (that's the whole point), but nothing stops a caller from firing
  requests at the gateway itself faster than intended — that's a
  concern for whatever sits in front of it (load balancer, API
  gateway, etc.), not something `nashgate/gateway/app.py` currently
  handles.

If you find a way any of these are worse than described here, or find
something not listed above, that's exactly what the reporting process
is for.
