#!/usr/bin/env python3
"""serve_api.py — the OpenAI-compatible front-end.

Owns everything text-shaped: the BPE tokenizer, Qwen's Jinja2 chat
template, the OpenAI request/response schema, and SSE framing. Talks to
the engine daemon (`halogen --serve`) over the newline-framed integer
protocol in src/serve.cpp — the engine only ever sees token ids.

This is `tools/`-shaped by design: it never touches a
kernel, and the engine stays framework-free and CLI-testable. A C++
tokenizer is the recorded future path, after the eval harness exists.

Endpoints: /v1/models, /v1/completions, /v1/chat/completions (both
support stream=true), /health.

Run (box, in the container — tools/run-serve.sh wraps both processes):
    python3 tools/serve_api.py --tokenizer <snapshot> --engine 127.0.0.1:8730
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_parse import (ToolStream, normalize_messages,      # noqa: E402
                        split_tool_calls)

SSE_PROF = {"t": 0.0, "n": 0}
MODEL_ID = "halogen-qwen3.8-27b"

# OpenAI's error envelope. FastAPI's native shape is {"detail": "..."}, which
# every OpenAI SDK misreads: they look for error.message and fall back to the
# raw body, so a perfectly clear 400 arrives at the user as an unparsed blob.
# Nothing in this repo consumed `detail`, so this replaces it rather than
# adding beside it.
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
               403: "permission_error", 404: "not_found_error",
               429: "rate_limit_error"}


def oai_error(status, message, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": ERROR_TYPES.get(status, "server_error"),
        "param": None, "code": code}})


# the timeouts that keep ONE bad request from wedging the server.
# At ONE slot the engine lock is the whole server: anything that can block
# while holding it can stop every other client indefinitely, which is what
# happened on 2026-08-24 (busy=true, GPU 0%, queued piling up behind a request
# that had already gone away). an earlier change makes that a per-request failure instead
# of an outage -- with N slots a wedged request costs one slot, and the
# batched path deliberately does NOT drop the shared socket on timeout,
# because other requests are riding it.
FIRST_TOKEN_S = 1800.0   # waits out PREFILL; a cold 262K prompt is ~19 min
NEXT_TOKEN_S = 300.0     # between tokens a round is sub-second; minutes = wedge
ABORT_DRAIN_S = 10.0     # resync is an optimization; reconnecting is correct


class EngineBusy(HTTPException):
    def __init__(self):
        super().__init__(status_code=503,
                         detail="timed out waiting for the engine — "
                                "halogen serves one request at a time "
                                "(batch-1) and the queue did not clear")


class Engine:
    """Client for the src/serve.cpp line protocol.

    Slice 1 is batch 1 by construction, so a single connection is held and
    serialized behind a lock — a second concurrent request gets 503 rather
    than silently interleaving into one KV state. Batching is handled by the scheduler.
    """

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.r = self.w = None
        # an earlier change: a SEMAPHORE, not a lock. Sized from the engine's INFO after
        # connect (kv_slots); 1 reproduces the batch-1 behaviour exactly,
        # including `.locked()` meaning "no capacity right now".
        self.slots = asyncio.Semaphore(1)
        self.n_slots = 1
        self.req = 0
        self.active = None   # batch-1 path only: the in-flight req id
        self.waiting = 0     # callers queued for a slot
        self.inflight = {}   # slot key -> monotonic start, for /health
        self.busy_since = None   # oldest live hold; None when idle
        self.streams = {}    # req id -> asyncio.Queue, the DEMUX
        self.wlock = asyncio.Lock()   # serialises WRITES onto the one socket
        # §53: serialises CONNECTS. `_ensure()` is called by every request, so
        # a socket that goes away with N callers in flight raced all N into
        # connect() at once. Measured before this lock: six concurrent
        # _ensure() calls opened SIX connections, each one leaking the previous
        # socket, spawning another _demux on the same stream (the one thing
        # that function's docstring says must never happen), and replacing both
        # the stream table and the slot semaphore underneath live holders.
        # tools/gate-reconnect.py is the falsifier.
        self.clock = asyncio.Lock()
        self.reader = None   # background demux task (batched engines only)
        self.cstat = asyncio.Queue()  # CSTAT replies, via the demux
        self.info = {}       # engine capabilities, from INFO at connect

    async def connect(self):
        async with self.clock:
            await self._connect_locked()

    async def _connect_locked(self):
        # Re-checked INSIDE the lock: callers queued behind the connect that
        # just succeeded must USE it rather than replace it. Without this the
        # lock would only serialise the damage instead of preventing it.
        if self.w is not None and not self.w.is_closing():
            return
        # Wake anything still waiting on the old socket BEFORE the table it is
        # registered in goes away, or it waits out its whole token budget and
        # reports the engine as silent. Then stop the old reader, so exactly
        # one coroutine ever owns the stream.
        for q in list(self.streams.values()):
            q.put_nowait(None)
        self.streams = {}
        if self.reader is not None and not self.reader.done():
            self.reader.cancel()
        self.reader = None
        old_w = self.w
        self.r, self.w = await asyncio.open_connection(self.host, self.port)
        if old_w is not None:
            try:
                old_w.close()          # it was REPLACED, not merely dropped
            except Exception:
                pass
        self.info = await self._probe()
        n = max(1, int(self.info.get("kv_slots", 1)))
        # The semaphore is rebuilt only when its SIZE changes. Replacing it on
        # every reconnect discards the count of slots currently held, so a
        # caller that later released into the new object would raise the
        # capacity above what the engine actually has.
        if n != self.n_slots:
            self.n_slots = n
            self.slots = asyncio.Semaphore(n)
        if self.n_slots > 1:
            # One reader owns the socket and fans lines out by req id. Nothing
            # else may read it: two coroutines reading one stream is how a
            # response ends up spliced into another request's tokens.
            self.reader = asyncio.create_task(self._demux(self.streams))

    async def _demux(self, streams=None):
        """Fan req-tagged lines into per-request queues.

        The protocol was built for this: every T/D/X line carries <req>. On
        socket close, every waiting stream is woken with a sentinel rather
        than left hanging -- an orphaned reader is precisely the wedge,
        one layer up.
        """
        try:
            while True:
                raw = await self.r.readline()
                if not raw:
                    break
                parts = raw.decode().split()
                if parts and parts[0] == "C":
                    self.cstat.put_nowait(parts)   # /cache, not a generation
                    continue
                if len(parts) < 2 or parts[0] not in ("T", "D"):
                    continue
                try:
                    rid = int(parts[1])
                except ValueError:
                    continue
                q = self.streams.get(rid)
                if q is not None:
                    q.put_nowait(parts)
        except Exception:
            pass
        finally:
            # §53: the table THIS reader served, captured at creation. Reading
            # self.streams here would wake whatever table happens to be
            # current, which after a reconnect is a different one -- leaving
            # the callers this reader was responsible for to time out instead.
            for q in list((streams if streams is not None
                           else self.streams).values()):
                q.put_nowait(None)          # sentinel: engine went away

    async def _probe(self):
        """Ask the engine what the loaded checkpoint can do.

        /health should report capabilities rather than guess them. An engine
        that predates INFO ignores the verb entirely, so this must never
        block forever — on timeout we fall back to 'serial only', which is
        the safe assumption.
        """
        try:
            self.w.write(b"INFO\n")
            await self.w.drain()
            raw = await asyncio.wait_for(self.r.readline(), timeout=5)
            p = raw.decode().split()
            if p and p[0] == "I" and len(p) >= 6:
                return {"mtp": p[1] == "1", "draft_head": p[2] == "1",
                        "ctx": int(p[3]), "spec_rows": int(p[4]),
                        "default": int(p[5]),
                        # trailing, m5″ s2: weights present in the checkpoint.
                        # NOT the same as selectable — see /health below.
                        "drafter_weights": len(p) >= 7 and p[6] == "1",
                        # trailing, m5″ s6: drafter 2 is actually SELECTABLE
                        # (weights present AND the verify apparatus wired).
                        "dflash2": len(p) >= 8 and p[7] == "1",
                        # trailing, an earlier change: prompt-cache cap (MB, 0 = off) and
                        # the snapshot alignment. Alignment is reported, not
                        # buried, because it is what decides whether a warm
                        # answer is bitwise identical to a cold one (an earlier change).
                        "cache_mb": int(p[8]) if len(p) >= 9 else 0,
                        "cache_align": int(p[9]) if len(p) >= 10 else 0,
                        # trailing, an earlier change: the engine's KV slot pool. The
                        # front-end sizes its own concurrency from this rather
                        # than guessing -- admitting more than there are slots
                        # just rebuilds the queue one layer up. Absent (older
                        # engine) means one slot, i.e. batch-1, which is the
                        # safe assumption for the same reason 'serial only' is.
                        "kv_slots": int(p[10]) if len(p) >= 11 else 1,
                        "slot_ctx": int(p[11]) if len(p) >= 12 else 0}
        except Exception:
            pass
        return {"mtp": False, "draft_head": False, "default": 0,
                "kv_slots": 1, "slot_ctx": 0}

    async def _ensure(self):
        if self.w is None or self.w.is_closing():
            await self.connect()

    async def close(self):
        try:
            if self.w is not None:
                self.w.close()
        except Exception:
            pass
        self.r = self.w = None
        self.active = None

    @staticmethod
    def _done_dict(parts):
        """Parse a D line's trailing stats. ONE copy, used by both paths --
        the field indices are exactly what drifts when a trailing field gets
        added to the serial reader and not the demux."""
        d = {"reason": parts[2], "n_prompt": int(parts[3]),
             "n_gen": int(parts[4]), "prefill_ms": float(parts[5]),
             "decode_ms": float(parts[6])}
        if len(parts) >= 10:            # an earlier change trailing stats
            d["drafter"] = int(parts[7])
            d["rounds"] = int(parts[8])
            d["commit"] = int(parts[9])
        if len(parts) >= 11:            # an earlier change prompt cache
            d["n_cached"] = int(parts[10])
        return d

    async def _gen_batched(self, req, line):
        """One of many concurrent generations over the shared socket.

        The socket is read ONLY by _demux(); this coroutine waits on its own
        queue. The two timeout budgets are unchanged and still apply per
        request, because a wedge is still a wedge -- what changed is that it
        now takes down ONE request instead of the server.
        """
        q = asyncio.Queue()
        self.streams[req] = q
        finished = False
        try:
            async with self.wlock:
                self.w.write(line.encode())
                await self.w.drain()
            first = True
            while True:
                try:
                    parts = await asyncio.wait_for(
                        q.get(), timeout=FIRST_TOKEN_S if first else NEXT_TOKEN_S)
                except asyncio.TimeoutError:
                    # Do NOT drop the shared socket: other requests are riding
                    # it. Fail this one. That is the whole point of batching --
                    # one bad request stops being everyone's problem.
                    raise HTTPException(
                        504, "engine went silent for %ds%s"
                             % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                                " while prefilling" if first else " mid-decode"))
                if parts is None:
                    raise HTTPException(502, "engine closed the connection")
                if parts[0] == "T":
                    first = False
                    yield (int(parts[2]), None,
                           float(parts[3]) if len(parts) > 3 else None)
                elif parts[0] == "D":
                    if parts[2] == "error":
                        raise HTTPException(400, "engine rejected the request "
                                                 "(prompt longer than a slot?)")
                    finished = True
                    yield None, self._done_dict(parts), None
                    return
        finally:
            self.streams.pop(req, None)
            if not finished:
                # THE GENERATOR KNOWS ITS OWN REQ, which is why cancellation
                # lives here rather than in Engine.abort(): with N concurrent
                # requests a single self.active cannot say which one went
                # away. Without the cancel the engine decodes to max_tokens
                # into a slot nobody is reading -- an earlier change's failure, except it
                # now costs one slot instead of the server.
                #
                # Fire-and-forget: the D line comes back to a stream that no
                # longer exists and _demux drops it, which is correct. We must
                # NOT wait for it, because that would be the unbounded drain
                # an earlier change removed.
                try:
                    self.w.write(f"X {req}\n".encode())
                except Exception:
                    pass

    async def generate(self, ids, max_tokens, eos, drafter=None, sample=None,
                       penalty=""):
        """Yields (token_id, None) per token, then (None, done_dict).

        `drafter` is the trailing wire field (0 serial / 1 MTP); None omits
        it and lets the engine apply its own default.

        `sample` is (temp, top_k, top_p, min_p, seed) or None. It goes on the
        wire as a KEYED suffix, not more positional fields, so an engine built
        before sampling landed still parses the line -- it simply never sees the
        keyword. None omits it entirely and the engine decodes greedy.
        """
        await self._ensure()
        self.req += 1
        req = self.req
        line = (f"GEN {req} {max_tokens} {len(eos)} "
                + (" ".join(str(e) for e in eos) + " " if eos else "")
                + f"{len(ids)} " + " ".join(str(i) for i in ids)
                + (f" {drafter}" if drafter is not None else "")
                # %.9g, not %g. %g carries 6 significant digits, which
                # renders top_p=0.9999999 as "1" -- and the engine reads
                # top_p >= 1 as NO NUCLEUS FILTER, so an unusually tight
                # request would silently become an unfiltered one. 9 digits
                # round-trips a float exactly, which is what the engine parses
                # these into. Exponential notation is fine: C++ operator>>
                # reads "1e-05" correctly.
                + (" SAMPLE %.9g %d %.9g %.9g %d" % sample if sample else "")
                + (penalty or "")
                + "\n")
        if self.n_slots > 1:
            async for item in self._gen_batched(req, line):
                yield item
            return
        self.w.write(line.encode())
        await self.w.drain()
        self.active = req
        # TIMEOUTS ON THE MAIN READ, which had none. A bare readline() here
        # means a silent engine wedges the request FOREVER while holding the
        # batch-1 lock, so one bad request takes the whole server down rather
        # than failing by itself.
        #
        # Two different budgets, because the two waits are nothing alike:
        #   FIRST token waits out PREFILL, which at 262K context is minutes
        #     (a cold 262K prompt is estimated ~19 min), so this has to be
        #     generous or long prompts break.
        #   LATER tokens arrive every round -- a few hundred ms even deep in
        #     context -- so a gap of minutes is not slowness, it is a wedge.
        # A single timeout cannot serve both: sized for prefill it never fires
        # on a hang, sized for decode it kills every long prompt.
        first = True
        while True:
            try:
                raw = await asyncio.wait_for(
                    self.r.readline(),
                    timeout=FIRST_TOKEN_S if first else NEXT_TOKEN_S)
            except asyncio.TimeoutError:
                # Drop the socket: the stream's position is now unknown, and
                # a half-read stream desynchronizes the NEXT request.
                await self.close()
                raise HTTPException(
                    504, "engine went silent for %ds%s — the connection was "
                         "dropped and the next request will reconnect"
                         % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                            " while prefilling" if first else " mid-decode"))
            if not raw:
                self.active = None
                raise HTTPException(502, "engine closed the connection")
            parts = raw.decode().split()
            if not parts:
                continue
            if parts[0] == "T" and int(parts[1]) == req:
                # THREE elements, not two. The second is reserved for the
                # terminal `done` dict and the loop below tests it with
                # `is not None`, so putting a per-token logprob there would
                # make every token look like the end of the stream.
                # parts[3] is the an earlier change logprob column, present only when
                # LOGPROBS was sent -- absent for every older engine and every
                # greedy request, hence a length check and not a version.
                first = False
                yield (int(parts[2]), None,
                       float(parts[3]) if len(parts) > 3 else None)
            elif parts[0] == "D" and int(parts[1]) == req:
                self.active = None
                if parts[2] == "error":
                    print(f"[ERROR serve_api] engine rejected request {req}: len(ids)={len(ids)}, max_tokens={max_tokens}", flush=True)
                    raise HTTPException(400, "engine rejected the request "
                                             "(prompt longer than context?)")
                yield None, self._done_dict(parts), None
                return

    async def cache_stats(self):
        """Live prompt-cache counters. Under the same lock as a
        generate: the engine reads one line at a time, so interleaving a
        CSTAT into an in-flight request would desynchronize the stream."""
        await self._ensure()
        if self.n_slots > 1:
            # NEVER read self.r here: _demux() owns it. Two coroutines reading
            # one stream is how a CSTAT reply ends up spliced into some
            # request's token sequence.
            async with self.wlock:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
            p = await asyncio.wait_for(self.cstat.get(), timeout=10)
        else:
            async with self.slots:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
                raw = await asyncio.wait_for(self.r.readline(), timeout=10)
            p = raw.decode().split()
        if not p or p[0] != "C" or len(p) < 10:
            return None
        d = {"entries": int(p[1]), "bytes": int(p[2]),
             "reserved_bytes": int(p[3]), "cap_bytes": int(p[4]),
             "hits": int(p[5]), "misses": int(p[6]), "stores": int(p[7]),
             "evicted": int(p[8]), "prompt_tokens_saved": int(p[9])}
        if len(p) >= 14:   # trailing: the cache's own copy cost, an earlier change
            d.update(store_ms_total=float(p[10]),
                     restore_ms_total=float(p[11]),
                     last_store_ms=float(p[12]),
                     last_restore_ms=float(p[13]))
        if len(p) >= 15:   # stores the machine was too short to take
            d["refused"] = int(p[14])
        return d

    async def abort(self):
        """Stop the in-flight request AND resynchronize the stream.

        Both halves matter. Without the cancel the engine keeps decoding to
        max_tokens after the client is gone — minutes of GPU held while
        every other request 503s. Without draining through the D line the
        leftover T lines desynchronize the NEXT request. On any failure the
        connection is dropped so the next call reconnects clean rather than
        inheriting a half-read stream.
        """
        if self.n_slots > 1:
            # an earlier change: per-request cancellation is done by the generator that
            # owns the req (see _gen_batched). A shared-socket abort here
            # would have to guess WHICH request vanished, and dropping the
            # socket would take down every other request riding it.
            return
        req = self.active
        if req is None:
            return
        try:
            if self.w is not None and not self.w.is_closing():
                self.w.write(f"X {req}\n".encode())
                await self.w.drain()
            # BOUNDED AS A WHOLE, not per read. The first version used
            # `while True` with a 180 s timeout on each readline, which bounds
            # nothing: every line that arrives resets the clock, so a stream
            # that keeps talking keeps this loop alive indefinitely. And this
            # runs BEFORE the lock is released, so an unbounded drain is an
            # unbounded lock hold -- observed 2026-08-24 as busy=true with the
            # GPU at 0% and every later request queued behind it.
            #
            # Resynchronizing is an OPTIMIZATION (it saves a reconnect);
            # dropping the socket is always correct, because _ensure()
            # reconnects and the engine's accept loop takes a fresh client. So
            # give it a short deadline and fall back to the correct thing.
            deadline = time.monotonic() + ABORT_DRAIN_S
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                raw = await asyncio.wait_for(self.r.readline(), timeout=left)
                if not raw:
                    break
                p = raw.decode().split()
                if p and p[0] == "D" and int(p[1]) == req:
                    self.active = None
                    return
        except Exception:
            pass
        await self.close()


class CompletionReq(BaseModel):
    model: str = MODEL_ID
    prompt: str = ""
    max_tokens: int = 128
    stream: bool = False
    stop: list[str] | str | None = None
    # an earlier change: {"include_usage": true} appends a final chunk carrying usage,
    # per OpenAI. Without it a streaming client has no token accounting at
    # all — the non-streamed body is the only place usage appears.
    stream_options: dict | None = None
    # an earlier change: which drafter decodes this request (serial | mtp | dflash2).
    # Output is IDENTICAL whichever is picked — spec commit is trunk-argmax
    # equality — so this only changes speed, which is what makes it a clean
    # live A/B.
    drafter: str | None = None
    # THE SAMPLING FIELDS, AND WHY THEY WERE MISSING. §53.
    #
    # This endpoint returned a 500 on EVERY request, in every image from §37
    # s1 (8f1e71b, 2026-08-23) through the shipped 0.1.2. `completions()`
    # calls `check_sampling(req)`, `sample_spec(req)` and `penalty_spec(req)`,
    # all of which read `req.temperature`; when sampling shipped the fields
    # were added to ChatReq and this model was not touched, so the very first
    # attribute access raised AttributeError and the caller got an opaque
    # "internal error" on a route the public README advertises.
    #
    # It survived because nothing exercised it: every gate we had drives
    # /v1/chat/completions, and nothing anywhere listed which routes were
    # SUPPOSED to work. tools/gate-endpoints.py is the durable half of this
    # fix -- /health now advertises the route table and a route with no probe
    # there fails rather than passing silently.
    #
    # Declared to match ChatReq field for field, because the code behind this
    # endpoint has always been written as though they were here.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: int | bool | None = None
    top_logprobs: int | None = None
    n: int | None = None


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    # This model reasons before it answers and those tokens count against the
    # budget, so a budget too small does not shorten the answer, it removes it:
    # generation stops inside the thinking block, `parts()` finds no closing
    # marker, and the caller gets content "" with the whole reply in
    # reasoning_content. 8192 clears an ordinary request with headroom while
    # still bounding how long one request can hold a slot. Callers who need
    # more ask for more; the ceiling is separate policy (--max-tokens-cap).
    max_tokens: int = 8192
    # OpenAI deprecated `max_tokens` for Chat Completions when reasoning models
    # shipped: generated tokens began including reasoning the caller never sees,
    # so the bound needed a name for what it actually bounds.
    # `max_completion_tokens` is the current Chat Completions field and
    # `max_output_tokens` is the Responses API's. All three mean one thing here,
    # an upper bound on reasoning + content, so all three are declared and
    # `_resolve_budget` folds them into max_tokens for everything downstream.
    # Declared rather than left to pydantic's `ignore`, so that a caller using
    # the current spelling is honored instead of quietly getting the default.
    max_completion_tokens: int | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    stream_options: dict | None = None
    drafter: str | None = None
    # Qwen3.8 chat-template controls (an earlier change — template kwargs, no engine work).
    # The template accepts low | medium | xhigh and raises on anything else,
    # so OpenAI's vocabulary is mapped onto it in chat_kwargs().
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    tools: list[dict] | None = None
    # an earlier change. 'auto' (default) | 'none' | 'required' |
    # {"type": "function", "function": {"name": ...}}.
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def _resolve_budget(self):
        """Fold the three spellings of the token budget into `max_tokens`.

        Only names the caller actually SENT are considered, so the default
        never competes with an explicit value. Agreeing duplicates are fine,
        since sending `max_tokens` and `max_completion_tokens` together is how
        a client stays compatible with servers that read only one. Disagreeing
        values are a 400 rather than a guess: the caller asked for two
        different things, and silently picking one would misreport what the
        server did.
        """
        sent = {}
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if name in self.model_fields_set:
                v = getattr(self, name)
                if v is not None:
                    sent[name] = v
        distinct = set(sent.values())
        if len(distinct) > 1:
            raise ValueError(
                "conflicting token budgets ("
                + ", ".join(f"{k}={v}" for k, v in sorted(sent.items()))
                + "); they are the same budget under three names, send one")
        if distinct:
            self.max_tokens = distinct.pop()
        return self
    # an earlier change: IMPLEMENTED. temperature 0 (the default) is greedy and keeps
    # the speculative fast path; anything above 0 samples and is routed to
    # SERIAL decode, because the spec commit rule is argmax equality and
    # sampled tokens would both collapse acceptance and emit the wrong
    # distribution (an earlier change trap 1). an earlier change lifts that.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # vLLM-compatible extensions; not in the OpenAI schema but widely sent.
    top_k: int | None = None
    min_p: float | None = None
    # an earlier change: IMPLEMENTED, sampler-only (they need temperature > 0).
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # Still accepted and NOT implemented. Silently dropping this would
    # misrepresent the output, so /health reports it and the server logs.
    n: int | None = None


# OpenAI effort vocabulary -> what this template actually accepts.
EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium",
              "high": "xhigh", "xhigh": "xhigh"}
# What remains accepted-but-ignored. temperature/top_p/seed/top_k/min_p came
# off this list at an earlier change -- the point of the list is that it shrinks.
SAMPLING_FIELDS = ("n",)



def penalty_spec(req):
    """The PENALTY / BIAS / LOGPROBS suffixes, or "" for none.

    All three are sampler-only and the engine REJECTS them without
    temperature>0 rather than ignoring them: greedy takes the argmax, so an
    additive delta that does not reorder it changes nothing a client could
    see, and a logprob has no sampler to come from. Building them here only
    when sampling keeps that rejection unreachable from a well-formed request.
    """
    if sample_spec(req) is None:
        return ""
    out = ""
    pp = float(req.presence_penalty or 0.0)
    fp = float(req.frequency_penalty or 0.0)
    if pp or fp:
        out += " PENALTY %.9g %.9g" % (pp, fp)
    if req.logit_bias:
        items = []
        for k, v in req.logit_bias.items():
            try:
                tid = int(k)
            except (TypeError, ValueError):
                raise HTTPException(400, f"logit_bias key {k!r} is not a "
                                         f"token id")
            if not 0 <= tid < 248320:
                raise HTTPException(400, f"logit_bias token id {tid} is "
                                         f"outside 0..248319")
            items.append((tid, float(v)))
        if len(items) > 20480:
            raise HTTPException(400, f"logit_bias has {len(items)} entries, "
                                     f"over this server's cap of 20480")
        out += " BIAS %d " % len(items) + " ".join(
            "%d %.9g" % (i, v) for i, v in items)
    if req.logprobs:
        out += " LOGPROBS"
    return out


def sample_spec(req):
    """(temp, top_k, top_p, min_p, seed) or None for greedy.

    None is returned for temperature 0 or absent, and that is not a shortcut:
    temperature 0 IS greedy, and routing it through the sampler would put a
    new kernel under the identity hashes that certify the greedy path.
    """
    t = req.temperature
    if t is None or t <= 0.0:
        return None
    # A seed is REQUIRED by the engine (the RNG is counter-based and stateless),
    # but optional in the API. Absent, draw one: the request is then genuinely
    # unreproducible, which is the honest reading of "no seed given" -- rather
    # than a fixed default that would make every unseeded request identical.
    seed = req.seed if req.seed is not None else random.getrandbits(63)
    return (float(t), int(req.top_k or 0), float(req.top_p or 0.0),
            float(req.min_p or 0.0), int(seed))
_warned_sampling = set()
# Wire values of src/serve.cpp's GEN drafter field (an earlier change, an earlier change).
DRAFTERS = {"serial": 0, "mtp": 1, "dflash2": 2, "ngram": 3}
# Drafters that cannot serve temperature > 0, and why. An n-gram proposal is a
# LOOKUP, not a draw, so there is no q for the accept test min(1, p/q) to
# divide by; the engine refuses it (src/serve.cpp) and NgramDrafter's sampled
# entry points exit loudly if that refusal is ever bypassed.
#
# This map exists because the engine's refusal is NOT enough on its own. A
# `D ... error` reaches an SSE client as 200 + headers + zero bytes — the exact
# signature of a dead engine (§37, §38) — so a rule the engine enforces must
# also be stated HERE, where it can still become a 400 with a reason on it.
GREEDY_ONLY_DRAFTERS = {
    "ngram": "an n-gram proposal is a lookup, not a draw, so it carries no "
             "distribution for the speculative accept test to divide by",
}
# Which INFO capability makes each one selectable. serial always is.
# ngram maps to "mtp" deliberately: it carries no weights of its own, so what
# decides whether it is selectable is the VERIFY apparatus (the kSpecRows-wide
# DN slot ring and forward_spec), which is sized off has_mtp. Without this it
# would be advertised in drafters_available on a checkpoint that refuses it.
DRAFTER_CAP = {"mtp": "mtp", "dflash2": "dflash2", "ngram": "mtp"}


# an earlier change. The forced-call opener. This template has no tool_choice support of
# its own, so `required` and a named function are implemented as a PROMPT
# PREFILL: the opener is appended after the generation prompt and the model
# resumes inside a call it cannot decline. The cost is stated in /health and
# the prefill occupies the position the reasoning block would
# have used, so a forced call does not think.
FORCE_OPEN = "<tool_call>\n<function="


def tool_choice_mode(req):
    """-> (mode, forced_name); mode is auto | none | required | function."""
    tc = req.tool_choice
    if tc is None or tc == "auto":
        return "auto", None
    if isinstance(tc, str):
        if tc == "none":
            return "none", None
        if tc == "required":
            if not req.tools:
                raise HTTPException(400, "tool_choice 'required' needs tools")
            return "required", None
        raise HTTPException(400, f"tool_choice '{tc}' unknown; use "
                                 f"auto | none | required | "
                                 f"{{'type': 'function', ...}}")
    if isinstance(tc, dict) and tc.get("type") == "function":
        name = (tc.get("function") or {}).get("name")
        known = {(t.get("function") or t).get("name") for t in req.tools or []}
        if not isinstance(name, str):
            raise HTTPException(400, "tool_choice.function.name is required")
        if name not in known:
            raise HTTPException(400, f"tool_choice names '{name}', which is "
                                     f"not in tools")
        return "function", name
    raise HTTPException(400, "tool_choice must be a string or a "
                             "{'type': 'function', ...} object")


def force_prefill(mode, name, tools):
    """The text appended after the generation prompt to force a call.

    MEASURED 2026-08-22, and it is the difference between a guarantee and a
    hope: prefilling only `<tool_call>\n<function=` still lets the model
    decline, by emitting the closing tags with an EMPTY name —

        <tool_call>\n<function=\n</function>\n</tool_call>

    which is what it did when asked "Hi, how are you?" with tool_choice
    'required'. Naming the function in the prefill removes that escape
    entirely (same prompt, same effort: it called get_weather for New York).
    So when there is exactly one tool there is no choice to make and the name
    goes in — 'required' is then exact. With several tools the model must
    pick, the escape is open again, and serve() turns a declined forced call
    into a 502 rather than handing back raw markup as content.
    """
    if mode == "function":
        return FORCE_OPEN + name + ">\n"
    if mode == "required":
        names = [(t.get("function") or t).get("name") for t in tools or []]
        if len(names) == 1 and isinstance(names[0], str):
            return FORCE_OPEN + names[0] + ">\n"
        return FORCE_OPEN
    return ""


def wants_usage(req):
    so = req.stream_options
    return bool(isinstance(so, dict) and so.get("include_usage"))


def chat_kwargs(req, mode="auto"):
    """Template kwargs from an OpenAI-shaped request. Only pass what the
    caller actually set — the template's own defaults (thinking on, effort
    xhigh) are the model's intended behavior."""
    kw = {}
    if req.reasoning_effort is not None:
        e = EFFORT_MAP.get(req.reasoning_effort.lower())
        if e is None:
            raise HTTPException(400, f"reasoning_effort "
                                     f"'{req.reasoning_effort}' unsupported; "
                                     f"use {'|'.join(EFFORT_MAP)}")
        kw["reasoning_effort"] = e
    if req.enable_thinking is not None:
        kw["enable_thinking"] = req.enable_thinking
    if req.preserve_thinking is not None:
        kw["preserve_thinking"] = req.preserve_thinking
    # tool_choice='none' means the model must not call one, and the cleanest
    # guarantee of that is a prompt that never mentions tools. Note the
    # consequence: it is a DIFFERENT prefix from an 'auto' turn, so alternating
    # the two in one session costs a prompt-cache miss each time (an earlier change).
    if req.tools and mode != "none":
        kw["tools"] = req.tools
    return kw


class ThinkSplit:
    """Splits generated text into reasoning_content / content at the first
    '</think>'.

    The template puts the OPENING '<think>' into the prompt itself, so the
    model emits reasoning first and only ever produces the CLOSING tag —
    everything before it is reasoning, everything after is the answer. With
    enable_thinking=false the prompt is pre-closed, no marker appears, and
    all output is content. Tracking full text (rather than per-delta) keeps
    a marker split across two deltas correct.
    """

    MARK = "</think>"

    def __init__(self, thinking=True):
        # thinking=False: the template already emitted '<think>\n\n</think>'
        # into the PROMPT, so no marker appears in the generated text and
        # every byte is answer. Without this flag a no-marker stream is
        # ambiguous and the whole reply misfiles as reasoning.
        self.thinking = thinking
        self.full = ""
        self.sent_r = self.sent_c = 0

    def push(self, delta):
        """-> (reasoning_delta, content_delta)"""
        self.full += delta
        if not self.thinking:
            out = self.full[self.sent_c:]
            self.sent_c = len(self.full)
            return "", out
        i = self.full.find(self.MARK)
        if i < 0:
            # marker may be half-arrived; hold back a possible prefix
            hold = 0
            for k in range(len(self.MARK) - 1, 0, -1):
                if self.full.endswith(self.MARK[:k]):
                    hold = k
                    break
            r = self.full[:len(self.full) - hold]
            out = r[self.sent_r:]
            self.sent_r = len(r)
            return out, ""
        r, c = self.full[:i], self.full[i + len(self.MARK):]
        rd, cd = r[self.sent_r:], c[self.sent_c:]
        self.sent_r, self.sent_c = len(r), len(c)
        return rd, cd

    def parts(self):
        if not self.thinking:
            return "", self.full
        i = self.full.find(self.MARK)
        if i < 0:
            # thinking on but never closed (hit max_tokens mid-reasoning)
            return self.full.strip(), ""
        return (self.full[:i].strip(),
                self.full[i + len(self.MARK):].lstrip("\n"))


def build_app(tok, engine, ctx, max_cap=4096, queue_timeout=600):
    app = FastAPI(title="halogen")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        # 503 here is only ever the batch-1 queue timing out, and saying so in
        # `code` lets a client back off on that specifically rather than on
        # every 5xx.
        return oai_error(exc.status_code, str(exc.detail),
                         "engine_busy" if exc.status_code == 503 else None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        # Reported as 400, not FastAPI's 422: OpenAI uses 400 for a malformed
        # request and SDK retry logic is written against that.
        parts = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e.get("loc", ())
                           if x != "body") or "body"
            parts.append(f"{loc}: {e.get('msg', 'invalid')}")
        return oai_error(400, "; ".join(parts) or "invalid request")
    eos_ids = [i for i in {tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|im_end|>")}
               if isinstance(i, int) and i >= 0]

    def stop_list(stop):
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else list(stop)

    def check_sampling(req):
        """Reject what cannot be honoured, rather than accepting and dropping.

        the same lesson in a different place: a request that is silently not
        served the way it was asked is indistinguishable, from the response,
        from one that was.
        """
        # RANGE VALIDATION, and it is the an earlier change rule again rather than
        # schema pedantry. The engine reads top_p >= 1 as "no nucleus filter",
        # so top_p=5 does not fail — it silently serves an UNFILTERED nucleus
        # to a client that asked for a tight one, with a 200 and output that
        # looks entirely reasonable. That is the same defect an earlier change found from
        # the other direction, where "%g" rendered 0.9999999 as "1". A value
        # outside the range a parameter is DEFINED on is a client error and
        # must say so.
        for f, lo, hi in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0),
                          ("min_p", 0.0, 1.0)):
            v = getattr(req, f)
            if v is not None and not (lo <= float(v) <= hi):
                raise HTTPException(
                    400, f"{f}={v} is outside {lo}..{hi}. Values past the "
                         f"range are NOT clamped — a clamped request is "
                         f"indistinguishable, from the response, from one "
                         f"that was served as asked.")
        if req.top_k is not None and req.top_k < 0:
            raise HTTPException(400, f"top_k={req.top_k} must be >= 0 "
                                     f"(0 disables the filter).")
        if req.logprobs and req.stream:
            raise HTTPException(
                400, "logprobs with stream=true is not implemented — the "
                     "logprob column is collected and returned with the "
                     "finished response. Send stream=false, or drop "
                     "logprobs.")
        if req.top_logprobs:
            raise HTTPException(
                400, "top_logprobs is not implemented — this server returns "
                     "the logprob of the CHOSEN token only. Send "
                     "logprobs=true without top_logprobs.")
        if sample_spec(req) is not None and req.drafter:
            why = GREEDY_ONLY_DRAFTERS.get(req.drafter.lower())
            if why:
                raise HTTPException(
                    400, f"drafter '{req.drafter.lower()}' requires "
                         f"temperature 0 (greedy): {why}. Send temperature=0, "
                         f"or pick another drafter.")
        if sample_spec(req) is None:
            for f, why in (("presence_penalty", req.presence_penalty),
                           ("frequency_penalty", req.frequency_penalty),
                           ("logit_bias", req.logit_bias),
                           ("logprobs", req.logprobs)):
                if why:
                    raise HTTPException(
                        400, f"{f} requires temperature > 0 — it applies to "
                             f"the sampler, and temperature 0 is greedy "
                             f"argmax decoding.")

    def drafter_for(req):
        """Wire drafter value.

        A later change REMOVED the serial-only restriction on sampling. Until it
        landed, temperature>0 forced serial and cost ~1/2.2 of the throughput;
        speculative sampling now serves it on the same fast path, because the
        accept rule changed with it -- min(1, p/q) with a residual correction,
        which emits exactly p rather than requiring argmax equality.

        So sampling no longer constrains the drafter and this is a plain
        lookup again. What it must NOT become is a silent substitution:
        s7's rule is that a quiet downgrade makes any drafter comparison lie,
        which is why the restriction was a 400 rather than a fallback while it
        existed.
        """
        return drafter_id(req.drafter)

    def drafter_id(name):
        """Request drafter name -> wire value.

        Unknown or unavailable is a 400, never a silent downgrade: a request
        that quietly fell back to serial would make an mtp-vs-serial or an
        mtp-vs-dflash2 comparison lie about what actually ran. The benchmark varies
        exactly this field and nothing else, so the field has to be honest.
        """
        if name is None:
            return None                 # engine applies HALOGEN_DRAFTER
        key = name.lower()
        v = DRAFTERS.get(key)
        if v is None:
            raise HTTPException(400, f"drafter '{name}' unknown; use "
                                     f"{' | '.join(DRAFTERS)}")
        cap = DRAFTER_CAP.get(key)
        if cap is not None and not engine.info.get(cap):
            raise HTTPException(400, f"drafter '{key}' unavailable: the "
                                     f"loaded checkpoint cannot serve it")
        return v

    async def run(ids, max_tokens, stops, drafter=None, sample=None,
                  penalty=""):
        """Drives the engine and incrementally detokenizes.

        Detokenization is done by decoding the whole token list each step
        and diffing against what was already emitted — the only way to get
        BPE/multi-byte boundaries right, since a token's text can depend on
        its successor.
        """
        # an earlier change: ask the ENGINE, don't trust a CLI default. --context and
        # Model::kMaxCtx are two copies of one number and they drifted the
        # moment kMaxCtx moved: the front-end would have kept rejecting at
        # 32,768 against an engine serving 131,072, and the symptom is a 400
        # that looks like a client bug. INFO already reports it.
        limit = engine.info.get("ctx") or ctx
        if len(ids) >= limit:
            raise HTTPException(400, f"prompt {len(ids)} tokens exceeds "
                                     f"context {limit}")
        out, text, done = [], "", None
        lps = []
        # an earlier change step 9. an earlier change step 8 measured 32.5 ms/step of front-end cost
        # at c=8 and named a SUSPECT (O(n^2) detok) without measuring it.
        # METHOD 45: measure before fixing. These three sum to the wall of
        # this loop, so they apportion it rather than sampling it.
        prof = {"detok": 0.0, "wait": 0.0, "n": 0}
        _t_last = time.perf_counter()
        async for tid, d, lp in engine.generate(ids, max_tokens, eos_ids,
                                                drafter, sample, penalty):
            if lp is not None:
                lps.append(lp)
            if d is not None:
                done = d
                # The serving ledger an earlier change asked for: what real traffic
                # actually commits per round, not what a fixture does.
                # an earlier change: this WAS guarded by `if d.get("rounds")`, i.e. it
                # only logged when a DRAFTER ran. The batched scheduler emits a
                # plain 6-field D line, so batched requests logged NOTHING --
                # and tools/bench-serving.py, the benchmark of record, parses
                # exactly this line, so the whole batched path was invisible to
                # it. The throughput half is now unconditional and keeps that
                # regex's shape; the spec detail stays conditional because only
                # a drafter has rounds to report.
                # A 1-token response has NO decode rate to report: its only
                # token is the one prefill already produced, so decode_ms
                # rounds to 0. This used to floor dt at 1e-9 and print
                # "1000000000.00 t/s" -- a fabricated number sitting in the
                # exact field a real rate goes. `sweep`'s prefill probes
                # (max_tokens=1) hit it on every single call, so the first
                # thing a user benchmarking this image saw in the log was a
                # 1e9 t/s line.
                dt = d["decode_ms"] / 1000
                name = {1: "mtp", 2: "dflash2"}.get(d.get("drafter"),
                                                    "spec" if d.get("rounds")
                                                    else "batch")
                spec = ""
                if d.get("rounds"):
                    spec = (f"{d['rounds']} rounds, "
                            f"commit {d['commit'] / d['rounds']:.2f}/round | ")
                # LEDGER in tools/bench-serving.py requires `([\d.]+) t/s`
                # AND a `rounds, commit` clause. These degenerate lines never
                # carry rounds, so "n/a" cannot break that parser -- but the
                # rate keeps its numeric shape in every case that does.
                rate = (f"{d['n_gen'] / dt:.2f} t/s"
                        if dt > 0 and d["n_gen"] > 1 else "n/a")
                pref_s = d['prefill_ms'] / 1000
                pref_rate = f"{d['n_prompt'] / pref_s:.1f} t/s" if pref_s > 0 else "n/a"
                itl = f"{d['decode_ms'] / max(d['n_gen'], 1):.1f}ms/tok" if d['n_gen'] > 0 else "n/a"
                print(f"serve_api: {name} {d['n_gen']} tok in {dt:.2f}s = "
                      f"{rate} | {spec}"
                      f"prompt {d['n_prompt']}"
                      f"{f' ({c} cached)' if (c := d.get('n_cached')) else ''}"
                      f", prefill {pref_s:.2f}s ({pref_rate}) | itl {itl}"
                      # detok us/token is a CANARY, not a stat: this loop
                      # decodes the WHOLE token list every step, so its cost
                      # is O(n^2) and this figure climbs with output length
                      # -- 17 us/tok at 128 tokens, a projected 2000 us/tok at
                      # the 16,384 cap (~6% of the request). It is the only
                      # thing that would show that, and the measurement that
                      # "refuted" the O(n^2) suspect was taken at n=128, where
                      # O(n^2) is invisible by construction. `wait` and `sse`
                      # were dropped: wait duplicates the t/s field and
                      # changes meaning with concurrency, sse is a constant.
                      f" | detok {prof['detok'] / max(prof['n'], 1) * 1e6:.0f}us/tok",
                      flush=True)
                break
            # time spent WAITING on the engine == everything since we last
            # finished a token; it is the term the other two are competing
            # against and it must be in the ledger or the shares are wrong.
            _t0 = time.perf_counter()
            prof["wait"] += _t0 - _t_last
            out.append(tid)
            new = tok.decode(out, skip_special_tokens=True)
            prof["detok"] += time.perf_counter() - _t0
            prof["n"] += 1
            # HOLD BACK an incomplete multi-byte character. A byte-fallback
            # token can carry the first byte(s) of a UTF-8 sequence, and
            # tok.decode renders that as U+FFFD until the next token completes
            # it. Emitting the U+FFFD means the NEXT decode no longer starts
            # with what we already sent.
            #
            # This was a live bug, found by hitting the endpoint with a real
            # prompt: the old code answered a mismatch by resetting text to ""
            # — which makes the next delta the ENTIRE accumulated string, so
            # the client receives the whole reasoning block a second time,
            # spliced into the middle of the answer. `5 ÷ 2 = 2.5` came back as
            # 167 characters instead of 12. Its comment called the case a
            # "rare retokenization shuffle"; it is neither rare nor a shuffle,
            # it fires on any character the tokenizer byte-splits (÷ does;
            # café, 😊 and 日本語 are single tokens here and do not).
            while new.endswith("�"):
                new = new[:-1]
            if not new.startswith(text):
                # Genuine retokenization: emit only what actually differs
                # rather than re-sending everything. Never reset to "".
                n = 0
                while n < len(text) and n < len(new) and text[n] == new[n]:
                    n += 1
                delta, text = new[n:], new
            else:
                delta, text = new[len(text):], new
            hit = next((s for s in stops if s and s in text), None)
            if hit:
                cut = text.index(hit)
                tail = delta[:max(0, cut - (len(text) - len(delta)))]
                if tail:
                    yield tail, None
                await engine.abort()   # drains through D — never bare cancel
                yield None, {"reason": "stop", "n_gen": len(out),
                             "n_prompt": len(ids), "prefill_ms": 0.0,
                             "decode_ms": 0.0, "logprobs": lps}
                return
            if delta:
                _t_last = time.perf_counter()
            yield delta, None
        if done is not None:
            done["logprobs"] = lps
        yield None, done or {"reason": "length", "n_gen": len(out),
                             "n_prompt": len(ids), "prefill_ms": 0.0,
                             "decode_ms": 0.0, "logprobs": lps}

    def usage(d, n_prompt):
        u = {"prompt_tokens": n_prompt, "completion_tokens": d["n_gen"],
             "total_tokens": n_prompt + d["n_gen"]}
        # an earlier change. OpenAI's own field name for this, so a client that already
        # tracks cache hits reads it without a halogen-specific branch. It
        # counts prompt tokens the engine did NOT re-prefill; they are still
        # billed as prompt_tokens because they are still context.
        if d.get("n_cached"):
            u["prompt_tokens_details"] = {"cached_tokens": d["n_cached"]}
        return u

    async def sse(gen, cid, created, chat, split, tstream=None, pre="",
                  parallel=True, include_usage=False):
        def envelope():
            return {"id": cid, "created": created, "model": MODEL_ID,
                    "object": "chat.completion.chunk" if chat
                              else "text_completion"}

        def frame(payload, fin=None):
            f = envelope()
            f["choices"] = [{"index": 0, "finish_reason": fin, **payload}]
            # OpenAI: with include_usage every OTHER chunk carries an explicit
            # null usage, and only the extra final chunk carries the numbers.
            if include_usage:
                f["usage"] = None
            return f

        buf = ""
        # STREAMING MUST RETURN WHAT NON-STREAMING RETURNS. §53.
        #
        # The non-streaming arm finishes through `ThinkSplit.parts()`, which
        # does `.strip()` on the reasoning and `.lstrip("\n")` on the content.
        # This arm emitted the raw deltas. The template closes its thinking
        # block with "</think>\n\n", so EVERY streamed answer carried a
        # two-newline prefix the non-streamed one did not, and the reasoning
        # differed by leading and trailing whitespace. The generated TOKENS are
        # identical either way -- this was never the model, only two
        # serializers disagreeing about tidying.
        #
        # It mattered beyond cosmetics in one case that changes client logic:
        # on a tool-only turn the non-streamed content is "" and the streamed
        # content was "\n\n", so `if content:` answered differently for the
        # same turn depending only on how it was asked.
        #
        # Content: hold back leading newlines until the first real character.
        # Reasoning: hold back leading whitespace the same way, AND hold back a
        # whitespace-only tail until something follows it -- in a stream a
        # trailing newline is only trailing once nothing else comes.
        # tools/gate-stream-parity.py is the falsifier, red on both arms of
        # five shapes before this.
        c_started = False
        rd_started = False
        rd_pending = ""
        # an earlier change step 9: front-end CPU per token, excluding the await.
        # PROCESS-WIDE and never reset: 8 concurrent requests share this
        # module-level dict, so a per-request reset would have each stream
        # clobbering the others and reporting noise. The quantity wanted is
        # us/token of front-end CPU, which is concurrency-independent, so a
        # running mean over the process is the right shape.
        async for delta, d in gen:
            _s0 = time.perf_counter()
            if d is not None:
                fin = {"stop": "stop", "cancel": "stop",
                       "length": "length"}.get(d["reason"], "stop")
                # A run cut off by max_tokens reports 'length' even if a
                # complete call came first — the client must see that the
                # turn was truncated, not that it ended on a tool call.
                if tstream is not None and tstream.any_calls() \
                        and fin != "length":
                    fin = "tool_calls"
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {}} if chat else {'text': ''}, fin))}"
                       f"\n\n")
                if include_usage:
                    # Per OpenAI: choices is ALWAYS empty on this chunk, and
                    # it is the last thing before [DONE].
                    u = envelope()
                    u["choices"] = []
                    u["usage"] = usage(d, d.get("n_prompt", 0))
                    yield f"data: {json.dumps(u)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not chat:
                _ev = f"data: {json.dumps(frame({'text': delta}))}\n\n"
                SSE_PROF["t"] += time.perf_counter() - _s0
                SSE_PROF["n"] += 1
                yield _ev
                continue
            # chat: reasoning goes to its own field until '</think>'
            rd, cd = split.push(delta)
            SSE_PROF["n"] += 1
            SSE_PROF["t"] += time.perf_counter() - _s0
            if rd:
                if not rd_started:
                    rd = rd.lstrip()
                    if rd:
                        rd_started = True
                if rd_started and rd:
                    rd = rd_pending + rd
                    kept = rd.rstrip()
                    rd_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    yield (f"data: "
                           f"{json.dumps(frame({'delta': {'reasoning_content': rd}}))}"
                           f"\n\n")
            if not cd:
                continue
            if not c_started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                c_started = True
            if tstream is None:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': cd}}))}\n\n")
                continue
            # an earlier change: content and tool-call markup share one token stream, so
            # they are separated HERE and never both emitted. Feeding the full
            # accumulated text (not the delta) is what makes a marker split
            # across two tokens safe — same reason ThinkSplit does it.
            buf += cd
            td, evs = tstream.push(pre + buf)
            if td:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': td}}))}\n\n")
            for e in evs:
                if not parallel and e["index"] > 0:
                    continue
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'tool_calls': [e]}}))}"
                       f"\n\n")

    @app.get("/health")
    async def health():
        rev = {v: k for k, v in DRAFTERS.items()}
        return {"status": "ok", "model": MODEL_ID,
                # THE ROUTE TABLE THIS BUILD ACTUALLY SERVES. §53.
                #
                # Generated from `app.routes`, never hand-written, so it
                # cannot name a route that is not there or omit one that is.
                # It exists so tools/gate-endpoints.py can probe every
                # advertised route and FAIL on one it has no probe for --
                # which is what would have caught /v1/completions answering
                # 500 for five months while the README listed it.
                "endpoints": sorted(
                    r.path for r in app.routes
                    if getattr(r, "path", "").startswith("/v1/")),
                "context": engine.info.get("ctx") or ctx,
                "busy": engine.slots.locked(),
                "slots": engine.n_slots,
                "slot_ctx": engine.info.get("slot_ctx") or 0,
                "in_flight": len(engine.inflight),
                # Seconds the current request has held the batch-1 slot. A
                # number beats a boolean here: a wedge and a long generation
                # are both "busy", and only the duration separates them
                # without going to the GPU counter.
                "busy_for_s": (round(time.monotonic() - engine.busy_since, 1)
                               if engine.busy_since and engine.inflight
                               else 0),
                "queued": engine.waiting,
                "decode": "greedy",
                # an earlier change: what the LOADED checkpoint carries, straight from
                # the engine's INFO — not a guess, and not config.
                "drafters_available": [
                    n for n in DRAFTERS
                    if n not in DRAFTER_CAP
                    or engine.info.get(DRAFTER_CAP[n])],
                # Which of the above refuse temperature > 0. Reported rather
                # than left for a client to discover as a 400: `decode` above
                # already advertises sampling, and a drafter that silently
                # excludes it is exactly the kind of gap a health check exists
                # to close.
                "drafters_greedy_only": [
                    n for n in GREEDY_ONLY_DRAFTERS if n in DRAFTERS],
                "drafter_default": rev.get(engine.info.get("default", 0),
                                           "serial"),
                # False here means every draft step reads the full 248,320-row
                # lm_head instead of the 98,304 shortlist (an earlier change.f).
                "shortlist_draft_head": bool(engine.info.get("draft_head")),
                # m5″ s2: the checkpoint carries DFlash2 drafter weights.
                # Still reported separately from drafters_available: as of s6
                # the two normally agree, but they diverge on a checkpoint
                # with drafter weights and no MTP head (the verify batch is
                # sized off the MTP head), and a drafter you cannot select
                # must not appear selectable.
                "drafter_weights_loaded":
                    bool(engine.info.get("drafter_weights")),
                # an earlier change prompt cache. `align` is reported because it is
                # the property that decides whether a warm answer is
                # byte-identical to a cold one: at kMaxSeq (2048) a resumed
                # prefill runs the same chunk boundaries and kernel tiers a
                # cold one does, and anything else does not (an earlier change).
                "prompt_cache": {
                    "enabled": bool(engine.info.get("cache_mb")),
                    "cap_mb": engine.info.get("cache_mb", 0),
                    "snapshot_align": engine.info.get("cache_align", 0),
                    "bitwise_identical_to_cold":
                        engine.info.get("cache_align") == 2048,
                },
                # an earlier change. The wire format is Qwen's XML-in-XML, NOT the
                # JSON-in-<tool_call> shape the name usually implies, and the
                # parser needs each tool's JSON Schema to type its arguments
                # so a request that omits `tools` gets
                # best-effort types, which is why that is said here.
                "tool_calls": {
                    "wire_format": "qwen-xml (<function=>/<parameter=>)",
                    "streaming": True,
                    "tool_choice": ["auto", "none", "required",
                                    "{type: function, function: {name}}"],
                    # A forced call is a prompt prefill, not a grammar, so
                    # 'required' is only EXACT when the prefill can name the
                    # function: a named tool_choice, or 'required' with
                    # exactly one tool. With several the model can still
                    # close an empty <function=>, and that returns 502.
                    "forced_call_is_a_prefill": True,
                    "required_is_exact": "named tool_choice, or one tool",
                    "parallel_tool_calls": True,
                    # Nothing constrains the grammar. A malformed call is
                    # returned as content, never repaired into a call the
                    # model did not make.
                    "constrained_decoding": False,
                    # A forced call is a prompt prefill and takes the slot
                    # the reasoning block would have used.
                    "forced_call_disables_thinking": True,
                    "argument_types_need_schema": True,
                },
                "supported": ["reasoning_effort", "enable_thinking",
                              "preserve_thinking", "tools", "tool_choice",
                              "parallel_tool_calls", "stop",
                              "max_tokens", "max_completion_tokens",
                              "max_output_tokens", "stream", "stream_options"],
                # All three name one budget over reasoning + content. Listed
                # separately because a client reads this to find out which
                # spelling it may use, and the modern ones used to be dropped
                # in silence.
                "token_budget_aliases": ["max_tokens", "max_completion_tokens",
                                         "max_output_tokens"],
                "max_tokens_default": ChatReq.model_fields["max_tokens"].default,
                # an earlier change: errors use OpenAI's {"error": {...}} envelope, not
                # FastAPI's {"detail": ...}, and a malformed body is a 400
                # rather than a 422.
                "error_format": "openai",
                "accepted_but_ignored": list(SAMPLING_FIELDS),
                # an earlier change. Stated explicitly because "temperature works now"
                # and "temperature works at the same speed" are different
                # claims, and a client cannot tell them apart from a response.
                "sampling": {
                    "implemented": ["temperature", "top_p", "top_k", "min_p",
                                    "seed", "presence_penalty",
                                    "frequency_penalty", "logit_bias",
                                    "logprobs"],
                    "sampler_only": "presence_penalty, frequency_penalty, "
                                    "logit_bias and logprobs are REJECTED "
                                    "without temperature>0 rather than "
                                    "ignored",
                    "top_logprobs": "not implemented; logprobs returns the "
                                    "chosen token only",
                    "temperature_0": "greedy; keeps the speculative fast path "
                                     "and is byte-identical",
                    "temperature_gt_0": "sampled, and it KEEPS the "
                                        "speculative fast path: "
                                        "accept min(1,p/q) with a residual "
                                        "correction, so the emitted "
                                        "distribution is exactly p. Measured "
                                        "2.1-2.3x serial-sampled",
                    "seed_absent": "a random seed is drawn; the request is "
                                   "then not reproducible",
                },
                "max_tokens_cap": max_cap,
                # an earlier change: a client cannot distinguish a cap that REJECTS from
                # one that silently truncates by looking at a response — both
                # end with finish_reason "length". Say which this is.
                "max_tokens_over_cap": "400",
                "reasoning_effort_values": sorted(set(EFFORT_MAP))}

    @app.get("/cache")
    async def cache():
        """Live prompt-cache counters. Separate from /health so a
        poll for hit rate does not have to take the engine lock on every
        health check — this one does, and /health stays free."""
        st = await engine.cache_stats()
        if st is None:
            raise HTTPException(501, "this engine reports no prompt cache")
        total = st["hits"] + st["misses"]
        st["hit_rate"] = round(st["hits"] / total, 4) if total else None
        return st

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": MODEL_ID, "object": "model",
                          "owned_by": "halogen", "created": 0}]}

    async def serve(ids, max_tokens, stops, stream, chat, prefix,
                    thinking=True, drafter=None, tools=None, parallel=True,
                    sample=None, penalty="",
                    pre="", forced=False, include_usage=False):
        # batch-1: QUEUE rather than reject. Reasoning defaults to xhigh, so
        # one request routinely runs minutes at ~10 t/s; failing every other
        # caller instantly for that whole window made the endpoint look dead
        # to a second client. Waiting is slow but correct; only a wait past
        # the timeout is a real 503.
        # VALIDATE BEFORE QUEUEING. Everything below the lock can wait
        # queue_timeout seconds (600 by default) for a slot; telling a client
        # its request was malformed only after ten minutes of waiting is
        # strictly worse than telling it now, and none of these checks need
        # the engine.
        #
        # an earlier change: this used to CLAMP — `max_tokens = min(max_tokens, max_cap)`
        # — and a benchmark run lost its results to it. Asking for 16,384 got
        # exactly 4,096 back with no error and no field saying the server had
        # reduced the request, so the output looked like the MODEL stopping
        # rather than the SERVER truncating. `finish_reason: "length"` is the
        # same value in both cases, which is precisely why it could not be
        # diagnosed from the response. Three lines up in run(), an over-length
        # PROMPT already raises a 400 that names the number; there was never a
        # reason for the output limit to behave differently, and an earlier change's
        # rule applies — an error the client cannot read is not an error.
        want = int(max_tokens)
        if want > max_cap:
            raise HTTPException(
                400, f"max_tokens {want} exceeds this server's cap of "
                     f"{max_cap}. The cap is server policy, not a model "
                     f"limit: halogen is batch-1, so one long request holds "
                     f"the GPU for its whole run and every other client "
                     f"queues behind it. Raise it with --max-tokens-cap; "
                     f"/health reports the live value as max_tokens_cap.")
        # The other ceiling, and the one that is physics rather than policy.
        # an earlier change's lesson, applied to the output side: ask the ENGINE for the
        # context, never a CLI default, because --context and Model::kMaxCtx
        # are two copies of one number and they have already drifted once.
        limit = engine.info.get("ctx") or ctx
        room = limit - len(ids)
        if room <= 0:
            raise HTTPException(
                400, f"prompt is {len(ids)} tokens which exceeds context "
                     f"window of {limit}.")
        max_tokens = min(want, room)

        # an earlier change: a SEMAPHORE of engine slots, not a single lock. At one slot
        # this is the batch-1 behaviour verbatim -- queue rather than reject,
        # because reasoning defaults to xhigh and failing every other caller
        # for that window made the endpoint look dead.
        engine.waiting += 1
        try:
            await asyncio.wait_for(engine.slots.acquire(),
                                   timeout=queue_timeout)
        except asyncio.TimeoutError:
            raise EngineBusy()
        finally:
            engine.waiting -= 1
        # WHEN this slot was taken, keyed by request. /health reports the
        # OLDEST live hold: "busy" alone cannot distinguish a long xhigh
        # request from a wedge, and that ambiguity is what made the
        # 2026-08-24 lockup look like normal load until the GPU counter was
        # checked. With N slots a single scalar could not express it at all.
        slot_key = uuid.uuid4().hex
        engine.inflight[slot_key] = time.monotonic()
        engine.busy_since = min(engine.inflight.values())
        cid = f"{prefix}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        split = ThinkSplit(thinking)
        tstream = ToolStream(tools) if tools else None
        if stream:
            async def body():
                try:
                    async for ev in sse(run(ids, max_tokens, stops, drafter,
                                            sample, penalty),
                                        cid, created, chat, split,
                                        tstream, pre, parallel,
                                        include_usage):
                        yield ev
                finally:
                    # client hung up (or errored) mid-stream: stop the engine
                    # instead of letting it decode to max_tokens with nobody
                    # listening, then hand the slot to whoever is queued.
                    #
                    # THE RELEASE IS IN ITS OWN finally AND ABORT IS CAPPED.
                    # It used to read `await engine.abort(); lock.release()`,
                    # so anything that made abort() hang or raise skipped the
                    # release entirely and the batch-1 lock was held forever.
                    # Releasing the slot is the part other clients depend on;
                    # tidying the stream is best-effort and must never be able
                    # to prevent it.
                    try:
                        await asyncio.wait_for(engine.abort(),
                                               timeout=ABORT_DRAIN_S + 5)
                    except Exception:
                        await engine.close()
                    finally:
                        engine.inflight.pop(slot_key, None)
                        engine.busy_since = (min(engine.inflight.values())
                                             if engine.inflight else None)
                        engine.slots.release()
            return StreamingResponse(body(), media_type="text/event-stream")
        try:
            text, done = "", None
            async for delta, d in run(ids, max_tokens, stops, drafter,
                                      sample, penalty):
                if d is not None:
                    done = d
                    break
                text += delta
        finally:
            # Same shape as the streaming arm above, and for the same reason.
            try:
                await asyncio.wait_for(engine.abort(),
                                       timeout=ABORT_DRAIN_S + 5)
            except Exception:
                await engine.close()
            finally:
                engine.inflight.pop(slot_key, None)
                engine.busy_since = (min(engine.inflight.values())
                                     if engine.inflight else None)
                engine.slots.release()
        fin = {"stop": "stop", "cancel": "stop",
               "length": "length"}.get(done["reason"], "stop")
        if chat:
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools)
                if not parallel:
                    tool_calls = tool_calls[:1]
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if forced and not tool_calls:
                # tool_choice asked for a call and the model declined by
                # closing an empty <function=>. Returning its markup as
                # content would look like an answer; this is a failure.
                raise HTTPException(502, "tool_choice required a call and the "
                                         "model did not name a function — "
                                         "retry, or name the function in "
                                         "tool_choice")
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if fin != "length":
                    fin = "tool_calls"
            choice = {"index": 0, "finish_reason": fin, "message": msg}
        else:
            choice = {"index": 0, "finish_reason": fin, "text": text}
        # an earlier change. OpenAI's shape, minus top_logprobs (rejected at the door, not
        # returned empty) and minus `bytes`, which would need a per-token
        # detokenisation this loop does not keep -- run() decodes the whole
        # list each step and diffs, so individual token text is not retained.
        # Reporting the fields we do not have as null beats inventing them.
        if done and done.get("logprobs"):
            choice["logprobs"] = {"content": [
                {"token": None, "logprob": v, "bytes": None,
                 "top_logprobs": []} for v in done["logprobs"]]}
        return {"id": cid, "created": created, "model": MODEL_ID,
                "object": "chat.completion" if chat else "text_completion",
                "choices": [choice], "usage": usage(done, len(ids))}

    @app.post("/v1/completions")
    async def completions(req: CompletionReq):
        ids = tok(req.prompt, add_special_tokens=False)["input_ids"]
        check_sampling(req)
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, False, "cmpl",
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           include_usage=wants_usage(req))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatReq):
        # FIRST, before the template and the tokenizer. This call was missing
        # entirely: /v1/completions had it and chat did not, so on the endpoint
        # everyone actually uses, presence_penalty / frequency_penalty /
        # logit_bias / logprobs without temperature>0 were SILENTLY DROPPED
        # (penalty_spec returns "" when there is no sampling, which puts the
        # engine's own rejection out of reach), and logprobs+stream and
        # top_logprobs were never refused. /health advertised the opposite.
        # an earlier change's rule, on the primary endpoint: a request not served the way
        # it was asked must say so.
        check_sampling(req)
        for f in SAMPLING_FIELDS:
            if getattr(req, f) is not None and f not in _warned_sampling:
                _warned_sampling.add(f)
                print(f"serve_api: '{f}' accepted but IGNORED "
                      f"by this endpoint", flush=True)
        mode, forced = tool_choice_mode(req)
        pre = force_prefill(mode, forced, req.tools)
        thinking = req.enable_thinking is not False
        kw = chat_kwargs(req, mode)
        if pre:
            # The prefill lands where the reasoning block would start, so the
            # think block is pre-closed instead of left dangling — otherwise
            # '</think>' never arrives and the whole turn misfiles as
            # reasoning. A forced call trades thinking for the guarantee.
            kw["enable_thinking"] = False
            thinking = False
        try:
            # an earlier change F3: OpenAI sends `arguments` as a JSON STRING and this
            # template indexes it as a mapping. Without this the second turn
            # of every tool loop is a 400.
            msgs = normalize_messages(req.messages)
        except ValueError as e:
            raise HTTPException(400, str(e))
        try:
            ids = tok.apply_chat_template(msgs, tokenize=True,
                                          add_generation_prompt=True,
                                          return_dict=False, **kw)
        except HTTPException:
            raise
        except Exception as e:                      # template raise_exception
            raise HTTPException(400, f"chat template rejected the request: "
                                     f"{e}")
        ids = list(ids)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, True, "chatcmpl",
                           thinking=thinking,
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           tools=(req.tools if mode != "none" else None),
                           parallel=req.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre),
                           include_usage=wants_usage(req))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="HF snapshot dir or model id (tokenizer only)")
    ap.add_argument("--engine", default="127.0.0.1:8730")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--context", type=int, default=32768,
                    help="FALLBACK only — the engine's INFO reports the real "
                         "kMaxCtx and that is what is enforced")
    ap.add_argument("--queue-timeout", type=float, default=7200,
                    help="seconds a request waits for the batch-1 slot "
                         "before 503")
    ap.add_argument("--max-tokens-cap", type=int, default=65536,
                    help="server-side ceiling on max_tokens (batch-1: one "
                         "long request blocks every other client)")
    args = ap.parse_args()

    import uvicorn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    host, _, port = args.engine.partition(":")
    engine = Engine(host, int(port))
    app = build_app(tok, engine, args.context, args.max_tokens_cap,
                    args.queue_timeout)

    @app.on_event("startup")
    async def _connect():
        await engine.connect()
        print(f"serve_api: engine at {args.engine}, "
              f"listening on {args.host}:{args.port}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
