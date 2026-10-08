"""Scripted demo of the speech-language agent against the local API.

    python demo/run_demo.py              # starts the server if needed, runs every step, stops it
    python demo/run_demo.py --pause      # wait for Enter between steps (live walkthrough)
    python demo/run_demo.py --no-server  # use a server you already started

Needs GROQ_API_KEY in .env (see .env.example). Pronunciation needs audio and Azure keys, so it is not demoed here.
Uses only the standard library."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.getenv("PORT", "8000"))
BASE = f"http://localhost:{PORT}"
sys.stdout.reconfigure(line_buffering=True)
# the demo writes its own traces, so the report in step 6 covers this run only
os.environ.setdefault("TRACE_DIR", os.path.join(ROOT, "demo", "traces"))


def http(path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def alive():
    try:
        return json.load(http("/healthz", timeout=2)).get("status") == "ok"
    except Exception:
        return False


def start_server():
    log = open(os.path.join(ROOT, "demo", "server.log"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    for _ in range(120):
        if alive():
            return proc
        if proc.poll() is not None:
            sys.exit("server exited early, see demo/server.log")
        time.sleep(1)
    proc.terminate()
    sys.exit("server did not start within 2 minutes, see demo/server.log")


def title(n, text):
    print(f"\n{'=' * 72}\n[{n}] {text}\n{'=' * 72}")


def show_feedback(r):
    print(f"intent: {r['intent']}   degraded: {r['degraded'] or 'none'}   trace_id: {r['trace_id']}")
    g = r.get("grammar")
    if g:
        if g["errors"]:
            for e in g["errors"]:
                print(f"  grammar : '{e['original']}' -> '{e['correction']}'  [{e['error_type']}]")
                print(f"            {e['explanation']}")
        else:
            print("  grammar : no error found")
    v = r.get("vocabulary")
    if v and v["suggestions"]:
        for s in v["suggestions"][:2]:
            print(f"  vocab   : '{s['original']}' -> '{s['better']}'")
    a = r.get("answer")
    if a:
        print(f"  answer  : {a['answer'][:240]}")
        print(f"  grounded: {a['grounded']}   retrieved: {r['retrieved_sections']}")


def feedback(text, lang):
    t0 = time.perf_counter()
    r = json.load(http("/v1/feedback", {"text": text, "target_language": lang}))
    show_feedback(r)
    print(f"  ({(time.perf_counter() - t0) * 1000:.0f} ms end to end)")


def stream(text, lang):
    t0 = time.perf_counter()
    resp = http("/v1/feedback/stream", {"text": text, "target_language": lang})
    event = None
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            payload = json.loads(line[6:])
            detail = payload.get("intent") or ("done" if event == "done" else ", ".join(payload.keys()))
            print(f"  +{(time.perf_counter() - t0) * 1000:5.0f} ms  event={event:<11} {detail}")
            event = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--no-server", action="store_true")
    args = ap.parse_args()
    proc = None
    if not alive():
        if args.no_server:
            sys.exit(f"no server on {BASE}")
        print("starting server (first start loads the embedding model, about 10 to 20 s)...")
        proc = start_server()

    def step(n, text, fn):
        if args.pause:
            input("\n(press Enter) ")
        title(n, text)
        fn()

    try:
        step(1, "Readiness: does every configured model answer?",
             lambda: print(json.dumps(json.load(http("/readyz")), indent=2)))
        step(2, "Practice sentence with a word-order error (German)",
             lambda: feedback("Gestern ich habe Fußball gespielt.", "German"))
        step(3, "A question about the language goes to the RAG branch",
             lambda: feedback("When do I use the present perfect instead of the simple past?", "English"))
        step(4, "A correct sentence: no invented errors",
             lambda: feedback("She goes to the gym every morning.", "English"))
        step(5, "Streaming: one event per finished node",
             lambda: stream("Gestern ich habe Fußball gespielt.", "German"))
        step(6, "Per-request traces of this run: latency, fallbacks, cost",
             lambda: subprocess.run([sys.executable, "scripts/trace_report.py"], cwd=ROOT, env=os.environ.copy()))
        step(7, "Known weakness, shown on purpose: German dative error (should be 'mir')",
             lambda: feedback("Kannst du mich helfen?", "German"))
    finally:
        if proc:
            proc.terminate()
    print("\nDone.")


if __name__ == "__main__":
    main()
