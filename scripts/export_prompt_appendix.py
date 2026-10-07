"""Generate the paper's prompt-disclosure appendix (Appendix C) from the prompt
registry, the single-agent prompt constants, and the run database. $0, no LLM.

Requested by Dr. Braga-Neto (2026-09-10): "we need to disclose (in the
Appendix) the prompts you gave to the LLMs."

What is disclosed, and where each piece comes from:
  1. Every registered system prompt (ascension.common.prompts registry: the
     Council role prompts, the Solo baseline, the director prompts, the Justice
     court prompts, and the Agora contrastive-distillation prompt), verbatim,
     with its SHA-256 so a reader can match it against the per-call
     `system_prompt_hash` in the released run records.
  2. The single-agent ODE prompts (ascension.agent SYSTEM_PROMPT_V1..V6).
  3. A usage table: for every system-prompt hash that appears in the run
     database since 2026-05-30 (the window of every experiment in the paper),
     how many calls, how many runs, first and last date. Registered prompts
     that were never called in that window are listed as such.
  4. Two fully rendered requests, verbatim from the database: the first
     request one Council role received in a family-only run (showing the
     multi-configuration family feed) and the last request one role received
     in an instrument-only run (showing the deterministic instrument's
     feedback blocks). The placeholders in the templates of (1) are what
     these fill.

Output: LaTeX using the `listings` package with line wrapping. Written to
the Overleaf-synced paper repo as appendix_prompts.tex.

Run: poetry run python scripts/export_prompt_appendix.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "appendix_prompts.tex"
WINDOW_START = "2026-05-30"

FAMILY_RUN = "f507b784-9eb9-4f67-addb-0369e7e0328a"      # EXP-089 family_only seed 4
FAMILY_HASH_PREFIX = "806417c6a5ab"                       # aristotle_hypothesis 1.0
INSTRUMENT_RUN = "b6cbe34a-ae58-4d74-af8f-f94b107e0142"  # EXP-089 instrument_only seed 4
INSTRUMENT_HASH_PREFIX = "f61781f63774"                   # parmenides_hypothesis 1.0


def psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "ascension-postgres", "psql", "-U", "ascension", "-d", "ascension", "-tA", "-F", "\t", "-c", sql],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def tex_escape_text(s: str) -> str:
    """Escape for ordinary LaTeX text (labels, captions), not for listings."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def listing(body: str, caption: str) -> str:
    # listings' verbatim-like environment; the body is NOT escaped. The only
    # thing that can break it is the literal string "\end{lstlisting}", which
    # none of the prompts contain (checked below).
    assert "\\end{lstlisting}" not in body
    body = body.replace("\t", "    ")
    return (
        "\\Needspace{8\\baselineskip}\n"
        "\\begin{lstlisting}[caption={" + tex_escape_text(caption) + "}]\n"
        + body.rstrip("\n") + "\n"
        + "\\end{lstlisting}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # The prompt registry is shipped as prompts/ (one file per registered prompt, with its
    # SHA-256 in prompts/index.tsv); the packages that register them at import time are not
    # part of this release.
    from ascension.common.prompts import Prompt

    registry: list[Prompt] = []
    agent_prompts: list[tuple[str, str]] = []
    for line in (REPO / "prompts/index.tsv").read_text().splitlines()[1:]:
        name, version, sha, _chars, fname = line.split("\t")
        body = (REPO / "prompts" / fname).read_text()
        assert hashlib.sha256(body.encode("utf-8")).hexdigest() == sha, fname
        if name.startswith("single_agent_ode_loop"):
            agent_prompts.append((f"single-agent ODE loop, {name.split()[-1]}", body))
        else:
            registry.append(Prompt(name=name, version=version, body=body))

    # usage since the window start
    rows = psql(
        "select system_prompt_hash, count(*), count(distinct run_id), min(created_at)::date, max(created_at)::date "
        "from llm_calls where created_at >= '%s' and system_prompt_hash is not null and system_prompt_hash <> '' "
        "group by 1 order by 2 desc;" % WINDOW_START
    ).strip().splitlines()
    usage = {}
    for r in rows:
        h, n, nr, d0, d1 = r.split("\t")
        usage[h] = (int(n), int(nr), d0, d1)
    embed = psql("select count(*) from llm_calls where created_at >= '%s' and model like 'gemini-embedding%%';" % WINDOW_START).strip()

    known = {}
    for p in registry:
        known[p.body_sha256] = (f"{p.name} {p.version}", p.body)
    for label, body in agent_prompts:
        known[hashlib.sha256(body.encode("utf-8")).hexdigest()] = (label, body)

    # any hash used in the window that is not in the registry or the agent constants
    unknown = [h for h in usage if h not in known]
    unknown_bodies = {}
    for h in unknown:
        b = psql("select request->>'system' from llm_calls where system_prompt_hash='%s' limit 1;" % h).rstrip("\n")
        unknown_bodies[h] = b

    # rendered examples
    def rendered(run: str, hp: str, last: bool) -> str:
        order = "desc" if last else "asc"
        return psql(
            "select request->>'contents' from llm_calls where run_id='%s' and system_prompt_hash like '%s%%' "
            "order by created_at %s limit 1;" % (run, hp, order)
        ).rstrip("\n")

    def collapse_numeric_runs(text: str, keep_head: int = 3, keep_tail: int = 1) -> str:
        """The rendered requests carry the observed trajectories as long runs of
        numeric lines (t=..., x1=..., ...). Keep the first and last few of each run
        and replace the rest with an explicit marker, so the reader sees the format
        without three pages of numbers. Everything else is untouched."""
        import re as _re
        lines = text.split("\n")
        is_num = [bool(_re.match(r"^\s*t=[-\d.]+\s", ln)) for ln in lines]
        out_lines, i = [], 0
        while i < len(lines):
            if not is_num[i]:
                out_lines.append(lines[i]); i += 1; continue
            j = i
            while j < len(lines) and is_num[j]:
                j += 1
            run = lines[i:j]
            if len(run) > keep_head + keep_tail + 1:
                out_lines.extend(run[:keep_head])
                out_lines.append(f"    [... {len(run) - keep_head - keep_tail} further sampled timesteps omitted here; present verbatim in the run record ...]")
                out_lines.extend(run[-keep_tail:])
            else:
                out_lines.extend(run)
            i = j
        return "\n".join(out_lines)

    fam = collapse_numeric_runs(rendered(FAMILY_RUN, FAMILY_HASH_PREFIX, last=False))
    ins = collapse_numeric_runs(rendered(INSTRUMENT_RUN, INSTRUMENT_HASH_PREFIX, last=True))

    out = []
    out.append("\\section{Prompts given to the language models}\\label{appendix-prompts}\n")
    out.append(
        "Every prompt a language model received in the experiments of Section~6 is reproduced here verbatim, "
        "generated by \\texttt{scripts/export\\_prompt\\_appendix.py} from the prompt registry and the run database, "
        "so the text below is the text that was sent. Each system prompt carries its SHA-256; every model call in the "
        "released run records stores the SHA-256 of the system prompt it used, so a reader can match any call to its prompt. "
        "The role prompts are templates with named slots (judging criteria, benchmark task description, recent Agora context, "
        "role emphasis, output-format instructions, and the candidate count); Section~\\ref{appendix-prompts-rendered} shows two fully rendered requests "
        "so the reader can see what the slots contained. The deterministic instrument's feedback enters the Agora-context slot "
        "as the two blocks visible in the second rendered request. No language model participates in scoring; the scorer's "
        "code contains no model client, and an automated source check enforces that (Section~9). The Argus overseer in every "
        "reported run is a deterministic poller and issues no model calls.\n"
    )

    # C.1 usage table
    out.append("\\subsection{Which prompts were called}\\label{appendix-prompts-usage}\n")
    out.append(
        "Model calls in the run database from %s onward, the window that contains every experiment in Section~6 except "
        "the two early negatives EXP-072 and EXP-074, grouped by system prompt. Embedding calls (\\texttt{gemini-embedding-001}, %s calls, no system prompt) are omitted.\n"
        % (WINDOW_START, embed)
    )
    out.append("\\footnotesize\\begin{longtable}{@{}p{0.34\\linewidth}p{0.16\\linewidth}rrp{0.22\\linewidth}@{}}\n\\toprule\n"
               "prompt & SHA-256 (first 12) & calls & runs & first to last date \\\\\n\\midrule\n\\endhead\n")
    for h, (n, nr, d0, d1) in sorted(usage.items(), key=lambda kv: -kv[1][0]):
        label = known[h][0] if h in known else "unregistered (see C.5)"
        out.append(f"{tex_escape_text(label)} & \\texttt{{{h[:12]}}} & {n} & {nr} & {d0} to {d1} \\\\\n")
    out.append("\\bottomrule\n\\end{longtable}\\normalsize\n")
    never = sorted({known[h][0] for h in known if h not in usage})
    out.append(
        "The registry also holds %d prompts with no hashed call in this window: the 0.1 and 2.0-tier2 variants of the five "
        "role prompts, the three Justice court prompts (proponent, critic, judge), the two director prompts and their system "
        "prompt, the 0.1 distillation prompt, and the six system prompts of the April single-agent ODE loop. One of these "
        "was used by a result: the director prompt made the six calls of the live loop (EXP-078), which the run database "
        "stored without a system-prompt hash, so they are absent from the table above. The others were not used by any "
        "result. They ship with the code release and are omitted here.\n"
        % len(never)
    )

    # C.2 registry prompts, used ones first
    out.append("\\subsection{The system prompts}\\label{appendix-prompts-registry}\n")
    out.append("Verbatim from the registry, in order of calls. Text in braces is a template slot filled at call time.\n")
    used_sorted = sorted([p for p in registry if p.body_sha256 in usage],
                         key=lambda p: (-usage[p.body_sha256][0], p.name, p.version))
    for p in used_sorted:
        n = usage[p.body_sha256][0]
        cap = f"{p.name.replace('_', ' ')} {p.version}, SHA-256 {p.body_sha256[:12]}, {n} calls"
        out.append(listing(p.body, cap))

    # C.4 rendered examples
    out.append("\\subsection{Two fully rendered requests}\\label{appendix-prompts-rendered}\n")
    out.append(
        "The first request the Aristotle role received in an EXP-089 family-only run (seed 4, run "
        + FAMILY_RUN[:8] + "), showing the multi-configuration family feed, and the last request the Parmenides role received "
        "in an EXP-089 instrument-only run (seed 4, run " + INSTRUMENT_RUN[:8] + "), showing the two deterministic feedback "
        "blocks. Both verbatim from the run database, except that the long runs of sampled trajectory lines are shortened to their first three and last entries with a marker in place of the rest.\n"
    )
    out.append(listing(fam, f"Rendered request: EXP-089 family-only run {FAMILY_RUN[:8]}, Aristotle, first call"))
    out.append(listing(ins, f"Rendered request: EXP-089 instrument-only run {INSTRUMENT_RUN[:8]}, Parmenides, last call"))

    # C.5 unregistered
    if unknown:
        out.append("\\subsection{System prompts used in the window but not in the registry}\\label{appendix-prompts-unregistered}\n")
        out.append("These reached a model in the window but are not registered prompts or single-agent constants; each is reproduced from the run database.\n")
        for h in unknown:
            n, nr, d0, d1 = usage[h]
            out.append(listing(unknown_bodies[h], f"unregistered (SHA-256 {h[:12]}; {n} calls, {nr} runs, {d0} to {d1})"))

    args.out.write_text("".join(out))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes); {len(usage)} used hashes, {len(unknown)} unregistered, {len(never)} registered-but-unused")


if __name__ == "__main__":
    main()
