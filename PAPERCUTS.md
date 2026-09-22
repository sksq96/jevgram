# Papercuts

Friction agents hit while working here. Newest day on top. To add yours:

```
papercut "what bit you"          # or ./papercut from the repo root
papercut list [n]               # read the newest n
```

Not on PATH? `ln -sf "$(git rev-parse --show-toplevel)/papercut" ~/.local/bin/papercut`

## 2026-09-22

- Running a cloned python CLI (bewinxed/jevgpt) with 'python3 -m pkg.cli' exited 0 with no output and no file, because the package wires main() only through a console-script entry point and has no __main__ guard. Ten minutes lost assuming the API call was silently failing. Check for 'if __name__' before blaming the network, or invoke the entry point directly: python3 -c 'from pkg.cli import main; main()' -- args. (claude-code_2-1-278_agent)

## 2026-09-21

- Pulling a stratified sample from a big HF dataset, the datasets-server /filter and /api/datasets/.../parquet endpoints rate-limited this box's IP (429, 'create a HF account and pass HF_TOKEN') after a dozen calls, and the parquet conversion only covers the first 5GB of a dataset anyway. Streaming the source CSV off huggingface.co/datasets/.../resolve/main/ with curl and sampling in pyarrow worked at 52MB/s with no limit; an HF_TOKEN in the box's env would have avoided the detour entirely. (claude-code_2-1-278_agent)
- Reading the twitter lane's voice/ledger.md and voice/pangram.py mid-session, both files vanished from the working tree while that session reorganised the directory, so a re-read failed with no such file. 'git show HEAD:voice/ledger.md' recovered them. When one lane reads another live repo, read through git rather than the working tree. (claude-code_2-1-278_agent)
