"""Stream RAID's 11.8GB train.csv once and keep a sampled slice -> data/raid_raw.parquet.

The HF parquet conversion stops at 5GB (only 4 of the 11 domains survive) and its API is IP
rate-limited, so we read the source CSV straight off the CDN and sample as it flows; nothing
but the slice touches disk. train.csv holds 8 domains (code/czech/german live in extra.csv).
Human rows are ~0.4% of the file, so they are kept whole and the rest is thinned.

  curl -sL https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv | python3 raid_pull.py
"""
import hashlib, sys, pyarrow as pa, pyarrow.csv as pcsv, pyarrow.parquet as pq

COLS = ["id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
OUT = "/root/github/jevgram/data/raid_raw.parquet"

def keep(i, model, attack):
    if model == "human" and attack == "none":
        return True                      # rare: keep all
    h = int(hashlib.md5(i.encode()).hexdigest()[:6], 16)
    return h % (40 if attack == "none" else 200) == 0

reader = pcsv.open_csv(sys.stdin.buffer,
                       read_options=pcsv.ReadOptions(block_size=1 << 26),
                       parse_options=pcsv.ParseOptions(newlines_in_values=True))
writer, n_in, n_out = None, 0, 0
for batch in reader:
    t = pa.Table.from_batches([batch]).select(COLS)
    mask = pa.array(list(map(keep, t.column("id").to_pylist(),
                             t.column("model").to_pylist(), t.column("attack").to_pylist())))
    t = t.filter(mask)
    n_in += batch.num_rows; n_out += t.num_rows
    if t.num_rows:
        writer = writer or pq.ParquetWriter(OUT, t.schema)
        writer.write_table(t)
    if n_in % (1 << 20) < batch.num_rows:
        print(f"{n_in/1e6:.1f}M read, {n_out} kept", flush=True)
if writer:
    writer.close()
print(f"DONE {n_in} read, {n_out} kept", flush=True)
