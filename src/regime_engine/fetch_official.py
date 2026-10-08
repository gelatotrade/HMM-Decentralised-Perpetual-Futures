"""Recover the Oct-Nov 2025 BTC funding premium from an authoritative source.

PURPOSE
-------
The paper's headline result (the mean-reversion half-life asymmetry in the
funding premium) depends on the 2 Oct - 30 Nov 2025 segment. That segment is
absent from the shipped hourly panel (`data/processed/panel_BTC_subsample.csv`,
which starts 2025-12-03), and the hourly candles for it cannot be re-pulled from
the `candleSnapshot` endpoint, which retains only ~5,000 hourly candles.

RETENTION (checked 2026-08-18)
------------------------------
The ~5,000-record retention limit applies to `candleSnapshot`, NOT to
`fundingHistory`. The `fundingHistory` info endpoint is free, unauthenticated,
and serves BTC `fundingRate` + `premium` back to 2023-05-12 with no retention
limit at all. It returns the full Oct-Nov 2025 window, hour for hour, including
the full-sample premium maximum of +23.6132 bps at 2025-10-10 22:00 UTC.

The segment is absent from the hourly panel because the panel is built by an
INNER MERGE of funding against candles (`panel.build_hourly_panel`, called from
`fetch_data_legacy.get_dataset()`). Candles are retention-capped; funding is
not. The join truncates the panel to the candle window, so the premium rows are
discarded locally although the endpoint still serves them.

TWO SOURCES, BOTH IMPLEMENTED
-----------------------------
`--source rest` (DEFAULT)
    api.hyperliquid.xyz/info `fundingHistory`. Free, no AWS account, no
    credentials, no egress charges. Returns exactly {coin, fundingRate,
    premium, time} -- the panel schema, natively. Bit-exact against the
    shipped panel on the Dec 2025 - Apr 2026 overlap (max |dpremium| = 0.0
    over all 3,507 shared hours).

`--source s3`
    s3://hyperliquid-archive/asset_ctxs/[YYYYMMDD].csv.lz4 (us-east-1,
    requester-pays). An INDEPENDENT operator-published archive, useful as a
    provenance cross-check, not as the primary route. Requires an
    AWS account. Cost for BTC-only over ~60 days is under one US cent.

MODES
-----
    --dry-run     verify access, list what WOULD be fetched, report bytes and
                  estimated cost, and report the archive's true coverage.
                  No downloads. Run this first.
    --validate    fetch the overlap with the shipped panel and report the max
                  absolute premium difference. This is the acceptance test.
    --self-test   offline unit tests of the pure parsing/alignment/schema code
                  against a synthetic LZ4-compressed CSV fixture. No network.
    (default)     fetch the requested range and write the panel-schema CSV.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import struct
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "asset_ctxs"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_BTC_subsample.csv"

BUCKET = "hyperliquid-archive"
BUCKET_REGION = "us-east-1"  # verified via x-amz-bucket-region, 2026-08-18
ASSET_CTXS_PREFIX = "asset_ctxs/"

REST_URL = "https://api.hyperliquid.xyz/info"
REST_PAGE_LIMIT = 500  # fundingHistory returns at most 500 records per call

# AWS S3 us-east-1 list pricing (checked 2026-08-18).
USD_PER_GET = 0.0004 / 1000.0
USD_PER_LIST = 0.005 / 1000.0
USD_PER_GB_EGRESS = 0.09  # first 100 GB/month are free tier
FREE_EGRESS_GB_PER_MONTH = 100.0

# asset_ctxs column aliases. The canonical snake_case order is
#   time, coin, funding, open_interest, prev_day_px, day_ntl_vlm,
#   premium, oracle_px, mark_px, mid_px, impact_bid_px, impact_ask_px
# We resolve by NAME, never by position, and accept camelCase variants,
# because the exact archive header is not published by Hyperliquid.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "timestamp", "ts", "datetime", "date"),
    "coin": ("coin", "asset", "name", "symbol"),
    "funding": ("funding", "fundingrate", "funding_rate"),
    "premium": ("premium", "premium_index", "premiumindex"),
    "oracle_px": ("oracle_px", "oraclepx", "oracle_price"),
    "mark_px": ("mark_px", "markpx", "mark_price"),
    "mid_px": ("mid_px", "midpx", "mid_price"),
}

PANEL_COLUMNS = ["time", "fundingRate", "premium"]


class RecoveryError(RuntimeError):
    """Actionable failure: the message is meant to be read by the user."""


# --------------------------------------------------------------------------
# Date helpers (pure)
# --------------------------------------------------------------------------


def parse_day(s: str) -> datetime:
    """Parse YYYY-MM-DD or YYYYMMDD into a UTC midnight datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise RecoveryError(
        f"Could not parse date {s!r}. Use YYYY-MM-DD (e.g. 2025-10-02)."
    )


def day_keys(start: datetime, end: datetime) -> list[str]:
    """S3 keys for every UTC day in [start, end], end inclusive.

    >>> day_keys(parse_day("2025-10-01"), parse_day("2025-10-03"))
    ['asset_ctxs/20251001.csv.lz4', 'asset_ctxs/20251002.csv.lz4', 'asset_ctxs/20251003.csv.lz4']
    """
    if end < start:
        raise RecoveryError(
            f"--end ({end:%Y-%m-%d}) is before --start ({start:%Y-%m-%d})."
        )
    keys = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= last:
        keys.append(f"{ASSET_CTXS_PREFIX}{day:%Y%m%d}.csv.lz4")
        day += timedelta(days=1)
    return keys


def key_to_date(key: str) -> datetime | None:
    """Extract the UTC date from an asset_ctxs key, or None if it does not match."""
    stem = key.rsplit("/", 1)[-1]
    if not stem.endswith(".csv.lz4"):
        return None
    digits = stem[: -len(".csv.lz4")]
    if len(digits) != 8 or not digits.isdigit():
        return None
    try:
        return datetime.strptime(digits, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------
# Cost model (pure)
# --------------------------------------------------------------------------


def estimate_cost(
    n_objects: int, total_bytes: int, n_list_calls: int = 1, free_tier: bool = True
) -> dict[str, float]:
    """Estimate the USD cost of a requester-pays pull.

    Returns a dict with the request charge, the egress charge under both the
    free-tier and the full-rate assumption, and the total.
    """
    gb = total_bytes / (1024.0 ** 3)
    request_usd = n_objects * USD_PER_GET + n_list_calls * USD_PER_LIST
    egress_full_usd = gb * USD_PER_GB_EGRESS
    egress_usd = 0.0 if (free_tier and gb <= FREE_EGRESS_GB_PER_MONTH) else egress_full_usd
    return {
        "n_objects": float(n_objects),
        "total_bytes": float(total_bytes),
        "total_mib": total_bytes / (1024.0 ** 2),
        "request_usd": request_usd,
        "egress_usd": egress_usd,
        "egress_full_rate_usd": egress_full_usd,
        "total_usd": request_usd + egress_usd,
        "total_usd_no_free_tier": request_usd + egress_full_usd,
    }


# --------------------------------------------------------------------------
# LZ4 frame handling
# --------------------------------------------------------------------------

_LZ4_MAGIC = b"\x04\x22\x4d\x18"
_XXH_P1, _XXH_P2, _XXH_P3, _XXH_P4, _XXH_P5 = (
    2654435761,
    2246822519,
    3266489917,
    668265263,
    374761393,
)
_M32 = 0xFFFFFFFF


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M32


def xxh32(data: bytes, seed: int = 0) -> int:
    """Pure-Python xxHash32. Needed for the LZ4 frame header checksum byte.

    Only used to BUILD the offline test fixture and to validate frame headers;
    real decompression goes through the `lz4` package.
    """
    n = len(data)
    i = 0
    if n >= 16:
        v1 = (seed + _XXH_P1 + _XXH_P2) & _M32
        v2 = (seed + _XXH_P2) & _M32
        v3 = seed & _M32
        v4 = (seed - _XXH_P1) & _M32
        while i + 16 <= n:
            for slot in range(4):
                lane = struct.unpack_from("<I", data, i + 4 * slot)[0]
                v = (v1, v2, v3, v4)[slot]
                v = _rotl32((v + lane * _XXH_P2) & _M32, 13)
                v = (v * _XXH_P1) & _M32
                if slot == 0:
                    v1 = v
                elif slot == 1:
                    v2 = v
                elif slot == 2:
                    v3 = v
                else:
                    v4 = v
            i += 16
        h = (_rotl32(v1, 1) + _rotl32(v2, 7) + _rotl32(v3, 12) + _rotl32(v4, 18)) & _M32
    else:
        h = (seed + _XXH_P5) & _M32
    h = (h + n) & _M32
    while i + 4 <= n:
        lane = struct.unpack_from("<I", data, i)[0]
        h = (h + lane * _XXH_P3) & _M32
        h = (_rotl32(h, 17) * _XXH_P4) & _M32
        i += 4
    while i < n:
        h = (h + data[i] * _XXH_P5) & _M32
        h = (_rotl32(h, 11) * _XXH_P1) & _M32
        i += 1
    h ^= h >> 15
    h = (h * _XXH_P2) & _M32
    h ^= h >> 13
    h = (h * _XXH_P3) & _M32
    h ^= h >> 16
    return h & _M32


def build_lz4_frame(payload: bytes, block_size: int = 1 << 16) -> bytes:
    """Build a spec-valid LZ4 frame using STORED (uncompressed) blocks.

    Pure Python, no dependencies. This exists so the offline self-test can
    construct a genuine .lz4 fixture on a machine where the `lz4` package is
    not installed. It is a test helper, not a compressor.

    Frame layout: magic | FLG | BD | HC | (blocks) | EndMark
      FLG = 0x60  -> version 01, block-independence, no checksums, no size
      BD  = 0x70  -> 4 MiB max block size
      HC  = (xxh32(FLG,BD) >> 8) & 0xFF
      block = <4-byte LE size with high bit set> <raw bytes>
    """
    flg, bd = 0x60, 0x70
    header = bytes((flg, bd))
    hc = (xxh32(header) >> 8) & 0xFF
    out = bytearray(_LZ4_MAGIC + header + bytes((hc,)))
    for off in range(0, len(payload), block_size):
        chunk = payload[off : off + block_size]
        out += struct.pack("<I", len(chunk) | 0x80000000)
        out += chunk
    out += struct.pack("<I", 0)  # EndMark
    return bytes(out)


def _decompress_stored_frame(blob: bytes) -> bytes:
    """Decompress an LZ4 frame that contains only STORED blocks.

    Fallback used by the self-test when the `lz4` package is unavailable.
    Raises RecoveryError on any compressed block, since decoding those requires
    the real library.
    """
    if not blob.startswith(_LZ4_MAGIC):
        raise RecoveryError("Not an LZ4 frame (bad magic number).")
    pos = 4
    flg, bd = blob[pos], blob[pos + 1]
    pos += 2
    if (flg >> 6) != 1:
        raise RecoveryError("Unsupported LZ4 frame version.")
    expected_hc = (xxh32(bytes((flg, bd))) >> 8) & 0xFF
    if blob[pos] != expected_hc:
        raise RecoveryError("LZ4 frame header checksum mismatch.")
    pos += 1
    if flg & 0x08:  # content size present
        pos += 8
    if flg & 0x01:  # dict id present
        pos += 4
    out = bytearray()
    while True:
        (size_field,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        if size_field == 0:
            break
        size = size_field & 0x7FFFFFFF
        if not (size_field & 0x80000000):
            raise RecoveryError(
                "This LZ4 frame contains compressed blocks. Install the `lz4` "
                "package to read it:  python3 -m pip install --user 'lz4>=4.3'"
            )
        out += blob[pos : pos + size]
        pos += size
        if flg & 0x10:  # block checksum present
            pos += 4
    return bytes(out)


def lz4_decompress(blob: bytes) -> bytes:
    """Decompress an LZ4 frame, preferring the real library."""
    try:
        import lz4.frame  # type: ignore
    except ImportError:
        return _decompress_stored_frame(blob)
    return lz4.frame.decompress(blob)


# --------------------------------------------------------------------------
# Parsing / alignment (pure -- these are what the self-test covers)
# --------------------------------------------------------------------------


def _resolve_columns(header: Sequence[str]) -> dict[str, int]:
    """Map canonical field names to column indices in an asset_ctxs header."""
    norm = [h.strip().lower().replace(" ", "") for h in header]
    resolved: dict[str, int] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.replace("_", "")
            for idx, name in enumerate(norm):
                if name.replace("_", "") == key:
                    resolved[canonical] = idx
                    break
            if canonical in resolved:
                break
    missing = [c for c in ("time", "coin", "funding", "premium") if c not in resolved]
    if missing:
        raise RecoveryError(
            f"asset_ctxs header does not contain the required columns {missing}.\n"
            f"Observed header: {list(header)}\n"
            "The archive schema has changed. Update COLUMN_ALIASES in "
            "src/regime_engine/fetch_official.py, or use --source rest, which "
            "does not depend on this schema."
        )
    return resolved


def _parse_timestamp(raw: str) -> datetime:
    """Parse an asset_ctxs timestamp. Accepts epoch ms, epoch s, or ISO-8601."""
    s = raw.strip()
    if not s:
        raise RecoveryError("Empty timestamp in asset_ctxs row.")
    if s.isdigit():
        val = int(s)
        # > 1e11 means milliseconds; below that, seconds.
        if val > 100_000_000_000:
            return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(val, tz=timezone.utc)
    iso = s.replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise RecoveryError(f"Unrecognised timestamp format: {raw!r}") from exc
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_asset_ctxs_csv(blob: bytes, coin: str = "BTC") -> list[dict[str, object]]:
    """Parse a (decompressed) asset_ctxs CSV, returning rows for one coin.

    Returns a list of dicts with keys: time (tz-aware UTC datetime),
    fundingRate (float), premium (float), and mark_px/oracle_px when present.
    Columns are located by NAME, so column order is irrelevant.
    """
    text = blob.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise RecoveryError("asset_ctxs file is empty.") from exc
    idx = _resolve_columns(header)
    want = coin.strip().upper()
    rows: list[dict[str, object]] = []
    for lineno, fields in enumerate(reader, start=2):
        if not fields or len(fields) <= max(idx.values()):
            continue
        if fields[idx["coin"]].strip().upper() != want:
            continue
        try:
            row: dict[str, object] = {
                "time": _parse_timestamp(fields[idx["time"]]),
                "fundingRate": float(fields[idx["funding"]]),
                "premium": float(fields[idx["premium"]]),
            }
        except (ValueError, RecoveryError) as exc:
            raise RecoveryError(
                f"Malformed asset_ctxs row at line {lineno}: {exc}"
            ) from exc
        for extra in ("mark_px", "oracle_px", "mid_px"):
            if extra in idx and fields[idx[extra]].strip():
                try:
                    row[extra] = float(fields[idx[extra]])
                except ValueError:
                    pass
        rows.append(row)
    return rows


def align_hourly(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Floor timestamps to the hour, drop duplicate hours (keep first), sort.

    Matches the convention in fetch_data_legacy.py: `.dt.floor("h")` then
    `drop_duplicates("time")` (which keeps the first occurrence) then sort.
    """
    seen = set()
    out: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda r: r["time"]):  # type: ignore[index,arg-type]
        floored = row["time"].replace(  # type: ignore[union-attr]
            minute=0, second=0, microsecond=0
        )
        if floored in seen:
            continue
        seen.add(floored)
        new = dict(row)
        new["time"] = floored
        out.append(new)
    return out


def to_panel_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project onto the panel schema: [time, fundingRate, premium]."""
    return [{k: r[k] for k in PANEL_COLUMNS} for r in rows]


def missing_hours(rows: list[dict[str, object]]) -> list[datetime]:
    """UTC hours absent between the first and last observation."""
    if len(rows) < 2:
        return []
    times = sorted(r["time"] for r in rows)  # type: ignore[misc]
    present = set(times)
    gaps = []
    cur = times[0]
    while cur <= times[-1]:
        if cur not in present:
            gaps.append(cur)
        cur += timedelta(hours=1)
    return gaps


# --------------------------------------------------------------------------
# Source A: fundingHistory REST (free, default)
# --------------------------------------------------------------------------


def _rest_post(payload: dict, max_retries: int = 5, timeout: int = 45) -> object:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            REST_URL, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RecoveryError(
        f"fundingHistory request failed after {max_retries} attempts: {last}\n"
        f"Check your network connection. The endpoint is {REST_URL} and needs no "
        "authentication."
    )


def fetch_rest(
    coin: str, start: datetime, end: datetime, verbose: bool = True
) -> list[dict[str, object]]:
    """Paginate fundingHistory over [start, end). Free; no credentials."""
    start_ms, end_ms = to_ms(start), to_ms(end)
    raw: list[dict] = []
    cursor, page = start_ms, 0
    while cursor < end_ms:
        page += 1
        chunk = _rest_post(
            {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": cursor,
                "endTime": end_ms,
            }
        )
        if not isinstance(chunk, list) or not chunk:
            break
        raw.extend(chunk)
        last_t = int(chunk[-1]["time"])
        if last_t + 1 <= cursor:
            break
        cursor = last_t + 1
        if verbose and page % 5 == 0:
            at = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)
            print(f"  page {page:>3}: {len(raw):>6} records, cursor {at:%Y-%m-%d %H:%M}")
        if len(chunk) < REST_PAGE_LIMIT:
            break
        time.sleep(0.15)  # be polite
    if verbose:
        print(f"  {page} pages, {len(raw)} raw records")
    rows = [
        {
            "time": datetime.fromtimestamp(int(r["time"]) / 1000, tz=timezone.utc),
            "fundingRate": float(r["fundingRate"]),
            "premium": float(r["premium"]),
        }
        for r in raw
    ]
    return align_hourly(rows)


# --------------------------------------------------------------------------
# Source B: s3://hyperliquid-archive/asset_ctxs/ (requester-pays)
# --------------------------------------------------------------------------

_CREDENTIALS_HELP = """
No AWS credentials found. The official archive is a REQUESTER-PAYS bucket, so
anonymous access is impossible -- S3 replies:

    AccessDenied: Anonymous users cannot invoke requests against Requester Pays
    buckets. Please authenticate.

You do not need the `aws` CLI. boto3 reads credentials from environment
variables directly:

    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1

or from ~/.aws/credentials, which you can create by hand:

    mkdir -p ~/.aws
    cat > ~/.aws/credentials <<'EOF'
    [default]
    aws_access_key_id = AKIA...
    aws_secret_access_key = ...
    EOF
    chmod 600 ~/.aws/credentials

The bucket is requester-pays: it needs an AWS account and an access key.

BEFORE YOU DO ANY OF THAT: you almost certainly do not need it. Run

    python3 -m regime_engine.fetch_official --source rest --validate

which uses the free public fundingHistory endpoint and needs no AWS account.
"""


def _require_boto3():
    try:
        import boto3  # type: ignore
        import botocore  # type: ignore
    except ImportError as exc:
        raise RecoveryError(
            "boto3 is not installed (it is listed in requirements.txt but absent "
            "from this interpreter). Install it with:\n\n"
            "    python3 -m pip install --user 'boto3>=1.34' 'lz4>=4.3'\n\n"
            "Or avoid it entirely with --source rest, which uses only the "
            "standard library."
        ) from exc
    return boto3, botocore


def _s3_client():
    boto3, botocore = _require_boto3()
    session = boto3.session.Session()
    if session.get_credentials() is None:
        raise RecoveryError(_CREDENTIALS_HELP.strip())
    return session.client("s3", region_name=BUCKET_REGION), botocore


def s3_list_available(
    client, botocore, start: datetime | None = None, end: datetime | None = None
) -> tuple[list[tuple[datetime, str, int]], int]:
    """List asset_ctxs objects, optionally restricted to [start, end].

    Returns (sorted [(date, key, size)], number_of_LIST_calls). Also used by
    --dry-run to answer the retention question directly: the min and max dates
    returned ARE the archive's true coverage.
    """
    found: list[tuple[datetime, str, int]] = []
    calls = 0
    token = None
    while True:
        kwargs = {
            "Bucket": BUCKET,
            "Prefix": ASSET_CTXS_PREFIX,
            "RequestPayer": "requester",
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token
        try:
            resp = client.list_objects_v2(**kwargs)
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise RecoveryError(
                    f"S3 refused the request ({code}). Your credentials are present but "
                    "not accepted, or your IAM user lacks s3:ListBucket/s3:GetObject.\n"
                    "Attach the AWS managed policy AmazonS3ReadOnlyAccess to the IAM "
                    f"user, then retry.\n\nUnderlying error: {exc}"
                ) from exc
            raise RecoveryError(f"S3 list_objects_v2 failed: {exc}") from exc
        calls += 1
        for obj in resp.get("Contents", []):
            date = key_to_date(obj["Key"])
            if date is None:
                continue
            if start and date < start.replace(hour=0, minute=0, second=0, microsecond=0):
                continue
            if end and date > end.replace(hour=0, minute=0, second=0, microsecond=0):
                continue
            found.append((date, obj["Key"], int(obj["Size"])))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    found.sort()
    return found, calls


def s3_fetch_day(client, botocore, key: str, cache_dir: pathlib.Path) -> bytes:
    """Download one asset_ctxs object, caching immutably. Never re-downloads."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / key.rsplit("/", 1)[-1]
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_bytes()
    try:
        resp = client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
        blob = resp["Body"].read()
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            raise RecoveryError(
                f"The archive has no object for {key}. That date is not covered.\n"
                "Run --dry-run to see the archive's true coverage window."
            ) from exc
        raise RecoveryError(f"S3 get_object failed for {key}: {exc}") from exc
    tmp = cached.with_suffix(cached.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(cached)  # atomic; cache entries are immutable once present
    return blob


def fetch_s3(
    coin: str, start: datetime, end: datetime, verbose: bool = True
) -> list[dict[str, object]]:
    client, botocore = _s3_client()
    keys = day_keys(start, end)
    rows: list[dict[str, object]] = []
    for i, key in enumerate(keys, 1):
        blob = s3_fetch_day(client, botocore, key, RAW_DIR)
        rows.extend(parse_asset_ctxs_csv(lz4_decompress(blob), coin))
        if verbose and (i % 10 == 0 or i == len(keys)):
            print(f"  {i}/{len(keys)} days, {len(rows)} {coin} rows")
    return align_hourly(rows)


# --------------------------------------------------------------------------
# Panel comparison (the acceptance test)
# --------------------------------------------------------------------------


def load_panel(path: pathlib.Path = PANEL_PATH) -> list[dict[str, object]]:
    """Load the shipped panel's [time, fundingRate, premium] columns."""
    if not path.exists():
        raise RecoveryError(f"Reference panel not found at {path}")
    rows = []
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            rows.append(
                {
                    "time": _parse_timestamp(rec["time"]).replace(
                        minute=0, second=0, microsecond=0
                    ),
                    "fundingRate": float(rec["fundingRate"]),
                    "premium": float(rec["premium"]),
                }
            )
    return rows


def compare_on_overlap(
    fetched: list[dict[str, object]], panel: list[dict[str, object]]
) -> dict[str, object]:
    """Max absolute difference in premium and fundingRate on shared hours."""
    by_time = {r["time"]: r for r in fetched}
    n = 0
    max_dp = 0.0
    max_df = 0.0
    worst: datetime | None = None
    exact = 0
    for prow in panel:
        frow = by_time.get(prow["time"])
        if frow is None:
            continue
        n += 1
        dp = abs(float(frow["premium"]) - float(prow["premium"]))
        df = abs(float(frow["fundingRate"]) - float(prow["fundingRate"]))
        if float(frow["premium"]) == float(prow["premium"]):
            exact += 1
        if dp > max_dp:
            max_dp, worst = dp, prow["time"]  # type: ignore[assignment]
        max_df = max(max_df, df)
    return {
        "panel_rows": len(panel),
        "fetched_rows": len(fetched),
        "overlap_rows": n,
        "max_abs_premium_diff": max_dp,
        "max_abs_premium_diff_bps": max_dp * 1e4,
        "max_abs_funding_diff": max_df,
        "exact_match_fraction": (exact / n) if n else 0.0,
        "worst_hour": worst,
    }


def write_panel_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PANEL_COLUMNS)
        for r in rows:
            w.writerow(
                [
                    "{:%Y-%m-%d %H:%M:%S}+00:00".format(r["time"]),
                    repr(float(r["fundingRate"])),
                    repr(float(r["premium"])),
                ]
            )


def summarise(rows: list[dict[str, object]], label: str) -> None:
    if not rows:
        print(f"  {label}: NO ROWS")
        return
    prem = [float(r["premium"]) for r in rows]
    hi = max(range(len(prem)), key=lambda i: prem[i])
    lo = min(range(len(prem)), key=lambda i: prem[i])
    gaps = missing_hours(rows)
    print(f"  {label}")
    print(f"    rows          : {len(rows)}")
    print(
        "    span          : {:%Y-%m-%d %H:%M} -> {:%Y-%m-%d %H:%M} UTC".format(
            rows[0]["time"], rows[-1]["time"]
        )
    )
    print(f"    missing hours : {len(gaps)}")
    print(
        "    premium max   : {:+.4f} bps at {:%Y-%m-%d %H:%M}".format(
            prem[hi] * 1e4, rows[hi]["time"]
        )
    )
    print(
        "    premium min   : {:+.4f} bps at {:%Y-%m-%d %H:%M}".format(
            prem[lo] * 1e4, rows[lo]["time"]
        )
    )


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_dry_run(args, start: datetime, end: datetime) -> int:
    print("=" * 74)
    print("DRY RUN -- nothing will be downloaded")
    print("=" * 74)
    print(f"coin   : {args.coin}")
    print(f"range  : {start:%Y-%m-%d} .. {end:%Y-%m-%d} (UTC, inclusive)")
    print(f"source : {args.source}")
    print()

    if args.source == "rest":
        print(f"Source: {REST_URL}  (fundingHistory)")
        print("  auth        : none required")
        print("  cost        : $0.00 (free public endpoint)")
        print(f"  page limit  : {REST_PAGE_LIMIT} records/call")
        hours = int((end - start).total_seconds() // 3600) + 24
        print(f"  hours wanted: ~{hours}")
        print(f"  calls needed: ~{-(-hours // REST_PAGE_LIMIT)}")
        print()
        print("Probing the endpoint for the FIRST hour of the requested range...")
        probe = _rest_post(
            {
                "type": "fundingHistory",
                "coin": args.coin,
                "startTime": to_ms(start),
                "endTime": to_ms(start + timedelta(hours=6)),
            }
        )
        if isinstance(probe, list) and probe:
            first = probe[0]
            print(
                "  OK: {} records. First = {:%Y-%m-%d %H:%M} UTC, premium "
                "{:+.4f} bps".format(
                    len(probe),
                    datetime.fromtimestamp(int(first["time"]) / 1000, tz=timezone.utc),
                    float(first["premium"]) * 1e4,
                )
            )
            print("  RETENTION: the requested start IS served. Proceed.")
        else:
            print("  EMPTY -- the endpoint returned no records for that start.")
            print("  Either the coin is wrong or the range predates the market.")
            return 1
        print()
        print("Next step:")
        print("  python3 -m regime_engine.fetch_official --source rest --validate")
        return 0

    # --- S3 ---
    print(f"Source: s3://{BUCKET}/{ASSET_CTXS_PREFIX}  (region {BUCKET_REGION}, requester-pays)")
    client, botocore = _s3_client()
    print("  credentials : OK")
    print()
    print("Listing archive coverage (this is the retention answer)...")
    all_objs, list_calls = s3_list_available(client, botocore)
    if not all_objs:
        print("  The asset_ctxs/ prefix is EMPTY or unreadable.")
        return 1
    print(f"  objects available : {len(all_objs)}")
    print(f"  earliest date     : {all_objs[0][0]:%Y-%m-%d}")
    print(f"  latest date       : {all_objs[-1][0]:%Y-%m-%d}")
    print()

    wanted = day_keys(start, end)
    have = {k: s for _, k, s in all_objs}
    present = [k for k in wanted if k in have]
    absent = [k for k in wanted if k not in have]
    total_bytes = sum(have[k] for k in present)

    print("Requested range:")
    print(f"  days requested : {len(wanted)}")
    print(f"  days present   : {len(present)}")
    print(f"  days MISSING   : {len(absent)}")
    if absent:
        show = [k.rsplit("/", 1)[-1] for k in absent[:10]]
        ellipsis = " ..." if len(absent) > 10 else ""
        print(f"    e.g. {', '.join(show)}{ellipsis}")
        print()
        print("  *** THE ARCHIVE DOES NOT FULLY COVER THE REQUESTED WINDOW. ***")
        print("  Use --source rest instead: fundingHistory serves the whole window.")
    print()

    cost = estimate_cost(len(present), total_bytes, n_list_calls=list_calls)
    print("Cost estimate (AWS us-east-1 list price):")
    print("  bytes to transfer : {:,} ({:.2f} MiB)".format(
        int(cost["total_bytes"]), cost["total_mib"]))
    print(f"  GET requests      : {len(present)}  @ ${USD_PER_GET:.7f} each")
    print(f"  LIST requests     : {list_calls}  @ ${USD_PER_LIST:.6f} each")
    print("  request charge    : ${:.6f}".format(cost["request_usd"]))
    print("  egress @ $0.09/GB : ${:.6f}".format(cost["egress_full_rate_usd"]))
    print("  egress w/ free    : ${:.6f}  (first {:.0f} GB/month free)".format(
        cost["egress_usd"], FREE_EGRESS_GB_PER_MONTH))
    print("  ------------------------------------")
    print("  TOTAL (free tier) : ${:.6f}".format(cost["total_usd"]))
    print("  TOTAL (full rate) : ${:.6f}".format(cost["total_usd_no_free_tier"]))
    print()
    print(f"NOTE: these files contain EVERY perp, not just {args.coin}. The byte count")
    print(f"above is the true transfer; {args.coin} is filtered after decompression.")
    return 0 if not absent else 1


def run_validate(args) -> int:
    panel = load_panel()
    p_start = min(r["time"] for r in panel)
    p_end = max(r["time"] for r in panel)
    print("=" * 74)
    print("VALIDATE -- acceptance test against data/processed/panel_BTC_subsample.csv")
    print("=" * 74)
    print(f"panel span: {p_start:%Y-%m-%d %H:%M} -> {p_end:%Y-%m-%d %H:%M} UTC ({len(panel)} rows)")
    print(f"fetching the same span from source={args.source} ...")
    fetch = fetch_rest if args.source == "rest" else fetch_s3
    fetched = fetch(args.coin, p_start, p_end + timedelta(hours=1))
    print()
    res = compare_on_overlap(fetched, panel)
    print("RESULT")
    print("  overlap rows           : {} / {}".format(res["overlap_rows"], res["panel_rows"]))
    print("  max |d premium|        : {:.12g}".format(res["max_abs_premium_diff"]))
    print("  max |d premium| in bps : {:.12g}".format(res["max_abs_premium_diff_bps"]))
    print("  max |d fundingRate|    : {:.12g}".format(res["max_abs_funding_diff"]))
    print("  exact-match fraction   : {:.6f}".format(res["exact_match_fraction"]))
    if res["worst_hour"] is not None:
        print("  worst hour             : {:%Y-%m-%d %H:%M} UTC".format(res["worst_hour"]))
    print()
    if res["overlap_rows"] == 0:
        print("FAIL: no overlapping hours. The source is not serving this window.")
        return 1
    if res["max_abs_premium_diff"] <= args.tolerance:
        print(f"PASS: premium agrees within tolerance ({args.tolerance:g}).")
        print("      The two sources describe the same object; the recovered")
        print("      Oct-Nov 2025 rows are usable downstream.")
        return 0
    print(f"FAIL: premium differs by more than the tolerance ({args.tolerance:g}).")
    print("      DO NOT use the recovered segment. The archive and the shipped")
    print("      panel are not the same series; resolve this first.")
    return 1


def run_fetch(args, start: datetime, end: datetime) -> int:
    print("=" * 74)
    print(f"FETCH  coin={args.coin}  {start:%Y-%m-%d} .. {end:%Y-%m-%d}  source={args.source}")
    print("=" * 74)
    fetch = fetch_rest if args.source == "rest" else fetch_s3
    rows = fetch(args.coin, start, end + timedelta(days=1))
    rows = [r for r in rows if start <= r["time"] < end + timedelta(days=1)]
    print()
    summarise(rows, "recovered series")
    if not rows:
        return 1
    out = args.out and pathlib.Path(args.out) or (
        INTERIM_DIR
        / f"official_{args.coin}_{start:%Y%m%d}_{end:%Y%m%d}_{args.source}.csv"
    )
    write_panel_csv(to_panel_rows(rows), out)
    print()
    print(f"wrote {out}")
    print("columns: {} (premium is a raw decimal; x1e4 for bps)".format(
        ",".join(PANEL_COLUMNS)))
    gaps = missing_hours(rows)
    if gaps:
        print()
        print(f"WARNING: {len(gaps)} missing hours in the recovered range, e.g.")
        for g in gaps[:5]:
            print(f"  {g:%Y-%m-%d %H:%M} UTC")
        return 1
    return 0


# --------------------------------------------------------------------------
# Offline self-test (no network, no credentials, no AWS)
# --------------------------------------------------------------------------

_FIXTURE_HEADER = (
    "time,coin,funding,open_interest,prev_day_px,day_ntl_vlm,"
    "premium,oracle_px,mark_px,mid_px,impact_bid_px,impact_ask_px"
)


def _build_fixture_csv() -> bytes:
    """Synthetic asset_ctxs CSV: 3 hours x 2 coins, deliberately shuffled.

    Row order is scrambled and one hour is duplicated with a sub-hour offset,
    so alignment (flooring + dedupe + sort) is actually exercised.
    """
    lines = [_FIXTURE_HEADER]
    recs = [
        # (epoch_ms, coin, funding, premium, oracle, mark)
        (1759366800056, "ETH", 0.0000090, -0.0001111, 4100.0, 4100.5),
        (1759370400057, "BTC", 0.0000125, -0.0002784901, 120100.0, 120105.0),
        (1759363200016, "BTC", 0.0000125, -0.0002119458, 120000.0, 120010.0),
        (1759366800056, "BTC", 0.0000125, -0.0003266543, 120050.0, 120055.0),
        (1759366802999, "BTC", 0.0000125, -0.0009999999, 120051.0, 120056.0),  # dup hour
        (1759363200016, "ETH", 0.0000090, -0.0002222, 4090.0, 4090.5),
    ]
    # Filler values for the columns the parser must ignore: open_interest,
    # prev_day_px, day_ntl_vlm. Their exact values are irrelevant -- the point
    # of the fixture is that resolution happens by NAME, not by position.
    oi, prev_px, vlm = 1234.5, 119000.0, 9.87e8
    for ts, coin, funding, premium, oracle, mark in recs:
        lines.append(
            f"{ts},{coin},{funding},{oi},{prev_px},{vlm},"
            f"{premium},{oracle},{mark},{mark},{mark - 1},{mark + 1}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}  {detail}")
            failures.append(name)

    print("=" * 74)
    print("SELF-TEST -- offline, no network, no credentials")
    print("=" * 74)

    # --- xxHash32 against the reference vectors from the xxHash spec ---
    print("\n[1] xxHash32 (needed for LZ4 frame headers)")
    check("xxh32(b'') == 0x02CC5D05", xxh32(b"") == 0x02CC5D05, hex(xxh32(b"")))
    check("xxh32(b'a') == 0x550D7456", xxh32(b"a") == 0x550D7456, hex(xxh32(b"a")))
    check(
        "xxh32(b'abc') == 0x32D153FF", xxh32(b"abc") == 0x32D153FF, hex(xxh32(b"abc"))
    )
    long_in = b"0123456789abcdefghijklmnopqrstuvwxyz" * 4
    check(
        "xxh32 long input is deterministic and 32-bit",
        0 <= xxh32(long_in) <= 0xFFFFFFFF and xxh32(long_in) == xxh32(long_in),
    )

    # --- LZ4 frame round-trip ---
    print("\n[2] LZ4 frame fixture")
    payload = _build_fixture_csv()
    frame = build_lz4_frame(payload, block_size=64)  # tiny blocks -> multi-block
    check("frame starts with the LZ4 magic number", frame.startswith(_LZ4_MAGIC))
    # 7-byte header + per-block (4-byte size + <=64 payload) + 4-byte EndMark
    n_blocks = -(-len(payload) // 64)
    check(
        f"frame is genuinely multi-block ({n_blocks} blocks of 64B)",
        n_blocks > 1 and len(frame) == 7 + 4 * n_blocks + len(payload) + 4,
        f"len={len(frame)} expected={7 + 4 * n_blocks + len(payload) + 4}",
    )
    check("pure-python reader round-trips", _decompress_stored_frame(frame) == payload)
    try:
        import lz4.frame  # type: ignore

        check(
            "python-lz4 accepts our frame (proves it is spec-valid)",
            lz4.frame.decompress(frame) == payload,
        )
        real = lz4.frame.compress(payload)
        check(
            "lz4_decompress() handles a REAL compressed frame",
            lz4_decompress(real) == payload,
        )
    except ImportError:
        print("  SKIP  python-lz4 cross-check (lz4 not installed)")
    check("lz4_decompress() handles our fixture", lz4_decompress(frame) == payload)

    # --- Column resolution ---
    print("\n[3] asset_ctxs schema mapping")
    idx = _resolve_columns(_FIXTURE_HEADER.split(","))
    check("resolves canonical snake_case header", idx["premium"] == 6 and idx["coin"] == 1)
    camel = ["time", "coin", "fundingRate", "openInterest", "premium", "markPx"]
    idx2 = _resolve_columns(camel)
    check("resolves camelCase header by NAME not position",
          idx2["premium"] == 4 and idx2["funding"] == 2)
    shuffled = ["premium", "coin", "funding", "time"]
    idx3 = _resolve_columns(shuffled)
    check("column ORDER is irrelevant",
          idx3["premium"] == 0 and idx3["time"] == 3 and idx3["coin"] == 1)
    try:
        _resolve_columns(["time", "coin", "open_interest"])
        check("missing required columns raises", False, "no exception")
    except RecoveryError as exc:
        check("missing required columns raises an actionable error",
              "premium" in str(exc) and "--source rest" in str(exc))

    # --- Parsing ---
    print("\n[4] CSV parsing")
    rows = parse_asset_ctxs_csv(lz4_decompress(frame), "BTC")
    check("filters to the requested coin (4 BTC rows, ETH dropped)", len(rows) == 4,
          f"got {len(rows)}")
    check("premium parsed as a raw decimal", any(
        abs(r["premium"] - (-0.0002119458)) < 1e-15 for r in rows))
    check("fundingRate parsed as the hourly rate", all(
        abs(r["fundingRate"] - 1.25e-05) < 1e-15 for r in rows))
    check("optional price columns captured", all("mark_px" in r for r in rows))
    check("timestamps are tz-aware UTC", all(
        r["time"].tzinfo is not None and r["time"].utcoffset() == timedelta(0)
        for r in rows))
    eth = parse_asset_ctxs_csv(lz4_decompress(frame), "eth")  # case-insensitive
    check("coin filter is case-insensitive", len(eth) == 2, f"got {len(eth)}")

    # --- Timestamp conventions ---
    print("\n[5] timestamp conventions")
    check("epoch ms", _parse_timestamp("1759363200016")
          == datetime(2025, 10, 2, 0, 0, 0, 16000, tzinfo=timezone.utc))
    check("epoch s", _parse_timestamp("1759363200")
          == datetime(2025, 10, 2, tzinfo=timezone.utc))
    check("ISO with Z", _parse_timestamp("2025-10-02T00:00:00Z")
          == datetime(2025, 10, 2, tzinfo=timezone.utc))
    check("ISO with a space and no zone (assumed UTC)",
          _parse_timestamp("2025-10-02 05:00:00")
          == datetime(2025, 10, 2, 5, tzinfo=timezone.utc))
    check("ISO with an explicit offset",
          _parse_timestamp("2025-10-02 05:00:00+00:00")
          == datetime(2025, 10, 2, 5, tzinfo=timezone.utc))

    # --- Hourly alignment ---
    print("\n[6] hourly alignment")
    aligned = align_hourly(rows)
    check("duplicate hour collapsed (4 raw -> 3 hourly)", len(aligned) == 3,
          f"got {len(aligned)}")
    check("sorted ascending", all(
        aligned[i]["time"] < aligned[i + 1]["time"] for i in range(len(aligned) - 1)))
    check("all timestamps floored to the hour", all(
        r["time"].minute == 0 and r["time"].second == 0 and r["time"].microsecond == 0
        for r in aligned))
    check("dedupe keeps the FIRST record of the hour, matching legacy .drop_duplicates",
          abs(aligned[1]["premium"] - (-0.0003266543)) < 1e-15,
          "got {}".format(aligned[1]["premium"]))
    check("no missing hours in a contiguous run", missing_hours(aligned) == [])
    gappy = [aligned[0], aligned[2]]
    check("missing_hours detects a one-hour gap", len(missing_hours(gappy)) == 1)

    # --- Panel schema ---
    print("\n[7] panel schema mapping")
    panel_rows = to_panel_rows(aligned)
    check("exactly the panel columns", all(
        sorted(r.keys()) == sorted(PANEL_COLUMNS) for r in panel_rows))
    if PANEL_PATH.exists():
        with PANEL_PATH.open(newline="") as fh:
            real_header = next(csv.reader(fh))
        check("panel columns are a subset of the shipped panel's header",
              all(c in real_header for c in PANEL_COLUMNS),
              f"shipped header = {real_header}")
        real = load_panel()
        check("shipped panel loads", len(real) > 3000, f"got {len(real)}")
        check("shipped premium is a raw decimal (|.| < 0.01)",
              all(abs(float(r["premium"])) < 0.01 for r in real))
        ident = compare_on_overlap(real, real)
        check("compare_on_overlap is 0.0 against itself",
              ident["max_abs_premium_diff"] == 0.0
              and ident["overlap_rows"] == len(real))
        shifted = [dict(r, premium=float(r["premium"]) + 1e-6) for r in real]
        got = compare_on_overlap(shifted, real)["max_abs_premium_diff"]
        check("compare_on_overlap detects a 1e-6 perturbation",
              abs(got - 1e-6) < 1e-15, f"got {got}")
    else:
        print("  SKIP  shipped-panel checks (panel not found)")

    # --- Round-trip through the CSV writer ---
    print("\n[8] CSV writer round-trip")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "rt.csv"
        write_panel_csv(panel_rows, tmp)
        back = load_panel(tmp)
        check("row count preserved", len(back) == len(panel_rows))
        check("premium preserved bit-exactly", all(
            float(a["premium"]) == float(b["premium"])
            for a, b in zip(back, panel_rows)))
        check("timestamps preserved", all(
            a["time"] == b["time"] for a, b in zip(back, panel_rows)))

    # --- Key generation + cost model ---
    print("\n[9] S3 key generation and cost model")
    keys = day_keys(parse_day("2025-10-02"), parse_day("2025-11-30"))
    check("Oct 2 - Nov 30 is 60 daily keys", len(keys) == 60, f"got {len(keys)}")
    check("first key", keys[0] == "asset_ctxs/20251002.csv.lz4", keys[0])
    check("last key", keys[-1] == "asset_ctxs/20251130.csv.lz4", keys[-1])
    check("single-day range yields one key",
          len(day_keys(parse_day("2025-10-02"), parse_day("2025-10-02"))) == 1)
    check("key_to_date round-trips",
          key_to_date(keys[0]) == parse_day("2025-10-02"))
    check("key_to_date rejects non-matching keys",
          key_to_date("asset_ctxs/README.md") is None)
    try:
        day_keys(parse_day("2025-11-30"), parse_day("2025-10-02"))
        check("reversed range raises", False, "no exception")
    except RecoveryError:
        check("reversed range raises", True)
    c = estimate_cost(60, 30 * 1024 * 1024, n_list_calls=2)
    check("60 GETs + 2 LISTs cost ~$0.000034",
          abs(c["request_usd"] - (60 * USD_PER_GET + 2 * USD_PER_LIST)) < 1e-12)
    check("30 MiB is free under the 100 GB/month allowance", c["egress_usd"] == 0.0)
    check("30 MiB at full rate is ~$0.0026",
          abs(c["egress_full_rate_usd"] - 0.0026) < 0.0005,
          "{}".format(c["egress_full_rate_usd"]))
    check("total under one US cent either way", c["total_usd_no_free_tier"] < 0.01)

    print("\n" + "=" * 74)
    if failures:
        print("SELF-TEST FAILED: {} failing check(s): {}".format(
            len(failures), ", ".join(failures)))
        return 1
    print("SELF-TEST PASSED -- all offline checks green")
    print("=" * 74)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python3 -m regime_engine.fetch_official",
        description="Recover BTC funding premium from fundingHistory (free) or "
                    "s3://hyperliquid-archive/asset_ctxs (requester-pays).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # 1. check access and see what would be fetched (no downloads)
  python3 -m regime_engine.fetch_official --dry-run --start 2025-10-02 --end 2025-11-30

  # 2. acceptance test against the shipped panel
  python3 -m regime_engine.fetch_official --validate

  # 3. recover the missing segment
  python3 -m regime_engine.fetch_official --start 2025-10-02 --end 2025-11-30

  # 4. offline unit tests (no network, no AWS)
  python3 -m regime_engine.fetch_official --self-test
""",
    )
    ap.add_argument("--start", help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", help="last UTC day, YYYY-MM-DD (inclusive)")
    ap.add_argument("--coin", default="BTC")
    ap.add_argument(
        "--source",
        choices=("rest", "s3"),
        default="rest",
        help="rest = free fundingHistory endpoint (default); "
             "s3 = hyperliquid-archive asset_ctxs, requester-pays",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="verify access, list what would be fetched, estimate cost")
    ap.add_argument("--validate", action="store_true",
                    help="acceptance test: compare premium against the shipped panel")
    ap.add_argument("--self-test", action="store_true",
                    help="offline unit tests against a synthetic lz4 CSV fixture")
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="max |d premium| accepted by --validate (default 1e-9)")
    ap.add_argument("--out", help="output CSV path (default: data/interim/...)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            return run_self_test()
        if args.validate:
            return run_validate(args)
        if not args.start or not args.end:
            raise RecoveryError(
                "--start and --end are required for --dry-run and for fetching.\n"
                "The missing segment is:  --start 2025-10-02 --end 2025-11-30"
            )
        start, end = parse_day(args.start), parse_day(args.end)
        if args.dry_run:
            return run_dry_run(args, start, end)
        return run_fetch(args, start, end)
    except RecoveryError as exc:
        sys.stdout.flush()  # keep the banner above the error when piped
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
