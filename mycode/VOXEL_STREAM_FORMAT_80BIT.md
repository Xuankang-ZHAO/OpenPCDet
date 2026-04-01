# 80-bit Packed Voxel Stream File Format Specification

This document defines the **cycle-accurate stimulus file format** for driving `voxel_data_in[79:0]` in the RTL module `block_wise_mem_ctrl`.

The primary goal is to make it **easy and unambiguous** for:
- algorithm engineers (Python/C++ data generation) to export voxel streams, and
- verification engineers (SystemVerilog testbench) to read and replay the stream with `voxel_valid/voxel_ready` handshake.

---

## 1. Signal / Word Definition

- **Signal name (RTL):** `voxel_data_in`
- **Width:** 80 bits
- **One word = one voxel** payload
- **Packing convention (exactly as RTL):**

> `voxel_data_in = { feat[31:0], z[15:0], y[15:0], x[15:0] }`

Equivalently, by bit ranges:

| Field | Bits (inclusive) | Width | Type | Notes |
|------:|------------------:|------:|------|------|
| `x`   | `[15:0]`          | 16    | unsigned | voxel x coordinate |
| `y`   | `[31:16]`         | 16    | unsigned | voxel y coordinate |
| `z`   | `[47:32]`         | 16    | unsigned | voxel z coordinate |
| `feat`| `[79:48]`         | 32    | unsigned | feature payload |

### 1.1 Range constraints (recommended)

To avoid truncation/overflow when packing:
- `0 <= x,y,z <= 65535` (16-bit)
- `0 <= feat <= 2^32-1` (32-bit)

If your upstream algorithm uses signed coordinates or larger ranges, you must define and document the mapping before packing (e.g., bias/offset, clamp, or modulo). The RTL interprets these fields as **raw bit-vectors**.

---

## 2. Canonical File Representation (Recommended)

### 2.1 One voxel per line, 80-bit hex token

**Each line contains exactly one 80-bit word** written as a 20-hex-digit token (MSB-first):

- Hex digit 0 (leftmost) corresponds to bits `[79:76]`
- Hex digit 19 (rightmost) corresponds to bits `[3:0]`

**Canonical form (recommended):**

```
<20 hex digits, lowercase or uppercase>
```

Examples:

- `AABBCCDD000300020001`  
  Means:
  - `feat = 0xAABBCCDD`
  - `z    = 0x0003`
  - `y    = 0x0002`
  - `x    = 0x0001`

- `00000000000000000000`  (all zeros)

### 2.2 Allowed whitespace and comments (testbench-friendly)

To keep the testbench simple and robust, the following is recommended:
- Leading/trailing whitespace **may** be present and should be ignored.
- Blank lines **may** be present and should be skipped.
- Comment lines (optional): lines beginning with `#` are comments and should be skipped.

Example:

```
# voxel stream for frame 0
AABBCCDD000300020001
00000000000000000000

# end
```

### 2.3 NOT recommended (avoid unless you update the TB accordingly)

- Using separators inside the hex token (e.g., `AABB_CCDD_0003_0002_0001`)
- Using `0x` prefix
- CSV with multiple columns

These can be supported, but they complicate `$fscanf` patterns and error handling.

---

## 3. Endianness / Bit Ordering Clarification

There are two distinct concepts:

1) **Bit ordering inside the 80-bit word** (this spec):
- `x` occupies bits `[15:0]` (LSB side)
- `feat` occupies bits `[79:48]` (MSB side)

2) **Textual hex ordering in the file**:
- The hex token is written **MSB-first** (standard human-readable hex)
- Therefore the file token is:

```
<feat:8 hex><z:4 hex><y:4 hex><x:4 hex>
```

This is intentionally chosen so a SystemVerilog testbench can do:

```systemverilog
logic [79:0] word;
$fscanf(fd, "%h", word);
```

and directly drive:

```systemverilog
voxel_data_in = word;
```

---

## 4. Python Packing / Unpacking Reference

### 4.1 Pack `(x, y, z, feat)` into 80-bit word

```python
# word[79:0] = {feat[31:0], z[15:0], y[15:0], x[15:0]}

def pack_voxel_word(x: int, y: int, z: int, feat: int) -> int:
    assert 0 <= x < (1 << 16)
    assert 0 <= y < (1 << 16)
    assert 0 <= z < (1 << 16)
    assert 0 <= feat < (1 << 32)

    word = (feat << 48) | (z << 32) | (y << 16) | x
    # word fits in 80 bits
    assert 0 <= word < (1 << 80)
    return word


def word_to_hex80(word: int) -> str:
    # 80 bits = 20 hex digits, MSB-first
    return f"{word:020X}"
```

### 4.2 Unpack 80-bit word into fields

```python
def unpack_voxel_word(word: int) -> tuple[int, int, int, int]:
    x = (word >> 0) & 0xFFFF
    y = (word >> 16) & 0xFFFF
    z = (word >> 32) & 0xFFFF
    feat = (word >> 48) & 0xFFFFFFFF
    return x, y, z, feat
```

### 4.3 Generate a stream file

```python
from pathlib import Path


def write_voxel_stream_hex80(path: str | Path, voxels):
    """voxels: iterable of (x, y, z, feat). Writes one 80-bit hex token per line."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for (x, y, z, feat) in voxels:
            word = pack_voxel_word(x, y, z, feat)
            f.write(word_to_hex80(word) + "\n")
```

---

## 5. Testbench Consumption Guidance (for SV engineers)

### 5.1 Handshake requirement

The DUT uses `voxel_valid/voxel_ready` handshake; a voxel word is accepted only when:

- at a rising clock edge: `voxel_valid == 1` and `voxel_ready == 1`

Recommended stimulus policy:
- read the next 80-bit token from file,
- wait until `voxel_ready` is high,
- drive `voxel_valid=1` for **exactly one cycle** with `voxel_data_in` stable,
- then drop `voxel_valid` and proceed to the next token.

### 5.2 Trace correlation (recommended)

To correlate DRAM writes back to voxel indices, keep a counter `voxel_accept_idx` incremented on each accepted voxel handshake.

When dumping DRAM traces on each `dram_req`, optionally include:
- simulation cycle index
- `voxel_accept_idx` (most recent accepted voxel)
- `dram_addr`, `dram_wdata`

---

## 6. Versioning / Compatibility

- This spec matches the current RTL packing documented in `block_wise_mem_ctrl`.
- If `INPUT_BUS_WIDTH`, field widths, or packing order changes in RTL, this document must be updated **together** with the generator and testbench.
