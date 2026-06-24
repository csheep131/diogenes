---
name: remove-emojis
description: Remove emojis, icons, and unicode symbols from source code files with proper artifact cleanup
source: auto-skill
extracted_at: '2026-06-24T10:44:50.480Z'
---

# Remove Emojis and Unicode Icons from Source Code

## When to Use

- User asks to remove emojis, smileys, icons, or decorative unicode symbols from code
- Cleaning up terminal output strings, log messages, or UI labels
- Stripping decorative characters for a cleaner codebase

## Procedure

### Step 1: Scan for Unicode symbols

Use a comprehensive regex pattern covering common emoji/icon ranges. Search with `grep` or a Python script across target directories (`src/`, `scripts/`, `tests/`, etc.):

```python
import re

emoji_pattern = re.compile(
    '['
    '\U0001F680\U00002705\U0000274C\U000026A0\U0000FE0F\U0001F525'  # 🚀✅❌⚠️🔥
    '\U0001F4A1\U0001F3AF\U0001F4CA\U0001F4DD\U0001F527\U0001F6E0'  # 💡🎯📊📝🔧🛠
    '\U00002713\U00002717\U00002718\U000025CB'                       # ✓✗✘○
    '\U00002192\U00002190\U00002191\U00002193'                       # →←↑↓
    '\U0001F916\U000026A1\U0001F504\U0001F4E6\U0001F4DC\U0001F4BE'  # 🤖⚡🔄📦📜💾
    '\U0001F6E1\U00002699\U0001F3CB\U0001F3B2\U0001F6AB'            # 🛡⚙🏋🎲🚫
    # ... extend as needed
    ']'
)
```

**Important**: Exclude files where unicode has semantic meaning:
- Math notation in docstrings (e.g., `α_{i→l}` in attention formulas)
- Training data content (e.g., `"reasoning... → ABSTAIN"` in dataset generators)
- Box-drawing characters used intentionally for CLI formatting (if desired)

### Step 2: Bulk removal

```python
for fpath in files_to_clean:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    cleaned = emoji_pattern.sub('', content)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(cleaned)
```

### Step 3: Artifact cleanup (CRITICAL)

Emoji removal leaves behind artifacts that must be cleaned:

1. **Double/triple spaces** after opening quotes: `"  Warning"` → `"Warning"`
   ```python
   content = re.sub(r'(")  ', r'\1', content)
   content = re.sub(r'(")   ', r'\1', content)
   ```

2. **Empty string values** in dicts/maps where emojis were the only content:
   ```python
   # BEFORE (broken):
   mode_emoji = {
       EpistemicMode.DIRECT_ANSWER.value: "",
       EpistemicMode.ABSTAIN.value: "",
   }
   # AFTER: Remove the entire dict if it only mapped emojis, or replace with text labels
   ```

3. **Tab labels and UI elements** with leading spaces: `" Model"` → `"Model"`

4. **Status variables** that were emoji-only: `status = "" if passed else ""` → `status = "OK" if passed else "FAIL"`

5. **Arrow symbols in log messages**: `"Pass@1 ↓ 0.02"` → `"Pass@1 -0.02"` or use ASCII alternatives

### Step 4: Manual review of special cases

After bulk removal, manually inspect:
- Dict/map values that become empty strings
- f-string format expressions with removed delta symbols (↓↑Δ)
- Status indicator variables
- UI tab/button labels

### Step 5: Verification

Run the same emoji regex scan again to confirm zero matches in cleaned files:

```python
for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            matches = emoji_pattern.findall(line)
            if matches:
                print(f'{fpath}:{i}: found {matches}')
```

## Key Lessons

- **Always do a second pass** for artifacts — bulk regex removal is never clean on first pass
- **Don't touch semantic unicode** — math notation, training data, and intentional formatting should be preserved
- **Empty string values are a common bug** — a dict mapping to `""` after emoji removal is broken
- **Leading spaces in strings** are easy to miss but look wrong in output
