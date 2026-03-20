# ===== Rui'an dialect dictionary checker (Enhanced) =====

import re

# ===== 1. 聲母 =====

initials = [
    "",

    "b","p","bb","m","f","v",
    "d","t","dd","n","l",
    "g","k","gg","ng","h","hh",

    # 舌面
    "j","q","jj","nj","x","w",

    # 齒齦
    "z","c","zz","s","zs",

    # 精確擴展（可選）
    "zh","ch","sh",

    "y"
]

# ===== 2. 韻母 =====

finals = [
    # 單
    "a","o","oo","ae","e","i","u","yu",

    # 舌尖元音
    "ii",

    # 複
    "ai","ei","ao","ou","eu",
    "iae","io","ia","iao","iou","ie",
    "ua","uai","uo","uoo",
    "yue","yo",

    # 鼻
    "ang","eng","ong","iang","iong",

    # 特殊
    "ng"
]

# ===== 3. 聲調 =====

tones = ["1","2","3","4","5","6","7","8"]

# ===== 4. 音系約束（核心） =====

def is_valid(initial, final):
    # --- ng 韻 ---
    if final == "ng":
        return initial == ""

    # --- 舌尖元音 ii ---
    if final == "ii":
        return initial in ["z","c","s","zs","zz"]

    # --- 舌面音 ---
    palatal = ["j","q","x","jj","nj"]

    if initial in palatal:
        if not (final.startswith("i") or final.startswith("y")):
            return False

    # --- u 系限制 ---
    if final.startswith("u"):
        if initial in ["j","q","x"]:
            return False

    # --- yu 限制 ---
    if final.startswith("yu"):
        if initial in ["b","p","m","f"]:
            return False

    # --- 鼻韻母細化 ---
    if final in ["iang","iong"]:
        if initial not in palatal:
            return False

    # --- ng 作聲母 ---
    if initial == "ng":
        if final not in ["a","o","oo","e","ae","u"]:
            return False

    # --- h / hh ---
    if initial in ["h","hh"] and final == "ii":
        return False

    return True


# ===== 5. 生成合法音節 =====

valid_syllables = set()
invalid_pairs = []

for i in initials:
    for f in finals:
        if not is_valid(i, f):
            invalid_pairs.append((i, f))
            continue
        for t in tones:
            valid_syllables.add(i + f + t)

print(f"✅ Generated {len(valid_syllables)} valid syllables")
print(f"🚫 Filtered out {len(invalid_pairs)} illegal initial-final pairs\n")


# ===== 6. normalize =====

NORMALIZE = {
    "xy": "zs",

    # 常見錯誤
    "ju": "jyu",
    "qu": "qyu",
    "xu": "xyu",
}

def normalize(p):
    for k, v in NORMALIZE.items():
        p = p.replace(k, v)
    return p


# ===== 7. suspicious 檢測 =====

def is_suspicious(p):
    # ng 出現在中間
    if "ng" in p and not p.startswith("ng"):
        return True

    # ii 不合法位置
    if "ii" in p and not any(p.startswith(x) for x in ["z","c","s","zs","zz"]):
        return True

    # v 濫用
    if "v" in p and not p.startswith(("v","f","hh")):
        return True

    return False


# ===== 8. 檢查 dict =====

def check_dict(file_path):
    errors = []
    suspicious = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            raw = line.strip()

            if not raw or raw.startswith("#") or raw.startswith("---"):
                continue

            parts = re.split(r"\s+", raw)

            if len(parts) < 2:
                continue

            char = parts[0]
            pinyin = parts[1]

            pinyin_norm = normalize(pinyin)

            # ❌ 非法
            if pinyin_norm not in valid_syllables:
                errors.append((line_num, char, pinyin))

            # ⚠️ 可疑
            if is_suspicious(pinyin):
                suspicious.append((line_num, char, pinyin))

    return errors, suspicious


# ===== 9. 執行 =====

FILE = "ruianese.character_01.dict.yaml"  # ← 改這裡

errors, suspicious = check_dict(FILE)

# ===== 10. 輸出 =====
print("❌ 非法拼音（不在音節系統）:")
for e in errors[:50]:
    print(e)

print(f"\nTotal errors: {len(errors)}\n")

print("⚠️ 可疑拼音（建議人工檢查）:")
for s in suspicious[:50]:
    print(s)

print(f"\nTotal suspicious: {len(suspicious)}")


# ===== 11. Debug（可選） =====

print("\n🔍 Sample valid syllables:")
for s in list(valid_syllables)[:20]:
    print(s)