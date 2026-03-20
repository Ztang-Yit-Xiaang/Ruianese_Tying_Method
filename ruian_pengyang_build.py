# ===== 瑞安話音節生成器 v1 =====

initials_origin = [
    "",

    # 唇音
    "p","ph","b","m","f","v",

    # 舌尖
    "t","th","d","n","l",

    # 舌根
    "k","kh","g","ng","h","gh",

    # 舌面
    "tsh","tshh","dzh","ny","sh","zh",

    # 齒齦
    "ts","tshs","dz","s","z"
]

finals_origin = [
    # 單元音
    "a","o","ɔ","æ","ə","i","u","y","ʉ",

    # 複合
    "ai","ei","au","ou",
    "ia","iau","iou","ie","iæ","iɔ",
    "ua","uai","uo","wo",
    "yə","yo",

    # 鼻音
    "aŋ","əŋ","oŋ","iaŋ","ioŋ",

    # 特殊
    "ŋ","ɿ"
]
# 聲母
initials = [
    "",  # 零聲母（非常重要！）
    "b","p","bb","m","f","v",
    "d","t","dd","n","l",
    "g","k","gg","ng","h","hh",
    "j","q","jj","nj","x",
    "z","c","zz","s","zs"
]

# 韻母
finals = [
    # 單韻母
    "a","o","oo","ae","e","i","u","yu",

    # 複韻母
    "ai","ei","ao","ou",
    "iae","io","ia","iao","iou","ie",
    "ua","uai","uo","uoo",
    "yue","yo",

    # 鼻韻母
    "ang","eng","ong","iang","iong",

    # 特殊
    "ng"
]

# 聲調
tones = ["1","2","3","4","5","6","7","8"]

# ===== 規則控制（關鍵🔥） =====

def is_valid(initial, final):
    # ng 作聲母限制
    if initial == "ng" and final not in ["a","o","e","u"]:
        return False

    # ng 作韻母時不能有聲母
    if final == "ng" and initial != "":
        return False

    # 舌尖音限制（簡化版）
    if final == "i" and initial not in ["z","c","s","zs"]:
        pass  # 先不嚴格限制（之後可以加）

    return True


# ===== 生成音節 =====

syllables = []

for i in initials:
    for f in finals:
        if not is_valid(i, f):
            continue
        for t in tones:
            syllables.append(i + f + t)

# ===== 查看結果 =====
print("總音節數：", len(syllables))
print(syllables[:50])