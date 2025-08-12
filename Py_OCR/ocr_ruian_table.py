import cv2, numpy as np, pytesseract, os, pandas as pd
from PIL import Image

# ---- Windows 才需要這行，改成你的安裝路徑 ----
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

IMG_FILES = ["p437.png", "p438.png", "p439.png"]  # 把你的檔名放這裡

def load_and_upright(path):
    img = cv2.imread(path)
    # 自動偵測方向（有時頁面是橫的）
    try:
        osd = pytesseract.image_to_osd(img)
        rot = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                rot = int(line.split(":")[1].strip())
        if rot != 0:
            # 逆時針旋轉還原正向
            rotates = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            img = cv2.rotate(img, rotates[(360-rot) % 360])
    except:
        pass
    return img

def binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 放大提高辨識
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    bw = 255 - bw  # 黑字白底
    return bw

def detect_cells(bw):
    # 取出水平/垂直線以找表格
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60,1)), iterations=1)
    vert  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,60)), iterations=1)
    grid = cv2.add(horiz, vert)

    # 利用輪廓找每個小格
    cnts, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 3000:  # 過小的噪點略過
            continue
        # 過細長的線也略過
        if w < 15 or h < 15:
            continue
        boxes.append((x,y,w,h))
    # 排序並分行
    boxes = sorted(boxes, key=lambda b:(b[1], b[0]))
    # 以 y 聚類成每一行
    rows = []
    cur = []
    for b in boxes:
        if not cur:
            cur = [b]
            continue
        if abs(b[1] - cur[-1][1]) < 20:  # y 接近算同一行
            cur.append(b)
        else:
            rows.append(sorted(cur, key=lambda t:t[0]))
            cur = [b]
    if cur:
        rows.append(sorted(cur, key=lambda t:t[0]))
    # 過濾「極短行/極短列」
    max_cols = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) >= max_cols*0.7]
    return rows

def ocr_cell(img):
    cfg = "--psm 7 -l chi_sim+eng"
    txt = pytesseract.image_to_string(img, config=cfg)
    return txt.strip()

def extract_table(path):
    img = load_and_upright(path)
    bw = binarize(img)
    rows = detect_cells(bw)
    H, W = bw.shape[:2]
    table = []
    for r in rows:
        row_text = []
        for (x,y,w,h) in r:
            pad = 3
            y0 = max(0, y+pad); y1 = min(H, y+h-pad)
            x0 = max(0, x+pad); x1 = min(W, x+w-pad)
            crop = bw[y0:y1, x0:x1]
            row_text.append(ocr_cell(crop))
        table.append(row_text)
    return table

def headers_and_legal_pairs(table):
    # 假設第一行是「列標題（聲母）」，第一列是「行標題（韻母）」；中間非空即為合法組合
    # 若你的表相反（行/列顛倒），把邏輯互換即可
    col_heads = table[0][1:]  # 第一行去掉左上角
    row_heads = [r[0] for r in table[1:]]  # 其餘行的第一列
    legal = []
    for i, r in enumerate(table[1:]):
        fin = row_heads[i]
        for j, cell in enumerate(r[1:]):
            ini = col_heads[j]
            has_content = len(cell.replace(" ", "")) > 0
            if has_content:
                legal.append((ini, fin))
    return col_heads, row_heads, legal

all_legal = []
for p in IMG_FILES:
    tab = extract_table(p)
    cols, rows, legal = headers_and_legal_pairs(tab)
    all_legal.extend(legal)

# 去重 & 輸出
all_legal = sorted(set(all_legal))
df = pd.DataFrame(all_legal, columns=["initial_raw","final_raw"])
df.to_csv("ruian_legal_pairs_raw.csv", index=False, encoding="utf-8")
print("Saved: ruian_legal_pairs_raw.csv, pairs =", len(df))
