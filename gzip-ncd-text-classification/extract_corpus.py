"""ブログの全既存記事から frontmatter(title, tags, series)と
本文プレーンテキストを抽出し、corpus.json に保存する。
(執筆中の gzip-ncd-text-classification 自身は除外)"""
import json
import re
from pathlib import Path

BLOG_DIR = Path(r"C:\Users\kouta\project\forsubwork\src\content\blog")

EXCLUDE = {
    "gzip-ncd-text-classification",  # 自分自身(執筆中)は除外
}


def parse_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    fm_text, body = m.group(1), m.group(2)
    fm = {}
    tm = re.search(r"^title:\s*(.+)$", fm_text, re.MULTILINE)
    if tm:
        fm["title"] = tm.group(1).strip().strip("'\"")
    tagm = re.search(r"^tags:\s*\[(.*?)\]", fm_text, re.MULTILINE)
    if tagm:
        raw = tagm.group(1)
        fm["tags"] = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    else:
        fm["tags"] = []
    sm = re.search(r"^series:\s*(.+)$", fm_text, re.MULTILINE)
    fm["series"] = sm.group(1).strip().strip("'\"") if sm else None
    return fm, body


def clean_body(body):
    # 前回のセルフ監査記事(blog-corpus-self-audit-umap)と同じ前処理
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r" \1 ", body)
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r" \1 ", body)
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"[#*_>-]", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def main():
    records = []
    for path in sorted(BLOG_DIR.glob("*.md")):
        slug = path.stem
        if slug in EXCLUDE:
            continue
        text = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        clean = clean_body(body)
        if len(clean) < 20:
            print(f"WARNING: very short body for {slug} ({len(clean)} chars)")
        records.append({
            "slug": slug,
            "title": fm.get("title", slug),
            "tags": fm.get("tags", []),
            "series": fm.get("series"),
            "body": clean,
        })

    out = Path(r"C:\tmp\gzip-ncd-text-classification-experiment\corpus.json")
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(records)}件の記事を抽出 -> {out}")

    from collections import Counter
    tag_counter = Counter()
    for r in records:
        for t in r["tags"]:
            tag_counter[t] += 1
    print("タグ分布:", tag_counter.most_common())
    series_counter = Counter(r["series"] or "(なし)" for r in records)
    print("series分布:", series_counter.most_common())


if __name__ == "__main__":
    main()
