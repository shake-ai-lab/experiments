"""ブログの全既存記事(このバッチで同時執筆中の6記事は除く)から
frontmatter(title, tags, series)と本文プレーンテキストを抽出し、
corpus.json に保存する。"""
import json
import re
from pathlib import Path

BLOG_DIR = Path(r"C:\Users\kouta\project\forsubwork\src\content\blog")

# 同じバッチで同時執筆中の兄弟記事(このセルフ監査の対象には含めない)
EXCLUDE = {
    "stanford-bunny-normal-estimation",
    "logistic-map-butterfly-effect",
    "dropout-implicit-ensemble-test",
    "learning-rate-warmup-ablation",
    "condorcet-jury-theorem",
    "kde-vs-histogram-bandwidth",
    "blog-corpus-self-audit-umap",  # 自分自身も除外
}


def parse_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    fm_text, body = m.group(1), m.group(2)
    fm = {}
    # title
    tm = re.search(r"^title:\s*(.+)$", fm_text, re.MULTILINE)
    if tm:
        fm["title"] = tm.group(1).strip().strip("'\"")
    # tags: ['a', 'b']
    tagm = re.search(r"^tags:\s*\[(.*?)\]", fm_text, re.MULTILINE)
    if tagm:
        raw = tagm.group(1)
        fm["tags"] = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    else:
        fm["tags"] = []
    # series: '...'
    sm = re.search(r"^series:\s*(.+)$", fm_text, re.MULTILINE)
    if sm:
        fm["series"] = sm.group(1).strip().strip("'\"")
    else:
        fm["series"] = None
    # pubDate
    pm = re.search(r"^pubDate:\s*(.+)$", fm_text, re.MULTILINE)
    if pm:
        fm["pubDate"] = pm.group(1).strip().strip("'\"")
    return fm, body


def clean_body(body):
    # コードフェンス除去
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    # iframe/htmlタグ除去
    body = re.sub(r"<[^>]+>", " ", body)
    # 画像構文 ![alt](path) -> alt文だけ残す(説明文なので有用な自然言語)
    body = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r" \1 ", body)
    # 通常リンク [text](url) -> text
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r" \1 ", body)
    # インラインコード `code`
    body = re.sub(r"`[^`]*`", " ", body)
    # 見出し記号、強調記号
    body = re.sub(r"[#*_>-]", " ", body)
    # 空白正規化
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
            "pubDate": fm.get("pubDate"),
            "body": clean,
        })

    out = Path(r"C:\tmp\blog-corpus-self-audit-umap-experiment\corpus.json")
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(records)}件の記事を抽出 -> {out}")

    # tags/seriesの分布をざっと確認
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
