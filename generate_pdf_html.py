#!/usr/bin/env python3
"""
AI 성능 엔지니어링 - 한국어 문서 HTML 생성기
브라우저에서 열어 Ctrl+P → PDF 저장으로 PDF를 만들 수 있습니다.
"""
import os
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))

CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'NanumBarunGothic', '나눔바른고딕', 'NanumGothic', '나눔고딕', 'Malgun Gothic', '맑은 고딕', sans-serif;
    font-size: 14px;
    line-height: 1.8;
    color: #1a1a2e;
    background: #fff;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 60px;
  }

  /* 표지 */
  .cover {
    text-align: center;
    padding: 120px 40px;
    page-break-after: always;
    border-bottom: 3px solid #0f3460;
    margin-bottom: 60px;
  }
  .cover h1 { font-size: 2.4em; color: #0f3460; margin-bottom: 16px; }
  .cover .subtitle { font-size: 1.1em; color: #444; margin-bottom: 8px; }
  .cover .meta { font-size: 0.9em; color: #888; margin-top: 40px; }

  /* 목차 */
  .toc { page-break-after: always; margin-bottom: 60px; }
  .toc h2 { font-size: 1.6em; color: #0f3460; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-bottom: 20px; }
  .toc ol { padding-left: 24px; }
  .toc li { margin: 6px 0; font-size: 0.95em; }
  .toc a { color: #16213e; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .toc .toc-section { font-weight: 700; margin-top: 12px; margin-bottom: 4px; }

  /* 챕터 섹션 */
  .chapter {
    page-break-before: always;
    padding-top: 20px;
    margin-bottom: 60px;
  }
  .chapter-badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    font-size: 0.75em;
    padding: 3px 10px;
    border-radius: 12px;
    margin-bottom: 10px;
    letter-spacing: 0.05em;
  }

  h1 { font-size: 1.9em; color: #0f3460; margin: 24px 0 12px; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }
  h2 { font-size: 1.4em; color: #16213e; margin: 20px 0 10px; }
  h3 { font-size: 1.15em; color: #1a237e; margin: 16px 0 8px; }

  p { margin: 8px 0; }

  /* 표 */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 0.9em;
  }
  th {
    background: #0f3460;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 8px 12px;
    border-bottom: 1px solid #e0e0e0;
    vertical-align: top;
  }
  tr:nth-child(even) td { background: #f5f7ff; }

  /* 코드 */
  code {
    font-family: 'Noto Sans Mono', 'Consolas', monospace;
    background: #f0f4f8;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.88em;
  }
  pre {
    background: #1a1a2e;
    color: #e8eaf6;
    padding: 16px 20px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 12px 0;
    font-size: 0.85em;
    line-height: 1.6;
    font-family: 'NanumGothicCoding', '나눔고딕코딩', 'Consolas', monospace;
  }
  pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
    font-family: inherit;
  }

  /* 리스트 */
  ul, ol { padding-left: 24px; margin: 8px 0; }
  li { margin: 4px 0; }

  /* 구분선 */
  hr { border: none; border-top: 1px solid #e0e0e0; margin: 24px 0; }

  /* 강조 박스 */
  blockquote {
    border-left: 4px solid #0f3460;
    padding: 12px 16px;
    margin: 16px 0;
    background: #f0f4ff;
    color: #333;
    border-radius: 0 6px 6px 0;
  }

  /* 섹션 구분 */
  .section-divider {
    text-align: center;
    color: #aaa;
    margin: 40px 0;
    font-size: 1.4em;
    letter-spacing: 8px;
  }

  .appendix-section { page-break-before: always; }

  @media print {
    body { padding: 20px 40px; max-width: 100%; }
    .chapter { page-break-before: always; }
    pre { white-space: pre-wrap; word-break: break-all; }
  }
</style>
"""

def read_md(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return f.read()
    return ""

def md_to_html(text):
    return markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'codehilite'],
        extension_configs={'codehilite': {'noclasses': True, 'linenums': False}}
    )

def build_html():
    sections = []

    # ── 표지 ──────────────────────────────────────────────────────────────
    sections.append("""
<div class="cover">
  <h1>🚀 AI 성능 엔지니어링</h1>
  <p class="subtitle">AI Systems Performance Engineering</p>
  <p class="subtitle">GPU 최적화, 분산 학습, 추론 스케일링, 풀스택 성능 튜닝</p>
  <p class="subtitle">— O'Reilly 도서 한국어 문서 모음 —</p>
  <p class="meta">저자: Chris Fregly &nbsp;|&nbsp; 출판: O'Reilly Media, November 2025<br>
  한국어 번역: Claude Code (Anthropic)</p>
</div>
""")

    # ── 목차 ──────────────────────────────────────────────────────────────
    toc_items = [
        ("개요", "overview"),
        ("챕터 01 – 성능 기초", "ch01"),
        ("챕터 02 – GPU 하드웨어 아키텍처", "ch02"),
        ("챕터 03 – 시스템 튜닝 (OS/Docker/Kubernetes)", "ch03"),
        ("챕터 04 – 다중 GPU 분산", "ch04"),
        ("챕터 05 – 스토리지 및 I/O 최적화", "ch05"),
        ("챕터 06 – CUDA 프로그래밍 기초", "ch06"),
        ("챕터 07 – 메모리 접근 패턴", "ch07"),
        ("챕터 08 – 점유율 및 파이프라인 튜닝", "ch08"),
        ("챕터 09 – 산술 강도 및 커널 퓨전", "ch09"),
        ("챕터 10 – 텐서 코어 파이프라인 및 클러스터 기능", "ch10"),
        ("챕터 11 – 스트림 및 동시성", "ch11"),
        ("챕터 12 – CUDA 그래프 및 동적 워크로드", "ch12"),
        ("챕터 13 – PyTorch 프로파일링 및 메모리 튜닝", "ch13"),
        ("챕터 14 – 컴파일러 및 Triton 최적화", "ch14"),
        ("챕터 15 – 분리된 추론 및 KV 관리", "ch15"),
        ("챕터 16 – 프로덕션 추론 최적화", "ch16"),
        ("챕터 17 – 동적 라우팅 및 하이브리드 서빙", "ch17"),
        ("챕터 18 – 고급 어텐션 및 디코딩", "ch18"),
        ("챕터 19 – 저정밀 학습 및 메모리 시스템", "ch19"),
        ("챕터 20 – 종합 케이스 스터디", "ch20"),
        ("부록 – 200개 이상의 성능 체크리스트 (영문)", "appendix"),
    ]
    toc_html = '<div class="toc"><h2>목차</h2><ol>\n'
    for label, anchor in toc_items:
        toc_html += f'  <li><a href="#{anchor}">{label}</a></li>\n'
    toc_html += '</ol></div>\n'
    sections.append(toc_html)

    # ── 개요 (루트 README) ────────────────────────────────────────────────
    readme_ko = read_md(os.path.join(BASE, 'README_ko.md'))
    sections.append(f'<div class="chapter" id="overview">\n{md_to_html(readme_ko)}\n</div>')

    # ── 챕터 01–20 ───────────────────────────────────────────────────────
    chapter_meta = [
        ("ch01", "성능 기초"),
        ("ch02", "GPU 하드웨어 아키텍처"),
        ("ch03", "시스템 튜닝 (OS/Docker/Kubernetes)"),
        ("ch04", "다중 GPU 분산"),
        ("ch05", "스토리지 및 I/O 최적화"),
        ("ch06", "CUDA 프로그래밍 기초"),
        ("ch07", "메모리 접근 패턴"),
        ("ch08", "점유율 및 파이프라인 튜닝"),
        ("ch09", "산술 강도 및 커널 퓨전"),
        ("ch10", "텐서 코어 파이프라인 및 클러스터 기능"),
        ("ch11", "스트림 및 동시성"),
        ("ch12", "CUDA 그래프 및 동적 워크로드"),
        ("ch13", "PyTorch 프로파일링 및 메모리 튜닝"),
        ("ch14", "컴파일러 및 Triton 최적화"),
        ("ch15", "분리된 추론 및 KV 관리"),
        ("ch16", "프로덕션 추론 최적화"),
        ("ch17", "동적 라우팅 및 하이브리드 서빙"),
        ("ch18", "고급 어텐션 및 디코딩"),
        ("ch19", "저정밀 학습 및 메모리 시스템"),
        ("ch20", "종합 케이스 스터디"),
    ]

    for ch_id, ch_title in chapter_meta:
        path = os.path.join(BASE, 'code', ch_id, 'README_ko.md')
        content = read_md(path)
        num = ch_id[2:]
        badge = f'<span class="chapter-badge">챕터 {num}</span>'
        sections.append(
            f'<div class="chapter" id="{ch_id}">\n{badge}\n{md_to_html(content)}\n</div>'
        )

    # ── 부록 (영문) ──────────────────────────────────────────────────────
    appendix_md = read_md(os.path.join(BASE, 'docs', 'appendix.md'))
    sections.append(f'''
<div class="appendix-section" id="appendix">
  <blockquote>
    📌 <strong>부록</strong>은 원문(영어) 그대로 포함되어 있습니다.
    O'Reilly 도서에서 발췌한 200개 이상의 성능 체크리스트입니다.
  </blockquote>
  {md_to_html(appendix_md)}
</div>
''')

    # ── 최종 HTML 조립 ────────────────────────────────────────────────────
    body = '\n'.join(sections)
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI 성능 엔지니어링 – 한국어 문서</title>
  {CSS}
</head>
<body>
{body}
</body>
</html>"""
    return html


if __name__ == '__main__':
    out_path = os.path.join(BASE, 'ai_performance_engineering_ko.html')
    html = build_html()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"✅ 생성 완료: {out_path} ({size_kb} KB)")
    print("📄 브라우저에서 열고 Ctrl+P → 'PDF로 저장'을 선택하세요.")
    print("   (인쇄 설정: 여백 '없음' 또는 '최소', 배경 그래픽 체크)")
