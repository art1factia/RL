#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# README 읽기
with open('readme.md', 'r', encoding='utf-8') as f:
    content = f.read()

# GitHub 스타일 HTML 생성
html_template = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Particle Control</title>
    <style>
        @page {
            margin: 2cm;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕", sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            font-size: 14px;
        }
        h1 {
            border-bottom: 3px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 2em;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h2 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 1.5em;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h3 {
            font-size: 1.25em;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        code {
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
            font-size: 85%;
            margin: 0;
            padding: 0.2em 0.4em;
        }
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
            font-size: 85%;
            line-height: 1.45;
            overflow: auto;
            padding: 16px;
        }
        pre code {
            background-color: transparent;
            border: 0;
            display: inline;
            line-height: inherit;
            margin: 0;
            overflow: visible;
            padding: 0;
            word-wrap: normal;
        }
        table {
            border-collapse: collapse;
            border-spacing: 0;
            width: 100%;
            margin-top: 0;
            margin-bottom: 16px;
        }
        table th {
            font-weight: 600;
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
            background-color: #f6f8fa;
        }
        table td {
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }
        table tr {
            background-color: #fff;
            border-top: 1px solid #c6cbd1;
        }
        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }
        ul, ol {
            margin-top: 0;
            margin-bottom: 16px;
            padding-left: 2em;
        }
        li {
            margin-top: 0.25em;
        }
        p {
            margin-top: 0;
            margin-bottom: 16px;
        }
        strong {
            font-weight: 600;
        }
        @media print {
            body {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
'''

# 간단한 마크다운 변환
import re

lines = content.split('\n')
html_content = []
in_code_block = False
in_table = False
code_lang = ''

for line in lines:
    # 코드 블록
    if line.strip().startswith('```'):
        if not in_code_block:
            code_lang = line.strip()[3:].strip()
            html_content.append('<pre><code>')
            in_code_block = True
        else:
            html_content.append('</code></pre>')
            in_code_block = False
        continue
    
    if in_code_block:
        html_content.append(line.replace('<', '&lt;').replace('>', '&gt;'))
        continue
    
    # 테이블
    if line.startswith('|'):
        if not in_table:
            html_content.append('<table>')
            in_table = True
        
        if '---' in line:
            continue
        
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if html_content[-1] == '<table>':
            html_content.append('<thead><tr>')
            for cell in cells:
                html_content.append(f'<th>{cell}</th>')
            html_content.append('</tr></thead><tbody>')
        else:
            html_content.append('<tr>')
            for cell in cells:
                html_content.append(f'<td>{cell}</td>')
            html_content.append('</tr>')
        continue
    elif in_table:
        html_content.append('</tbody></table>')
        in_table = False
    
    # 제목
    if line.startswith('# '):
        html_content.append(f'<h1>{line[2:]}</h1>')
    elif line.startswith('## '):
        html_content.append(f'<h2>{line[3:]}</h2>')
    elif line.startswith('### '):
        html_content.append(f'<h3>{line[4:]}</h3>')
    # 리스트
    elif line.strip().startswith('- ') or line.strip().startswith('* '):
        text = line.strip()[2:]
        # 볼드 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # 코드 처리
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        html_content.append(f'<ul><li>{text}</li></ul>')
    # 일반 텍스트
    elif line.strip():
        text = line
        # 볼드 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # 코드 처리
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        html_content.append(f'<p>{text}</p>')
    else:
        html_content.append('<br>')

if in_table:
    html_content.append('</tbody></table>')

html = html_template + '\n'.join(html_content) + '\n</body>\n</html>'

# HTML 저장
with open('readme_fixed.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✓ HTML 파일 생성 완료: readme_fixed.html")
print("브라우저에서 열어 Cmd+P를 눌러 PDF로 저장하세요.")
