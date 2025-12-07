#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import base64
from pathlib import Path

# README 읽기
with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 이미지를 Base64로 인코딩하는 함수
def image_to_base64(image_path):
    """이미지 파일을 Base64 문자열로 변환"""
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }
        mime = mime_types.get(ext, 'image/png')
        b64 = base64.b64encode(data).decode('utf-8')
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Warning: Failed to encode {image_path}: {e}")
        return image_path

# GitHub 스타일 HTML 템플릿 (들여쓰기 완전 제거)
html_template = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Particle Control</title>
    <style>
        @page {
            margin: 1.5cm;
            size: A4;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕", sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px 20px;
            font-size: 14px;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 600;
            line-height: 1.25;
        }
        h1 {
            border-bottom: 3px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 1.8em;
        }
        h2 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 1.4em;
        }
        h3 {
            font-size: 1.2em;
        }
        h4 {
            font-size: 1.1em;
        }
        code {
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            font-family: "SF Mono", Monaco, Menlo, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 85%;
            margin: 0;
            padding: 0.2em 0.4em;
        }
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            font-family: "SF Mono", Monaco, Menlo, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 85%;
            line-height: 1.45;
            overflow: auto;
            padding: 12px;
            margin-top: 0;
            margin-bottom: 12px;
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
            margin-bottom: 12px;
            font-size: 13px;
        }
        table th {
            font-weight: 600;
            padding: 6px 10px;
            border: 1px solid #dfe2e5;
            background-color: #f6f8fa;
        }
        table td {
            padding: 6px 10px;
            border: 1px solid #dfe2e5;
        }
        table tr {
            background-color: #fff;
            border-top: 1px solid #c6cbd1;
        }
        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }
        /* 들여쓰기 완전 제거 */
        ul, ol {
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 0;
            list-style-position: inside;
        }
        ul ul, ul ol, ol ul, ol ol {
            padding-left: 0;
            margin-bottom: 0;
            margin-left: 0;
        }
        li {
            margin-bottom: 4px;
            padding-left: 0;
        }
        li + li {
            margin-top: 0.25em;
        }
        p {
            margin-top: 0;
            margin-bottom: 12px;
        }
        strong {
            font-weight: 600;
        }
        hr {
            height: 0.25em;
            padding: 0;
            margin: 20px 0;
            background-color: #e1e4e8;
            border: 0;
        }
        img {
            max-width: 100%;
            box-sizing: content-box;
            background-color: #fff;
            border: 1px solid #dfe2e5;
            border-radius: 3px;
            padding: 8px;
            margin: 12px 0;
        }
        @media print {
            body {
                max-width: 100%;
                padding: 10px;
                font-size: 12px;
            }
            h1 {
                font-size: 1.6em;
            }
            h2 {
                font-size: 1.3em;
            }
            h3 {
                font-size: 1.15em;
            }
            img {
                page-break-inside: avoid;
                max-width: 90%;
            }
            table {
                font-size: 11px;
            }
            pre {
                font-size: 10px;
            }
        }
    </style>
</head>
<body>
'''

# Markdown을 HTML로 변환
lines = content.split('\n')
html_content = []
in_code_block = False
in_table = False
in_list = False
code_lang = ''

i = 0
while i < len(lines):
    line = lines[i]
    
    # 코드 블록
    if line.strip().startswith('```'):
        if not in_code_block:
            code_lang = line.strip()[3:].strip()
            html_content.append('<pre><code>')
            in_code_block = True
        else:
            html_content.append('</code></pre>')
            in_code_block = False
        i += 1
        continue
    
    if in_code_block:
        html_content.append(line.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;'))
        i += 1
        continue
    
    # 이미지 (Markdown 형식: ![alt](path))
    img_match = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
    if img_match:
        alt_text = img_match.group(1)
        img_path = img_match.group(2)
        # Base64로 인코딩
        if Path(img_path).exists():
            img_data = image_to_base64(img_path)
            html_content.append(f'<p><img src="{img_data}" alt="{alt_text}" /></p>')
        else:
            html_content.append(f'<p><img src="{img_path}" alt="{alt_text}" /></p>')
        i += 1
        continue
    
    # 테이블
    if line.startswith('|'):
        if not in_table:
            html_content.append('<table>')
            in_table = True
        
        if '---' in line:
            i += 1
            continue
        
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        # 첫 번째 테이블 행인지 확인 (헤더)
        # 다음 줄이 구분선인지 확인
        is_header = False
        if i + 1 < len(lines) and '---' in lines[i + 1]:
            is_header = True
            
        if is_header:
            html_content.append('<thead><tr>')
            for cell in cells:
                # 볼드, 코드 처리
                cell = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                cell = re.sub(r'`(.*?)`', r'<code>\1</code>', cell)
                html_content.append(f'<th>{cell}</th>')
            html_content.append('</tr></thead><tbody>')
        else:
            html_content.append('<tr>')
            for cell in cells:
                # 볼드, 코드 처리
                cell = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell)
                cell = re.sub(r'`(.*?)`', r'<code>\1</code>', cell)
                html_content.append(f'<td>{cell}</td>')
            html_content.append('</tr>')
        i += 1
        continue
    elif in_table:
        html_content.append('</tbody></table>')
        in_table = False
    
    # 수평선
    if line.strip() == '---':
        html_content.append('<hr>')
        i += 1
        continue
    
    # 제목
    if line.startswith('# '):
        html_content.append(f'<h1>{line[2:]}</h1>')
        i += 1
        continue
    elif line.startswith('## '):
        html_content.append(f'<h2>{line[3:]}</h2>')
        i += 1
        continue
    elif line.startswith('### '):
        html_content.append(f'<h3>{line[4:]}</h3>')
        i += 1
        continue
    elif line.startswith('#### '):
        html_content.append(f'<h4>{line[5:]}</h4>')
        i += 1
        continue
    
    # 리스트
    if line.strip().startswith('- ') or line.strip().startswith('* '):
        if not in_list:
            html_content.append('<ul>')
            in_list = True
        text = line.strip()[2:]
        # 볼드, 코드 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        html_content.append(f'<li>{text}</li>')
        i += 1
        continue
    elif in_list and not line.strip().startswith('-') and not line.strip().startswith('*'):
        html_content.append('</ul>')
        in_list = False
    
    # 숫자 리스트
    numbered_match = re.match(r'^\d+\.\s+(.+)', line.strip())
    if numbered_match:
        if not in_list:
            html_content.append('<ol>')
            in_list = True
        text = numbered_match.group(1)
        # 볼드, 코드 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        html_content.append(f'<li>{text}</li>')
        i += 1
        continue
    
    # 일반 텍스트
    if line.strip():
        text = line
        # 볼드, 코드, 링크 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        html_content.append(f'<p>{text}</p>')
    else:
        # 빈 줄은 무시 (이미 p 태그가 margin을 가지고 있음)
        pass
    
    i += 1

if in_table:
    html_content.append('</tbody></table>')
if in_list:
    html_content.append('</ul>')

html = html_template + '\n'.join(html_content) + '\n</body>\n</html>'

# HTML 저장
output_file = 'README.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ HTML 파일 생성 완료: {output_file}")
print("✓ 모든 들여쓰기 제거 완료 (padding-left: 0, list-style-position: inside)")
print("✓ 이미지가 Base64로 인코딩되어 포함되었습니다.")
print("\n브라우저에서 열어 Cmd+P를 눌러 PDF로 저장하세요:")
print(f"  open {output_file}")
