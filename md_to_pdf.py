#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import markdown2
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import re
import os

# 한글 폰트 등록 시도
korean_font = 'Helvetica'
font_paths = [
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    '/Library/Fonts/AppleGothic.ttf',
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
]

for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            if font_path.endswith('.ttc'):
                pdfmetrics.registerFont(TTFont('Korean', font_path, subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont('Korean', font_path))
            korean_font = 'Korean'
            print(f"Using font: {font_path}")
            break
        except Exception as e:
            continue

# README 읽기
with open('readme.md', 'r', encoding='utf-8') as f:
    content = f.read()

# PDF 생성
pdf_path = 'readme.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                        topMargin=0.75*inch, bottomMargin=0.75*inch,
                        leftMargin=0.75*inch, rightMargin=0.75*inch)

# 스타일 정의
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Korean', fontName=korean_font, fontSize=10, leading=14))
styles.add(ParagraphStyle(name='KoreanTitle', fontName=korean_font, fontSize=18, leading=22, spaceAfter=12, textColor=HexColor('#333333')))
styles.add(ParagraphStyle(name='KoreanHeading', fontName=korean_font, fontSize=14, leading=18, spaceAfter=10, textColor=HexColor('#333333')))
styles.add(ParagraphStyle(name='KoreanSubheading', fontName=korean_font, fontSize=12, leading=16, spaceAfter=8, textColor=HexColor('#666666')))
styles.add(ParagraphStyle(name='CodeBlock', fontName='Courier', fontSize=9, leading=11, leftIndent=20, textColor=HexColor('#000000')))

story = []

# 줄 단위로 처리
lines = content.split('\n')
i = 0
in_code_block = False
code_block = []

while i < len(lines):
    line = lines[i]
    
    # 코드 블록 처리
    if line.strip().startswith('```'):
        if not in_code_block:
            in_code_block = True
            code_block = []
        else:
            # 코드 블록 끝
            if code_block:
                code_text = '\n'.join(code_block)
                story.append(Preformatted(code_text, styles['CodeBlock']))
                story.append(Spacer(1, 0.2*inch))
            in_code_block = False
            code_block = []
        i += 1
        continue
    
    if in_code_block:
        code_block.append(line)
        i += 1
        continue
    
    # 제목 처리
    if line.startswith('# '):
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(line[2:], styles['KoreanTitle']))
    elif line.startswith('## '):
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph(line[3:], styles['KoreanHeading']))
    elif line.startswith('### '):
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(line[4:], styles['KoreanSubheading']))
    elif line.startswith('| ') and '|' in line:
        # 테이블은 간단하게 처리
        story.append(Paragraph(line.replace('|', ' | '), styles['CodeBlock']))
    elif line.strip().startswith('-') or line.strip().startswith('*'):
        # 리스트 항목
        text = line.strip()[1:].strip()
        if text:
            story.append(Paragraph('• ' + text, styles['Korean']))
    elif line.strip():
        # 일반 텍스트
        # 볼드 처리 (**text**)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        # 코드 처리 (`text`)
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        story.append(Paragraph(text, styles['Korean']))
    else:
        # 빈 줄
        story.append(Spacer(1, 0.1*inch))
    
    i += 1

# PDF 빌드
doc.build(story)
print(f"✓ PDF 생성 완료: {pdf_path}")
