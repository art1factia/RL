import markdown
from weasyprint import HTML
import sys

# Read markdown file
with open('readme.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML with CSS styling
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
    </style>
</head>
<body>
{markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])}
</body>
</html>
"""

# Convert HTML to PDF
HTML(string=html_content).write_pdf('readme.pdf')
print("✓ Successfully converted readme.md to readme.pdf")
