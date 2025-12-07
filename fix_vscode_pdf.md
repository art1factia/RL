# VS Code "no active editor" 문제 해결

## 방법 1: 파일을 제대로 열기
1. VS Code에서 readme.md를 **더블클릭**하여 엽니다 (한 번만 클릭하면 미리보기 모드)
2. 탭 제목이 기울임꼴이 아닌 **일반체**로 표시되는지 확인
3. Cmd+Shift+P → "Markdown PDF: Export (pdf)" 실행

## 방법 2: 확장 프로그램 재설치
1. VS Code에서 확장 탭 열기 (Cmd+Shift+X)
2. "Markdown PDF" 검색
3. 제거 후 재설치

## 방법 3: VS Code 재시작
1. VS Code 완전히 종료
2. 다시 열고 readme.md 열기
3. PDF 변환 시도

## 방법 4: 명령줄로 확인
VS Code가 파일을 인식하는지 확인:
```bash
code readme.md
```
