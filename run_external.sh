#!/bin/bash
# 조현병 신경회로 분석 - 외부 접속 실행 스크립트

echo "🧠 조현병 신경회로 분석 - 외부 접속 시작"
echo "=========================================="

# Streamlit 설정
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false
EOF

# Streamlit 백그라운드 실행
echo "1. Streamlit 서버 시작 중..."
python3 -m streamlit run streamlit_app.py &
STREAMLIT_PID=$!
sleep 3

# ngrok 터널 생성
echo "2. ngrok 터널 생성 중..."
ngrok http 8501 &
NGROK_PID=$!
sleep 3

# 공개 URL 표시
echo ""
echo "=========================================="
echo "✅ 서버가 실행되었습니다!"
echo ""
echo "📌 과제 제출용 URL (ngrok):"
echo "   아래 URL을 브라우저에서 열어 공개 주소를 확인하세요:"
echo "   http://localhost:4040"
echo ""
echo "📌 로컬 접속:"
echo "   http://localhost:8501"
echo ""
echo "=========================================="
echo ""
echo "종료하려면 Ctrl+C를 누르세요"

# 대기
wait $STREAMLIT_PID $NGROK_PID
