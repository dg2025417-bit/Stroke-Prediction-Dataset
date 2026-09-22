import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 기본 설정 (브라우저 탭 제목, 아이콘, 레이아웃)
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# 데이터 불러오기 함수
# (캐시를 사용해서 매번 새로 다운로드하지 않도록 함)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# -----------------------------
# 앱 제목 (아이콘 포함)
# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("뇌졸중(stroke) 데이터를 살펴보고 예측 모델을 만들어보는 실습 공간입니다.")

st.divider()

# -----------------------------
# 큰 숫자 카드 4개: 전체 인원수, 열 개수, stroke=1 인원수, 비율
# -----------------------------
st.subheader("📊 데이터 한눈에 보기")

total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열(컬럼) 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# -----------------------------
# 열 설명 표 만들기
# 열 이름 / 우리말 뜻(빈칸) / 값의 종류 / 빈 값 개수
# -----------------------------
st.subheader("📋 열(컬럼) 설명")

# 각 열의 '값의 종류'를 요약해서 보여주기 위한 함수
def summarize_values(col):
    unique_vals = df[col].dropna().unique()
    # 값 종류가 너무 많으면 (예: id, age, bmi 같은 연속형) 개수만 표시
    if len(unique_vals) > 10:
        return f"연속형 값 (고유값 {len(unique_vals)}개)"
    else:
        # 값 종류가 적으면 실제 값들을 정렬해서 보여주기
        try:
            sorted_vals = sorted(unique_vals)
        except TypeError:
            sorted_vals = unique_vals
        return ", ".join(str(v) for v in sorted_vals)

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],  # 학생이 직접 채워 넣을 빈 칸
    "값의 종류": [summarize_values(col) for col in df.columns],
    "빈 값 개수": [df[col].isnull().sum() for col in df.columns]
})

st.dataframe(column_info, use_container_width=True, hide_index=True)

st.info("💡 '우리말 뜻' 칸은 교재를 참고해서 직접 채워 넣어 보세요!")

st.divider()

# -----------------------------
# 데이터 처음 5줄 보여주기
# -----------------------------
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# -----------------------------
# 데이터 출처 (학생이 직접 작성)
# -----------------------------
st.subheader("📚 데이터 출처")

source_text = st.text_area(
    "교재에 나온 데이터 출처를 이곳에 적어보세요.",
    placeholder="예: 이 데이터는 ○○에서 제공하는 자료이며..."
)

if source_text:
    st.success("작성한 출처 내용:")
    st.write(source_text)
else:
    st.warning("아직 출처를 작성하지 않았어요. 위 칸에 입력해보세요!")
