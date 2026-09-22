import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# 데이터 불러오기 함수 (첫 화면과 동일한 데이터)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🔍 데이터 탐색")
st.markdown("뇌졸중 데이터를 여러 각도에서 살펴보는 페이지입니다.")

st.divider()

# -----------------------------
# 1. 나이와 평균 혈당 분포 히스토그램 (나란히)
# -----------------------------
st.subheader("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이", "count": "사람 수"}
    )
    fig_age.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    fig_glucose.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# -----------------------------
# 2. 뇌졸중 유무에 따른 나이/혈당 상자그림 + 평균값 표
# -----------------------------
st.subheader("2️⃣ 뇌졸중 유무에 따른 나이·혈당 비교")

# stroke 값을 사람이 읽기 쉬운 문자열로 바꾼 열 추가
df["뇌졸중_여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df, x="뇌졸중_여부", y="age",
        color="뇌졸중_여부",
        title="뇌졸중 유무별 나이 분포",
        labels={"age": "나이", "뇌졸중_여부": "뇌졸중 여부"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df, x="뇌졸중_여부", y="avg_glucose_level",
        color="뇌졸중_여부",
        title="뇌졸중 유무별 평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "뇌졸중_여부": "뇌졸중 여부"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
mean_table = df.groupby("뇌졸중_여부")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.markdown("**그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 막대그래프
# -----------------------------
st.subheader("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyper_ratio["stroke"] = hyper_ratio["stroke"] * 100

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension", y="stroke",
        title="고혈압 유무별 뇌졸중 비율",
        labels={"hypertension": "고혈압 여부", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100

    fig_heart = px.bar(
        heart_ratio, x="heart_disease", y="stroke",
        title="심장병 유무별 뇌졸중 비율",
        labels={"heart_disease": "심장병 여부", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 4. bmi 결측치 인원의 뇌졸중 비율 vs 전체 뇌졸중 비율
# -----------------------------
st.subheader("4️⃣ 체질량지수(bmi) 결측 여부에 따른 뇌졸중 비율")

bmi_missing_df = df[df["bmi"].isnull()]
bmi_missing_count = len(bmi_missing_df)
bmi_missing_stroke_ratio = bmi_missing_df["stroke"].mean() * 100 if bmi_missing_count > 0 else 0
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측치인 사람", "전체 사람"],
    "인원 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.subheader("5️⃣ 흡연 상태(smoking_status)별 사람 수")

smoking_count_table = df["smoking_status"].value_counts().reset_index()
smoking_count_table.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count_table, use_container_width=True, hide_index=True)
